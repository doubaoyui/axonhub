package responses

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/samber/lo"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
)

func applyImageGenerationMetadataToItem(item *Item, metadata map[string]any) {
	if item == nil || len(metadata) == 0 {
		return
	}

	if action, ok := metadata["action"].(string); ok && action != "" {
		item.Action = action
	}
	if background, ok := metadata["background"].(string); ok && background != "" {
		item.Background = lo.ToPtr(background)
	}
	if outputFormat, ok := metadata["output_format"].(string); ok && outputFormat != "" {
		item.OutputFormat = lo.ToPtr(outputFormat)
	}
	if quality, ok := metadata["quality"].(string); ok && quality != "" {
		item.Quality = lo.ToPtr(quality)
	}
	if size, ok := metadata["size"].(string); ok && size != "" {
		item.Size = lo.ToPtr(size)
	}
	if revisedPrompt, ok := metadata["revised_prompt"].(string); ok && revisedPrompt != "" {
		item.RevisedPrompt = lo.ToPtr(revisedPrompt)
	}
}

// TransformStream transforms the unified llm.Response stream to OpenAI Responses API SSE events.
func (t *InboundTransformer) TransformStream(
	ctx context.Context,
	stream streams.Stream[*llm.Response],
) (streams.Stream[*httpclient.StreamEvent], error) {
	return &responsesInboundStream{
		source:                   stream,
		ctx:                      ctx,
		toolCalls:                make(map[int]*llm.ToolCall),
		imageGenerationItemUpdates: make(map[string]map[string]any),
	}, nil
}

// responsesInboundStream implements the stateful stream transformation.
//
//nolint:containedctx // Checked.
type responsesInboundStream struct {
	source streams.Stream[*llm.Response]
	ctx    context.Context

	// State tracking
	hasStarted              bool
	hasResponseCreated      bool
	hasMessageItemStarted   bool
	hasReasoningItemStarted bool
	hasReasoningSummaryPart bool
	hasContentPartStarted   bool
	hasFinished             bool
	responseCompleted       bool

	// Response metadata
	responseID string
	model      string
	createdAt  int64

	// Content tracking
	outputIndex    int
	contentIndex   int
	sequenceNumber int
	currentItemID  string

	// Content accumulation for items (used for emitting done events)
	accumulatedText               strings.Builder
	accumulatedReasoning          strings.Builder
	accumulatedReasoningSignature strings.Builder
	imageGenerationParts          []llm.MessageContentPart
	imageGenerationItemUpdates    map[string]map[string]any

	// Tool call tracking
	toolCalls           map[int]*llm.ToolCall
	currentToolCallIdx  int
	toolCallItemStarted map[int]bool
	toolCallOutputIndex map[int]int // Maps tool call index to output index

	// Response accumulation using streamAggregator
	usage      *llm.Usage
	aggregator *streamAggregator

	// Event queue
	eventQueue []*httpclient.StreamEvent
	queueIndex int
	err        error
}

func (s *responsesInboundStream) enqueueEvent(ev *StreamEvent) error {
	ev.SequenceNumber = s.sequenceNumber
	s.sequenceNumber++

	eventData, err := json.Marshal(ev)
	if err != nil {
		return err
	}

	streamEvent := &httpclient.StreamEvent{
		Type: string(ev.Type),
		Data: eventData,
	}

	s.eventQueue = append(s.eventQueue, streamEvent)

	// Use aggregator to accumulate state for response.completed
	if s.aggregator == nil {
		s.aggregator = newStreamAggregator()
	}

	s.aggregator.processEvent(ev)

	return nil
}

//nolint:maintidx,gocognit // It is complex and hard to split.
func (s *responsesInboundStream) Next() bool {
	// If we have events in the queue, return them first
	if s.queueIndex < len(s.eventQueue) {
		return true
	}

	// Clear the queue and reset index for new events
	s.eventQueue = nil
	s.queueIndex = 0

	// Try to get the next chunk from source
	if !s.source.Next() {
		return false
	}

	chunk := s.source.Current()
	if chunk == nil {
		return s.Next() // Try next chunk
	}

	// Handle [DONE] marker
	if chunk.Object == "[DONE]" {
		return s.Next() // Try next chunk
	}

	// Initialize response metadata from first chunk
	if s.responseID == "" && chunk.ID != "" {
		s.responseID = chunk.ID
	}

	if s.model == "" && chunk.Model != "" {
		s.model = chunk.Model
	}

	// Track createdAt
	if s.createdAt == 0 && chunk.Created != 0 {
		s.createdAt = chunk.Created
	}

	// Track usage
	if chunk.Usage != nil {
		s.usage = chunk.Usage
	}

	// Generate response.created event if this is the first chunk
	if !s.hasResponseCreated {
		s.hasResponseCreated = true

		response := &Response{
			Object:    "response",
			ID:        s.responseID,
			Model:     s.model,
			CreatedAt: s.createdAt,
			Status:    lo.ToPtr("in_progress"),
			Output:    []Item{},
		}

		if s.usage != nil {
			response.Usage = ConvertLLMUsageToResponsesUsage(s.usage)
		}

		err := s.enqueueEvent(&StreamEvent{
			Type:     StreamEventTypeResponseCreated,
			Response: response,
		})
		if err != nil {
			s.err = fmt.Errorf("failed to enqueue response.created event: %w", err)
			return false
		}

		// Also emit response.in_progress
		err = s.enqueueEvent(&StreamEvent{
			Type:     StreamEventTypeResponseInProgress,
			Response: response,
		})
		if err != nil {
			s.err = fmt.Errorf("failed to enqueue response.in_progress event: %w", err)
			return false
		}
	}

	// Process choices
	if len(chunk.Choices) > 0 {
		choice := chunk.Choices[0]

		// Handle reasoning content (thinking) delta
		if choice.Delta != nil && choice.Delta.ReasoningContent != nil && *choice.Delta.ReasoningContent != "" {
			if err := s.handleReasoningContent(choice.Delta.ReasoningContent); err != nil {
				s.err = err
				return false
			}
		}

		// Handle encrypted reasoning content delta (stored in ReasoningSignature)
		if choice.Delta != nil && choice.Delta.ReasoningSignature != nil && *choice.Delta.ReasoningSignature != "" {
			// Encrypted reasoning may appear without any summary text. Ensure we still
			// create a reasoning output item so encrypted_content isn't silently dropped.
			if err := s.ensureReasoningItemStarted(); err != nil {
				s.err = err
				return false
			}
			s.accumulatedReasoningSignature.WriteString(*choice.Delta.ReasoningSignature)
		}

		// Handle text content delta
		if choice.Delta != nil && choice.Delta.Content.Content != nil && *choice.Delta.Content.Content != "" {
			if err := s.handleTextContent(choice.Delta.Content.Content); err != nil {
				s.err = err
				return false
			}
		}

		// Handle tool calls
		if choice.Delta != nil && len(choice.Delta.ToolCalls) > 0 {
			if err := s.handleToolCalls(choice.Delta.ToolCalls); err != nil {
				s.err = err
				return false
			}
		}

		if choice.Delta != nil && len(choice.Delta.Content.MultipleContent) > 0 {
			for _, part := range choice.Delta.Content.MultipleContent {
				if part.Type == "image_url" && part.ImageURL != nil && part.ImageURL.URL != "" {
					s.imageGenerationParts = append(s.imageGenerationParts, part)
				}
			}
		}

		if len(choice.TransformerMetadata) > 0 {
			s.applyChoiceTransformerMetadata(choice.TransformerMetadata)
		}

		// Handle finish reason
		if choice.FinishReason != nil && !s.hasFinished {
			s.hasFinished = true

			// Close any open content parts
			if err := s.closeCurrentContentPart(); err != nil {
				s.err = err
				return false
			}

			// Close any open output items
			if err := s.closeCurrentOutputItem(); err != nil {
				s.err = err
				return false
			}

			if err := s.flushImageGenerationParts(); err != nil {
				s.err = err
				return false
			}
		}
	}

	// Handle final usage chunk and complete response
	if chunk.Usage != nil && s.hasFinished && !s.responseCompleted {
		s.responseCompleted = true
		s.usage = chunk.Usage

		// Build final response using aggregator
		s.aggregator.status = "completed"
		response := s.aggregator.buildResponse()
		response.Usage = ConvertLLMUsageToResponsesUsage(s.usage)

		err := s.enqueueEvent(&StreamEvent{
			Type:     StreamEventTypeResponseCompleted,
			Response: response,
		})
		if err != nil {
			s.err = fmt.Errorf("failed to enqueue response.completed event: %w", err)
			return false
		}
	}

	// Continue to the next event
	return s.Next()
}

func (s *responsesInboundStream) handleReasoningContent(content *string) error {
	if err := s.ensureReasoningItemStarted(); err != nil {
		return err
	}

	// Start reasoning summary part only when we actually have summary text.
	if !s.hasReasoningSummaryPart {
		s.hasReasoningSummaryPart = true

		err := s.enqueueEvent(&StreamEvent{
			Type:         StreamEventTypeReasoningSummaryPartAdded,
			ItemID:       &s.currentItemID,
			OutputIndex:  s.outputIndex,
			SummaryIndex: lo.ToPtr(0),
			Part:         &StreamEventContentPart{Type: "summary_text"},
		})
		if err != nil {
			return fmt.Errorf("failed to enqueue reasoning_summary_part.added event: %w", err)
		}
	}

	// Accumulate reasoning content
	s.accumulatedReasoning.WriteString(*content)

	// Emit reasoning_summary_text.delta
	err := s.enqueueEvent(&StreamEvent{
		Type:         StreamEventTypeReasoningSummaryTextDelta,
		ItemID:       &s.currentItemID,
		OutputIndex:  s.outputIndex,
		SummaryIndex: lo.ToPtr(0),
		Delta:        *content,
	})
	if err != nil {
		return fmt.Errorf("failed to enqueue reasoning_summary_text.delta event: %w", err)
	}

	return nil
}

func (s *responsesInboundStream) ensureReasoningItemStarted() error {
	// Start reasoning output item if not started.
	if s.hasReasoningItemStarted {
		return nil
	}

	// Close any previous output item.
	if err := s.closeCurrentOutputItem(); err != nil {
		return err
	}

	s.hasReasoningItemStarted = true
	s.hasReasoningSummaryPart = false

	s.currentItemID = generateItemID()
	item := &Item{
		ID:      s.currentItemID,
		Type:    "reasoning",
		Status:  lo.ToPtr("in_progress"),
		Summary: []ReasoningSummary{},
	}

	err := s.enqueueEvent(&StreamEvent{
		Type:        StreamEventTypeOutputItemAdded,
		OutputIndex: s.outputIndex,
		Item:        item,
	})
	if err != nil {
		return fmt.Errorf("failed to enqueue output_item.added event: %w", err)
	}

	return nil
}

func (s *responsesInboundStream) handleTextContent(content *string) error {
	if len(s.imageGenerationParts) > 0 {
		if err := s.flushImageGenerationParts(); err != nil {
			return err
		}
	}

	// Close reasoning item if it was started
	if s.hasReasoningItemStarted {
		if err := s.closeReasoningItem(); err != nil {
			return err
		}
	}

	// Start message output item if not started
	if !s.hasMessageItemStarted {
		s.hasMessageItemStarted = true

		s.currentItemID = generateItemID()

		err := s.enqueueEvent(&StreamEvent{
			Type:        StreamEventTypeOutputItemAdded,
			OutputIndex: s.outputIndex,
			Item: &Item{
				ID:      s.currentItemID,
				Type:    "message",
				Status:  lo.ToPtr("in_progress"),
				Role:    "assistant",
				Content: &Input{Items: []Item{}},
			},
		})
		if err != nil {
			return fmt.Errorf("failed to enqueue output_item.added event: %w", err)
		}
	}

	// Start content part if not started
	if !s.hasContentPartStarted {
		s.hasContentPartStarted = true

		err := s.enqueueEvent(&StreamEvent{
			Type:         StreamEventTypeContentPartAdded,
			ItemID:       &s.currentItemID,
			OutputIndex:  s.outputIndex,
			ContentIndex: &s.contentIndex,
			Part: &StreamEventContentPart{
				Type: "output_text",
				Text: lo.ToPtr(""),
			},
		})
		if err != nil {
			return fmt.Errorf("failed to enqueue content_part.added event: %w", err)
		}
	}

	// Accumulate text content
	s.accumulatedText.WriteString(*content)

	// Emit output_text.delta
	err := s.enqueueEvent(&StreamEvent{
		Type:         StreamEventTypeOutputTextDelta,
		ItemID:       &s.currentItemID,
		OutputIndex:  s.outputIndex,
		ContentIndex: &s.contentIndex,
		Delta:        *content,
	})
	if err != nil {
		return fmt.Errorf("failed to enqueue output_text.delta event: %w", err)
	}

	return nil
}

func (s *responsesInboundStream) flushImageGenerationParts() error {
	for _, part := range s.imageGenerationParts {
		if part.Type != "image_url" || part.ImageURL == nil || part.ImageURL.URL == "" {
			continue
		}

		if err := s.closeCurrentOutputItem(); err != nil {
			return err
		}

		s.currentItemID = generateItemID()

		result := part.ImageURL.URL
		if idx := strings.Index(result, "base64,"); idx >= 0 {
			result = result[idx+7:]
		}

		inProgress := &Item{
			ID:     s.currentItemID,
			Type:   "image_generation_call",
			Status: lo.ToPtr("in_progress"),
		}

		if err := s.enqueueEvent(&StreamEvent{
			Type:        StreamEventTypeOutputItemAdded,
			OutputIndex: s.outputIndex,
			Item:        inProgress,
		}); err != nil {
			return fmt.Errorf("failed to enqueue image_generation output_item.added event: %w", err)
		}

		if err := s.enqueueEvent(&StreamEvent{
			Type:        StreamEventTypeImageGenerationInProgress,
			ItemID:      &s.currentItemID,
			OutputIndex: s.outputIndex,
			Status:      "in_progress",
		}); err != nil {
			return fmt.Errorf("failed to enqueue image_generation in_progress event: %w", err)
		}

		if err := s.enqueueEvent(&StreamEvent{
			Type:        StreamEventTypeImageGenerationGenerating,
			ItemID:      &s.currentItemID,
			OutputIndex: s.outputIndex,
			Status:      "generating",
		}); err != nil {
			return fmt.Errorf("failed to enqueue image_generation generating event: %w", err)
		}

		item := &Item{
			ID:     s.currentItemID,
			Type:   "image_generation_call",
			Status: lo.ToPtr("generating"),
			Result: lo.ToPtr(result),
		}

		metadata := make(map[string]any, len(part.TransformerMetadata))
		for key, value := range part.TransformerMetadata {
			metadata[key] = value
		}
		if pending, ok := s.imageGenerationItemUpdates[result]; ok {
			for key, value := range pending {
				metadata[key] = value
			}
		}

		applyImageGenerationMetadataToItem(item, metadata)

		partialImageEvent := &StreamEvent{
			Type:            StreamEventTypeImageGenerationPartialImage,
			ItemID:          &s.currentItemID,
			OutputIndex:     s.outputIndex,
			PartialImageB64: result,
			Status:          "generating",
		}
		if item.Background != nil {
			partialImageEvent.Background = *item.Background
		}
		if item.OutputFormat != nil {
			partialImageEvent.OutputFormat = *item.OutputFormat
		}
		if item.Quality != nil {
			partialImageEvent.Quality = *item.Quality
		}
		if item.Size != nil {
			partialImageEvent.Size = *item.Size
		}
		if item.RevisedPrompt != nil {
			partialImageEvent.RevisedPrompt = *item.RevisedPrompt
		}

		if err := s.enqueueEvent(partialImageEvent); err != nil {
			return fmt.Errorf("failed to enqueue image_generation partial_image event: %w", err)
		}

		if err := s.enqueueEvent(&StreamEvent{
			Type:        StreamEventTypeOutputItemDone,
			OutputIndex: s.outputIndex,
			Item:        item,
		}); err != nil {
			return fmt.Errorf("failed to enqueue image_generation output_item.done event: %w", err)
		}

		s.outputIndex++
	}

	s.imageGenerationParts = nil
	return nil
}

func (s *responsesInboundStream) applyChoiceTransformerMetadata(metadata map[string]any) {
	rawUpdates, ok := metadata["image_generation_item_updates"]
	if !ok {
		return
	}

	updates, ok := rawUpdates.(map[string]any)
	if !ok {
		return
	}

	for _, rawValue := range updates {
		updateMap, ok := rawValue.(map[string]any)
		if !ok {
			continue
		}

		for _, part := range s.imageGenerationParts {
			if part.Type != "image_url" || part.ImageURL == nil || part.ImageURL.URL == "" {
				continue
			}

			url := part.ImageURL.URL
			if idx := strings.Index(url, "base64,"); idx >= 0 {
				url = url[idx+7:]
			}

			dst := s.imageGenerationItemUpdates[url]
			if dst == nil {
				dst = map[string]any{}
				s.imageGenerationItemUpdates[url] = dst
			}
			for key, value := range updateMap {
				dst[key] = value
			}
		}
	}
}

func (s *responsesInboundStream) handleToolCalls(toolCalls []llm.ToolCall) error {
	// Close message item if it was started
	if s.hasMessageItemStarted {
		if err := s.closeMessageItem(); err != nil {
			return err
		}
	}

	// Close reasoning item if it was started
	if s.hasReasoningItemStarted {
		if err := s.closeReasoningItem(); err != nil {
			return err
		}
	}

	if s.toolCallItemStarted == nil {
		s.toolCallItemStarted = make(map[int]bool)
	}

	if s.toolCallOutputIndex == nil {
		s.toolCallOutputIndex = make(map[int]int)
	}

	for _, tc := range toolCalls {
		toolCallIndex := tc.Index

		// Initialize tool call tracking if needed
		if _, ok := s.toolCalls[toolCallIndex]; !ok {
			if err := s.initToolCall(tc); err != nil {
				return err
			}
		}

		// Process delta based on tool type
		switch {
		case tc.ResponseCustomToolCall != nil:
			if err := s.handleCustomToolCallDelta(tc); err != nil {
				return err
			}
		default:
			if err := s.handleFunctionCallDelta(tc); err != nil {
				return err
			}
		}
	}

	return nil
}

func (s *responsesInboundStream) initToolCall(tc llm.ToolCall) error {
	toolCallIndex := tc.Index

	if err := s.closeCurrentContentPart(); err != nil {
		return err
	}

	if err := s.closeCurrentOutputItem(); err != nil {
		return err
	}

	s.toolCalls[toolCallIndex] = &llm.ToolCall{
		Index:                  toolCallIndex,
		ID:                     tc.ID,
		Type:                   tc.Type,
		ResponseCustomToolCall: tc.ResponseCustomToolCall,
		Function: llm.FunctionCall{
			Name:      tc.Function.Name,
			Arguments: "",
		},
	}

	itemID := tc.ID
	if itemID == "" {
		itemID = generateItemID()
	}

	switch {
	case tc.ResponseCustomToolCall != nil:
		item := &Item{
			ID:     itemID,
			Type:   "custom_tool_call",
			Status: lo.ToPtr("in_progress"),
			CallID: tc.ResponseCustomToolCall.CallID,
			Name:   tc.ResponseCustomToolCall.Name,
			Input:  lo.ToPtr(""),
		}

		err := s.enqueueEvent(&StreamEvent{
			Type:        StreamEventTypeOutputItemAdded,
			OutputIndex: s.outputIndex,
			Item:        item,
		})
		if err != nil {
			return fmt.Errorf("failed to enqueue output_item.added event: %w", err)
		}

	default:
		item := &Item{
			ID:     itemID,
			Type:   "function_call",
			Status: lo.ToPtr("in_progress"),
			CallID: tc.ID,
			Name:   tc.Function.Name,
		}

		err := s.enqueueEvent(&StreamEvent{
			Type:        StreamEventTypeOutputItemAdded,
			OutputIndex: s.outputIndex,
			Item:        item,
		})
		if err != nil {
			return fmt.Errorf("failed to enqueue output_item.added event: %w", err)
		}
	}

	s.toolCallItemStarted[toolCallIndex] = true
	s.toolCallOutputIndex[toolCallIndex] = s.outputIndex
	s.currentItemID = itemID
	s.outputIndex++

	return nil
}

func (s *responsesInboundStream) handleFunctionCallDelta(tc llm.ToolCall) error {
	toolCallIndex := tc.Index
	s.toolCalls[toolCallIndex].Function.Arguments += tc.Function.Arguments

	if tc.Function.Arguments != "" {
		itemID := s.toolCalls[toolCallIndex].ID
		if itemID == "" {
			itemID = s.currentItemID
		}

		err := s.enqueueEvent(&StreamEvent{
			Type:         StreamEventTypeFunctionCallArgumentsDelta,
			ItemID:       &itemID,
			OutputIndex:  s.toolCallOutputIndex[toolCallIndex],
			ContentIndex: lo.ToPtr(0),
			Delta:        tc.Function.Arguments,
		})
		if err != nil {
			return fmt.Errorf("failed to enqueue function_call_arguments.delta event: %w", err)
		}
	}

	return nil
}

func (s *responsesInboundStream) handleCustomToolCallDelta(tc llm.ToolCall) error {
	toolCallIndex := tc.Index
	s.toolCalls[toolCallIndex].ResponseCustomToolCall.Input += tc.ResponseCustomToolCall.Input

	if tc.ResponseCustomToolCall.Input != "" {
		itemID := s.toolCalls[toolCallIndex].ID
		if itemID == "" {
			itemID = s.currentItemID
		}

		err := s.enqueueEvent(&StreamEvent{
			Type:        StreamEventTypeCustomToolCallInputDelta,
			ItemID:      &itemID,
			OutputIndex: s.toolCallOutputIndex[toolCallIndex],
			Delta:       tc.ResponseCustomToolCall.Input,
		})
		if err != nil {
			return fmt.Errorf("failed to enqueue custom_tool_call_input.delta event: %w", err)
		}
	}

	return nil
}

func (s *responsesInboundStream) closeReasoningItem() error {
	if !s.hasReasoningItemStarted {
		return nil
	}

	s.hasReasoningItemStarted = false
	fullReasoning := s.accumulatedReasoning.String()
	hadSummaryPart := s.hasReasoningSummaryPart

	// Emit reasoning summary done events only if we started the summary part.
	if hadSummaryPart {
		// Emit reasoning_summary_text.done with accumulated text
		err := s.enqueueEvent(&StreamEvent{
			Type:         StreamEventTypeReasoningSummaryTextDone,
			ItemID:       &s.currentItemID,
			OutputIndex:  s.outputIndex,
			SummaryIndex: lo.ToPtr(0),
			Text:         fullReasoning,
		})
		if err != nil {
			return fmt.Errorf("failed to enqueue reasoning_summary_text.done event: %w", err)
		}

		// Emit reasoning_summary_part.done
		err = s.enqueueEvent(&StreamEvent{
			Type:         StreamEventTypeReasoningSummaryPartDone,
			ItemID:       &s.currentItemID,
			OutputIndex:  s.outputIndex,
			SummaryIndex: lo.ToPtr(0),
			Part: &StreamEventContentPart{
				Type: "summary_text",
				Text: &fullReasoning,
			},
		})
		if err != nil {
			return fmt.Errorf("failed to enqueue reasoning_summary_part.done event: %w", err)
		}
	}
	s.hasReasoningSummaryPart = false

	// Emit output_item.done with complete reasoning item
	var encryptedContent *string
	if s.accumulatedReasoningSignature.Len() > 0 {
		encoded := s.accumulatedReasoningSignature.String()
		encryptedContent = lo.ToPtr(encoded)
	}

	var summary []ReasoningSummary
	if hadSummaryPart {
		summary = []ReasoningSummary{{
			Type: "summary_text",
			Text: fullReasoning,
		}}
	} else {
		summary = []ReasoningSummary{}
	}

	item := Item{
		ID:               s.currentItemID,
		Type:             "reasoning",
		Summary:          summary,
		EncryptedContent: encryptedContent,
	}

	err := s.enqueueEvent(&StreamEvent{
		Type:        StreamEventTypeOutputItemDone,
		OutputIndex: s.outputIndex,
		Item:        &item,
	})
	if err != nil {
		return fmt.Errorf("failed to enqueue output_item.done event: %w", err)
	}

	s.outputIndex++
	s.accumulatedReasoning.Reset()
	s.accumulatedReasoningSignature.Reset()

	return nil
}

func (s *responsesInboundStream) closeMessageItem() error {
	if !s.hasMessageItemStarted {
		return nil
	}

	s.hasMessageItemStarted = false
	fullText := s.accumulatedText.String()

	// Close content part first
	if err := s.closeCurrentContentPart(); err != nil {
		return err
	}

	// Emit output_item.done with complete message content
	item := Item{
		ID:     s.currentItemID,
		Type:   "message",
		Status: lo.ToPtr("completed"),
		Role:   "assistant",
		Content: &Input{
			Items: []Item{{
				Type:        "output_text",
				Text:        &fullText,
				Annotations: []Annotation{},
			}},
		},
	}

	err := s.enqueueEvent(&StreamEvent{
		Type:        StreamEventTypeOutputItemDone,
		OutputIndex: s.outputIndex,
		Item:        &item,
	})
	if err != nil {
		return fmt.Errorf("failed to enqueue output_item.done event: %w", err)
	}

	s.outputIndex++
	s.contentIndex = 0
	s.accumulatedText.Reset()

	return nil
}

func (s *responsesInboundStream) closeCurrentContentPart() error {
	if !s.hasContentPartStarted {
		return nil
	}

	s.hasContentPartStarted = false
	fullText := s.accumulatedText.String()

	// Emit output_text.done with accumulated text
	err := s.enqueueEvent(&StreamEvent{
		Type:         StreamEventTypeOutputTextDone,
		ItemID:       &s.currentItemID,
		OutputIndex:  s.outputIndex,
		ContentIndex: &s.contentIndex,
		Text:         fullText,
	})
	if err != nil {
		return fmt.Errorf("failed to enqueue output_text.done event: %w", err)
	}

	// Emit content_part.done with full text
	err = s.enqueueEvent(&StreamEvent{
		Type:         StreamEventTypeContentPartDone,
		ItemID:       &s.currentItemID,
		OutputIndex:  s.outputIndex,
		ContentIndex: &s.contentIndex,
		Part: &StreamEventContentPart{
			Type: "output_text",
			Text: lo.ToPtr(fullText),
		},
	})
	if err != nil {
		return fmt.Errorf("failed to enqueue content_part.done event: %w", err)
	}

	return nil
}

func (s *responsesInboundStream) closeCurrentOutputItem() error {
	// Close message item if open
	if s.hasMessageItemStarted {
		if err := s.closeMessageItem(); err != nil {
			return err
		}
	}

	// Close reasoning item if open
	if s.hasReasoningItemStarted {
		if err := s.closeReasoningItem(); err != nil {
			return err
		}
	}

	// Close any open tool call items
	for idx, tc := range s.toolCalls {
		if !s.toolCallItemStarted[idx] {
			continue
		}

		itemID := tc.ID
		if itemID == "" {
			itemID = s.currentItemID
		}

		switch {
		case tc.ResponseCustomToolCall != nil:
			// Custom tool call - emit custom_tool_call_input.done then output_item.done
			fullInput := tc.ResponseCustomToolCall.Input

			err := s.enqueueEvent(&StreamEvent{
				Type:        StreamEventTypeCustomToolCallInputDone,
				ItemID:      &itemID,
				OutputIndex: s.toolCallOutputIndex[idx],
				Input:       fullInput,
			})
			if err != nil {
				return fmt.Errorf("failed to enqueue custom_tool_call_input.done event: %w", err)
			}

			item := Item{
				ID:     itemID,
				Type:   "custom_tool_call",
				Status: lo.ToPtr("completed"),
				CallID: tc.ResponseCustomToolCall.CallID,
				Name:   tc.ResponseCustomToolCall.Name,
				Input:  lo.ToPtr(fullInput),
			}

			err = s.enqueueEvent(&StreamEvent{
				Type:        StreamEventTypeOutputItemDone,
				OutputIndex: s.toolCallOutputIndex[idx],
				Item:        &item,
			})
			if err != nil {
				return fmt.Errorf("failed to enqueue output_item.done event: %w", err)
			}

		default:
			// Function call - emit function_call_arguments.done then output_item.done
			err := s.enqueueEvent(&StreamEvent{
				Type:        StreamEventTypeFunctionCallArgumentsDone,
				ItemID:      &itemID,
				OutputIndex: s.toolCallOutputIndex[idx],
				Arguments:   tc.Function.Arguments,
			})
			if err != nil {
				return fmt.Errorf("failed to enqueue function_call_arguments.done event: %w", err)
			}

			item := Item{
				ID:        itemID,
				Type:      "function_call",
				Status:    lo.ToPtr("completed"),
				CallID:    tc.ID,
				Name:      tc.Function.Name,
				Arguments: tc.Function.Arguments,
			}

			err = s.enqueueEvent(&StreamEvent{
				Type:        StreamEventTypeOutputItemDone,
				OutputIndex: s.toolCallOutputIndex[idx],
				Item:        &item,
			})
			if err != nil {
				return fmt.Errorf("failed to enqueue output_item.done event: %w", err)
			}
		}

		s.toolCallItemStarted[idx] = false
	}

	return nil
}

func (s *responsesInboundStream) Current() *httpclient.StreamEvent {
	if s.queueIndex < len(s.eventQueue) {
		event := s.eventQueue[s.queueIndex]
		s.queueIndex++

		return event
	}

	return nil
}

func (s *responsesInboundStream) Err() error {
	if s.err != nil {
		return s.err
	}

	return s.source.Err()
}

func (s *responsesInboundStream) Close() error {
	return s.source.Close()
}

// AggregateStreamChunks aggregates streaming chunks into a complete response body.
func (t *InboundTransformer) AggregateStreamChunks(
	ctx context.Context,
	chunks []*httpclient.StreamEvent,
) ([]byte, llm.ResponseMeta, error) {
	return AggregateStreamChunks(ctx, chunks)
}

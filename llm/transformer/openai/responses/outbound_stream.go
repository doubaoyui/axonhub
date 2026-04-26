package responses

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/samber/lo"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
	"github.com/looplj/axonhub/llm/transformer/shared"
)

func imageGenerationMetadataFromItem(item *Item) map[string]any {
	if item == nil {
		return nil
	}

	metadata := map[string]any{}
	if item.Action != nil {
		metadata["action"] = item.Action
	}
	if item.Background != nil && *item.Background != "" {
		metadata["background"] = *item.Background
	}
	if item.OutputFormat != nil && *item.OutputFormat != "" {
		metadata["output_format"] = *item.OutputFormat
	}
	if item.Quality != nil && *item.Quality != "" {
		metadata["quality"] = *item.Quality
	}
	if item.Size != nil && *item.Size != "" {
		metadata["size"] = *item.Size
	}
	if item.RevisedPrompt != nil && *item.RevisedPrompt != "" {
		metadata["revised_prompt"] = *item.RevisedPrompt
	}
	if len(metadata) == 0 {
		return nil
	}

	return metadata
}

func imageGenerationToolCallFromItem(item *Item, eventType StreamEventType) *llm.ImageGenerationToolCall {
	if item == nil {
		return nil
	}

	return &llm.ImageGenerationToolCall{
		ID:            item.ID,
		EventType:     string(eventType),
		Status:        lo.FromPtr(item.Status),
		Action:        item.Action,
		Result:        lo.FromPtr(item.Result),
		Background:    lo.FromPtr(item.Background),
		OutputFormat:  lo.FromPtr(item.OutputFormat),
		Quality:       lo.FromPtr(item.Quality),
		Size:          lo.FromPtr(item.Size),
		RevisedPrompt: lo.FromPtr(item.RevisedPrompt),
	}
}

func imageGenerationStatusForEvent(eventType StreamEventType, status string) string {
	if status != "" {
		return status
	}

	switch eventType {
	case StreamEventTypeOutputItemAdded, StreamEventTypeImageGenerationInProgress:
		return "in_progress"
	case StreamEventTypeImageGenerationGenerating, StreamEventTypeImageGenerationPartialImage:
		return "generating"
	case StreamEventTypeImageGenerationCompleted:
		return "completed"
	default:
		return ""
	}
}

func applyImageGenerationMetadataToToolCall(toolCall *llm.ImageGenerationToolCall, metadata map[string]any) {
	if toolCall == nil || len(metadata) == 0 {
		return
	}

	if action := metadata["action"]; action != nil {
		toolCall.Action = action
	}
	if background, ok := metadata["background"].(string); ok {
		toolCall.Background = background
	}
	if outputFormat, ok := metadata["output_format"].(string); ok {
		toolCall.OutputFormat = outputFormat
	}
	if quality, ok := metadata["quality"].(string); ok {
		toolCall.Quality = quality
	}
	if size, ok := metadata["size"].(string); ok {
		toolCall.Size = size
	}
	if revisedPrompt, ok := metadata["revised_prompt"].(string); ok {
		toolCall.RevisedPrompt = revisedPrompt
	}
}

func imageGenerationToolCallFromStreamEvent(event *StreamEvent, metadata map[string]any) *llm.ImageGenerationToolCall {
	if event == nil {
		return nil
	}

	toolCall := &llm.ImageGenerationToolCall{
		ID:                lo.FromPtr(event.ItemID),
		EventType:         string(event.Type),
		Status:            imageGenerationStatusForEvent(event.Type, event.Status),
		PartialImageB64:   event.PartialImageB64,
		PartialImageIndex: event.PartialImageIndex,
		Background:        event.Background,
		OutputFormat:      event.OutputFormat,
		Quality:           event.Quality,
		Size:              event.Size,
		RevisedPrompt:     event.RevisedPrompt,
	}
	applyImageGenerationMetadataToToolCall(toolCall, metadata)

	return toolCall
}

func imageGenerationMetadataFromStreamEvent(event *StreamEvent) map[string]any {
	if event == nil {
		return nil
	}

	metadata := map[string]any{}
	if event.Background != "" {
		metadata["background"] = event.Background
	}
	if event.OutputFormat != "" {
		metadata["output_format"] = event.OutputFormat
	}
	if event.Quality != "" {
		metadata["quality"] = event.Quality
	}
	if event.Size != "" {
		metadata["size"] = event.Size
	}
	if event.RevisedPrompt != "" {
		metadata["revised_prompt"] = event.RevisedPrompt
	}
	if len(metadata) == 0 {
		return nil
	}

	return metadata
}

func mergeImageGenerationMetadata(base map[string]any, overlay map[string]any) map[string]any {
	if len(base) == 0 && len(overlay) == 0 {
		return nil
	}

	merged := map[string]any{}
	for key, value := range base {
		merged[key] = value
	}
	for key, value := range overlay {
		merged[key] = value
	}

	return merged
}

func (s *responsesOutboundStream) updateImageGenerationItem(itemID string, item *Item) {
	if itemID == "" || item == nil {
		return
	}

	itemCopy := *item
	s.state.imageGenerationItems[itemID] = &itemCopy
}

func (s *responsesOutboundStream) metadataForImageGenerationEvent(event *StreamEvent) map[string]any {
	if event == nil {
		return nil
	}

	metadata := imageGenerationMetadataFromStreamEvent(event)
	if event.ItemID == nil || *event.ItemID == "" {
		return metadata
	}

	item, ok := s.state.imageGenerationItems[*event.ItemID]
	if !ok || item == nil {
		return metadata
	}

	metadata = mergeImageGenerationMetadata(imageGenerationMetadataFromItem(item), metadata)
	return metadata
}

// TransformStream transforms OpenAI Responses API SSE events to unified llm.Response stream.
func (t *OutboundTransformer) TransformStream(
	ctx context.Context,
	stream streams.Stream[*httpclient.StreamEvent],
) (streams.Stream[*llm.Response], error) {
	// Append the DONE event to the stream
	doneEvent := lo.ToPtr(llm.DoneStreamEvent)
	streamWithDone := streams.AppendStream(stream, doneEvent)

	scope, _ := shared.GetTransportScope(ctx)
	return streams.NoNil(newResponsesOutboundStream(streamWithDone, scope)), nil
}

// responsesOutboundStream wraps a stream and maintains state during processing.
type responsesOutboundStream struct {
	stream streams.Stream[*httpclient.StreamEvent]
	state  *outboundStreamState

	// Event queue
	eventQueue []*llm.Response
	queueIndex int
	err        error
}

// outboundStreamState holds the state for a streaming session.
type outboundStreamState struct {
	responseID         string
	responseModel      string
	previousResponseID *string
	usage              *llm.Usage
	created            int64
	scope              shared.TransportScope

	// Content accumulation
	textContent      strings.Builder
	reasoningContent strings.Builder

	// Tool call tracking
	toolCalls     map[string]*llm.ToolCall // callID -> tool call
	itemToCallID  map[string]string        // item.id -> call_id mapping
	toolCallIndex map[string]int           // callID -> index in the output

	// Image generation tracking
	imageGenerationItems map[string]*Item // item.id -> latest image_generation_call item

	// Reasoning signature tracking
	encryptedContentEmitted map[string]bool
	hasEncryptedReasoning   bool
}

func newResponsesOutboundStream(stream streams.Stream[*httpclient.StreamEvent], scope shared.TransportScope) *responsesOutboundStream {
	return &responsesOutboundStream{
		stream: stream,
		state: &outboundStreamState{
			toolCalls:               make(map[string]*llm.ToolCall),
			itemToCallID:            make(map[string]string),
			toolCallIndex:           make(map[string]int),
			imageGenerationItems:    make(map[string]*Item),
			encryptedContentEmitted: make(map[string]bool),
			scope:                   scope,
		},
	}
}

func hasActionableToolCalls(toolCalls map[string]*llm.ToolCall) bool {
	for _, tc := range toolCalls {
		if tc == nil {
			continue
		}
		if tc.WebSearchToolCall == nil && tc.ImageGenerationToolCall == nil {
			return true
		}
	}

	return false
}

func (s *responsesOutboundStream) ensureImageGenerationToolCall(itemID string) (*llm.ToolCall, int) {
	if itemID == "" {
		return nil, 0
	}

	if callID, ok := s.state.itemToCallID[itemID]; ok {
		if tc, ok := s.state.toolCalls[callID]; ok {
			return tc, s.state.toolCallIndex[callID]
		}
	}

	toolCallIdx := len(s.state.toolCalls)
	tc := &llm.ToolCall{
		ID:    itemID,
		Type:  llm.ToolTypeImageGeneration,
		Index: toolCallIdx,
		ImageGenerationToolCall: &llm.ImageGenerationToolCall{
			ID: itemID,
		},
	}
	s.state.toolCalls[itemID] = tc
	s.state.itemToCallID[itemID] = itemID
	s.state.toolCallIndex[itemID] = toolCallIdx

	return tc, toolCallIdx
}

func (s *responsesOutboundStream) enqueue(resp *llm.Response) {
	s.eventQueue = append(s.eventQueue, resp)
}

func (s *responsesOutboundStream) Next() bool {
	// If we have events in the queue, return them first
	if s.queueIndex < len(s.eventQueue) {
		return true
	}

	// Clear the queue and reset index for new events
	s.eventQueue = nil
	s.queueIndex = 0

	// Try to get the next chunk from source
	if !s.stream.Next() {
		return false
	}

	event := s.stream.Current()

	err := s.transformStreamChunk(event)
	if err != nil {
		s.err = err
		return false
	}

	// Continue to the next event if no events were enqueued
	return s.Next()
}

// transformStreamChunk transforms a single OpenAI Responses API streaming chunk to unified llm.Response.
// Events are enqueued via s.enqueue() instead of being returned.
//
//nolint:maintidx,gocognit // It is complex and hard to split.
func (s *responsesOutboundStream) transformStreamChunk(event *httpclient.StreamEvent) error {
	if event == nil || len(event.Data) == 0 {
		return nil
	}

	// Handle [DONE] marker
	if string(event.Data) == "[DONE]" {
		s.enqueue(llm.DoneResponse)
		return nil
	}

	// Parse the streaming event
	var streamEvent StreamEvent

	err := json.Unmarshal(event.Data, &streamEvent)
	if err != nil {
		return fmt.Errorf("failed to unmarshal responses api stream event: %w", err)
	}

	if slog.Default().Enabled(context.Background(), slog.LevelDebug) {
		slog.DebugContext(context.Background(), "received response stream event", slog.Any("event", streamEvent))
	}

	// Build base response
	resp := &llm.Response{
		Object:             "chat.completion.chunk",
		ID:                 s.state.responseID,
		Model:              s.state.responseModel,
		Created:            s.state.created,
		PreviousResponseID: s.state.previousResponseID,
	}

	//nolint:exhaustive //Only process events we care about.
	switch streamEvent.Type {
	case StreamEventTypeResponseCreated:
		if streamEvent.Response != nil {
			s.state.responseID = streamEvent.Response.ID
			s.state.responseModel = streamEvent.Response.Model
			s.state.created = streamEvent.Response.CreatedAt
			s.state.previousResponseID = streamEvent.Response.PreviousResponseID

			resp.ID = s.state.responseID
			resp.Model = s.state.responseModel
			resp.Created = s.state.created
			resp.PreviousResponseID = s.state.previousResponseID

			if streamEvent.Response.Usage != nil {
				s.state.usage = streamEvent.Response.Usage.ToUsage()
				resp.Usage = s.state.usage
			}
		}

		resp.Choices = []llm.Choice{
			{
				Index: 0,
				Delta: &llm.Message{
					Role: "assistant",
				},
			},
		}

	case StreamEventTypeResponseInProgress:
		// Update state but don't emit an event
		if streamEvent.Response != nil {
			s.state.responseID = streamEvent.Response.ID
			s.state.responseModel = streamEvent.Response.Model
			s.state.created = streamEvent.Response.CreatedAt
			s.state.previousResponseID = streamEvent.Response.PreviousResponseID

			if streamEvent.Response.Usage != nil {
				s.state.usage = streamEvent.Response.Usage.ToUsage()
			}
		}

		return nil // Intentionally skip this event
	case StreamEventTypeOutputItemAdded:
		// Output item added - check type to determine how to handle
		if streamEvent.Item == nil {
			// No item data, skip
			return nil // Intentionally skip this event
		}

		item := streamEvent.Item
		switch item.Type {
		case "reasoning":
			if item.EncryptedContent == nil || *item.EncryptedContent == "" {
				return nil // Intentionally skip this event
			}

			if !s.state.encryptedContentEmitted[item.ID] {
				s.state.encryptedContentEmitted[item.ID] = true
				s.state.hasEncryptedReasoning = true
				resp.Choices = []llm.Choice{
					{
						Index: 0,
						Delta: &llm.Message{
							ReasoningSignature: shared.EncodeOpenAIEncryptedContentInScope(item.EncryptedContent, s.state.scope),
						},
					},
				}
			}

		case "function_call":
			// Initialize tool call tracking
			toolCallIdx := len(s.state.toolCalls)
			name := encodeResponsesFunctionCallName(item.Namespace, item.Name)
			s.state.toolCalls[item.CallID] = &llm.ToolCall{
				ID:   item.CallID,
				Type: "function",
				Function: llm.FunctionCall{
					Name:      name,
					Arguments: "",
				},
			}
			// Map item.id to call_id for later lookup
			s.state.itemToCallID[item.ID] = item.CallID
			s.state.toolCallIndex[item.CallID] = toolCallIdx

			resp.Choices = []llm.Choice{
				{
					Index: 0,
					Delta: &llm.Message{
						ToolCalls: []llm.ToolCall{
							{
								ID:    item.CallID,
								Type:  "function",
								Index: toolCallIdx,
								Function: llm.FunctionCall{
									Name: name,
								},
							},
						},
					},
				},
			}

		case "custom_tool_call":
			// Custom tool call - initialize tracking, input will be streamed via delta events
			toolCallIdx := len(s.state.toolCalls)
			s.state.toolCalls[item.CallID] = &llm.ToolCall{
				ID:   item.CallID,
				Type: llm.ToolTypeResponsesCustomTool,
				ResponseCustomToolCall: &llm.ResponseCustomToolCall{
					CallID: item.CallID,
					Name:   item.Name,
					Input:  "",
				},
			}
			s.state.itemToCallID[item.ID] = item.CallID
			s.state.toolCallIndex[item.CallID] = toolCallIdx

			resp.Choices = []llm.Choice{
				{
					Index: 0,
					Delta: &llm.Message{
						ToolCalls: []llm.ToolCall{
							{
								ID:    item.CallID,
								Type:  llm.ToolTypeResponsesCustomTool,
								Index: toolCallIdx,
								ResponseCustomToolCall: &llm.ResponseCustomToolCall{
									CallID: item.CallID,
									Name:   item.Name,
								},
							},
						},
					},
				},
			}

		case "web_search_call":
			toolCallIdx := len(s.state.toolCalls)
			callID := item.ID
			s.state.toolCalls[callID] = &llm.ToolCall{
				ID:    callID,
				Type:  llm.ToolTypeWebSearch,
				Index: toolCallIdx,
				WebSearchToolCall: &llm.WebSearchToolCall{
					ID:     item.ID,
					Status: lo.FromPtr(item.Status),
					Action: item.Action,
				},
			}
			s.state.itemToCallID[item.ID] = callID
			s.state.toolCallIndex[callID] = toolCallIdx

			resp.Choices = []llm.Choice{
				{
					Index: 0,
					Delta: &llm.Message{
						ToolCalls: []llm.ToolCall{
							{
								ID:    callID,
								Type:  llm.ToolTypeWebSearch,
								Index: toolCallIdx,
								WebSearchToolCall: &llm.WebSearchToolCall{
									ID:     item.ID,
									Status: lo.FromPtr(item.Status),
									Action: item.Action,
								},
							},
						},
					},
				},
			}

		case "image_generation_call":
			s.updateImageGenerationItem(item.ID, item)
			tc, toolCallIdx := s.ensureImageGenerationToolCall(item.ID)
			if tc == nil {
				return nil
			}
			tc.ImageGenerationToolCall = imageGenerationToolCallFromItem(item, StreamEventTypeOutputItemAdded)
			resp.Choices = []llm.Choice{
				{
					Index: 0,
					Delta: &llm.Message{
						ToolCalls: []llm.ToolCall{
							{
								ID:                      item.ID,
								Type:                    llm.ToolTypeImageGeneration,
								Index:                   toolCallIdx,
								ImageGenerationToolCall: tc.ImageGenerationToolCall,
							},
						},
					},
				},
			}

		default:
			// For other item types (e.g., message), skip - no meaningful content to emit
			return nil // Intentionally skip this event
		}

	case StreamEventTypeFunctionCallArgumentsDelta:
		// Function call arguments delta
		if streamEvent.ItemID != nil {
			// Look up call_id from item_id mapping
			callID, ok := s.state.itemToCallID[*streamEvent.ItemID]
			if !ok {
				// Fallback: item_id might be the call_id itself
				callID = *streamEvent.ItemID
			}

			if tc, ok := s.state.toolCalls[callID]; ok {
				tc.Function.Arguments += streamEvent.Delta
				toolCallIdx := s.state.toolCallIndex[callID]

				resp.Choices = []llm.Choice{
					{
						Index: 0,
						Delta: &llm.Message{
							ToolCalls: []llm.ToolCall{
								{
									Index: toolCallIdx,
									Function: llm.FunctionCall{
										Arguments: streamEvent.Delta,
									},
								},
							},
						},
					},
				}
			}
		}

	case StreamEventTypeFunctionCallArgumentsDone:
		// Function call completed - update state but don't emit an event
		if streamEvent.CallID != "" {
			if tc, ok := s.state.toolCalls[streamEvent.CallID]; ok {
				if streamEvent.Name != "" {
					if _, _, ok := decodeResponsesMCPToolName(tc.Function.Name); !ok {
						tc.Function.Name = streamEvent.Name
					}
				}
				tc.Function.Arguments = streamEvent.Arguments
			}
		}

		return nil // Intentionally skip this event

	case StreamEventTypeCustomToolCallInputDelta:
		// Custom tool call input delta - accumulate and emit as tool call delta
		if streamEvent.ItemID != nil {
			callID, ok := s.state.itemToCallID[*streamEvent.ItemID]
			if !ok {
				callID = *streamEvent.ItemID
			}

			if tc, ok := s.state.toolCalls[callID]; ok {
				tc.ResponseCustomToolCall.Input += streamEvent.Delta
				toolCallIdx := s.state.toolCallIndex[callID]

				resp.Choices = []llm.Choice{
					{
						Index: 0,
						Delta: &llm.Message{
							ToolCalls: []llm.ToolCall{
								{
									Index: toolCallIdx,
									Type:  llm.ToolTypeResponsesCustomTool,
									ResponseCustomToolCall: &llm.ResponseCustomToolCall{
										CallID: callID,
										Name:   tc.ResponseCustomToolCall.Name,
										Input:  streamEvent.Delta,
									},
								},
							},
						},
					},
				}
			}
		}

	case StreamEventTypeCustomToolCallInputDone:
		// Custom tool call input completed - update state but don't emit an event
		if streamEvent.ItemID != nil {
			callID, ok := s.state.itemToCallID[*streamEvent.ItemID]
			if !ok {
				callID = *streamEvent.ItemID
			}

			if tc, ok := s.state.toolCalls[callID]; ok {
				tc.ResponseCustomToolCall.Input = streamEvent.Input
			}
		}

		return nil // Intentionally skip this event

	case StreamEventTypeContentPartAdded:
		// Content part added - skip, no meaningful content to emit
		return nil // Intentionally skip this event

	case StreamEventTypeOutputTextDelta:
		// Text content delta
		s.state.textContent.WriteString(streamEvent.Delta)

		resp.Choices = []llm.Choice{
			{
				Index: 0,
				Delta: &llm.Message{
					Content: llm.MessageContent{
						Content: &streamEvent.Delta,
					},
				},
			},
		}

	case StreamEventTypeReasoningSummaryTextDelta:
		// Reasoning content delta
		s.state.reasoningContent.WriteString(streamEvent.Delta)

		resp.Choices = []llm.Choice{
			{
				Index: 0,
				Delta: &llm.Message{
					ReasoningContent: &streamEvent.Delta,
				},
			},
		}

	case StreamEventTypeOutputTextDone:
		// Text content completed - skip, content was already streamed via deltas
		return nil // Intentionally skip this event

	case StreamEventTypeReasoningSummaryTextDone:
		// Reasoning content completed - skip, content was already streamed via deltas
		return nil // Intentionally skip this event

	case StreamEventTypeOutputItemDone, StreamEventTypeContentPartDone,
		StreamEventTypeReasoningSummaryPartAdded, StreamEventTypeReasoningSummaryPartDone:
		if streamEvent.Type == StreamEventTypeOutputItemDone &&
			streamEvent.Item != nil &&
			streamEvent.Item.Type == "image_generation_call" &&
			streamEvent.Item.ID != "" {
			s.updateImageGenerationItem(streamEvent.Item.ID, streamEvent.Item)
			tc, toolCallIdx := s.ensureImageGenerationToolCall(streamEvent.Item.ID)
			if tc != nil {
				tc.ImageGenerationToolCall = imageGenerationToolCallFromItem(streamEvent.Item, StreamEventTypeOutputItemDone)
				resp.Choices = []llm.Choice{
					{
						Index: 0,
						Delta: &llm.Message{
							ToolCalls: []llm.ToolCall{
								{
									ID:                      streamEvent.Item.ID,
									Type:                    llm.ToolTypeImageGeneration,
									Index:                   toolCallIdx,
									ImageGenerationToolCall: tc.ImageGenerationToolCall,
								},
							},
						},
					},
				}
			}
		}
		if streamEvent.Type == StreamEventTypeOutputItemDone &&
			streamEvent.Item != nil &&
			streamEvent.Item.Type == "web_search_call" &&
			streamEvent.Item.ID != "" {
			if callID, ok := s.state.itemToCallID[streamEvent.Item.ID]; ok {
				if tc, ok := s.state.toolCalls[callID]; ok && tc.WebSearchToolCall != nil {
					tc.WebSearchToolCall.Status = lo.FromPtr(streamEvent.Item.Status)
					tc.WebSearchToolCall.Action = streamEvent.Item.Action
					resp.Choices = []llm.Choice{
						{
							Index: 0,
							Delta: &llm.Message{
								ToolCalls: []llm.ToolCall{
									{
										ID:                callID,
										Type:              llm.ToolTypeWebSearch,
										Index:             s.state.toolCallIndex[callID],
										WebSearchToolCall: tc.WebSearchToolCall,
									},
								},
							},
						},
					}
					break
				}
			}
		}

		// These events don't need special handling - skip
		if len(resp.Choices) == 0 {
			return nil // Intentionally skip this event
		}

	case StreamEventTypeResponseCompleted:
		// Response completed - emit two events: one with finish_reason, one with usage
		if streamEvent.Response != nil {
			s.state.previousResponseID = streamEvent.Response.PreviousResponseID
			resp.PreviousResponseID = s.state.previousResponseID

			for i := range streamEvent.Response.Output {
				item := &streamEvent.Response.Output[i]
				if item.Type == "image_generation_call" && item.ID != "" {
					s.updateImageGenerationItem(item.ID, item)
					tc, toolCallIdx := s.ensureImageGenerationToolCall(item.ID)
					if tc != nil {
						tc.ImageGenerationToolCall = imageGenerationToolCallFromItem(item, StreamEventTypeResponseCompleted)
						tc.ImageGenerationToolCall.EventType = "response.completed"
						s.enqueue(&llm.Response{
							Object:             "chat.completion.chunk",
							ID:                 s.state.responseID,
							Model:              s.state.responseModel,
							Created:            s.state.created,
							PreviousResponseID: s.state.previousResponseID,
							Choices: []llm.Choice{
								{
									Index: 0,
									Delta: &llm.Message{
										ToolCalls: []llm.ToolCall{
											{
												ID:                      item.ID,
												Type:                    llm.ToolTypeImageGeneration,
												Index:                   toolCallIdx,
												ImageGenerationToolCall: tc.ImageGenerationToolCall,
											},
										},
									},
								},
							},
						})
					}
				}
			}
		}

		finishReason := "stop"
		if hasActionableToolCalls(s.state.toolCalls) {
			finishReason = "tool_calls"
		}

		// First event: finish_reason with empty delta
		resp.Choices = []llm.Choice{
			{
				Index:        0,
				Delta:        &llm.Message{},
				FinishReason: &finishReason,
			},
		}
		// Second event: usage (if available)
		if streamEvent.Response != nil && streamEvent.Response.Usage != nil {
			s.state.usage = streamEvent.Response.Usage.ToUsage()
			usageResp := &llm.Response{
				Object:             "chat.completion.chunk",
				ID:                 s.state.responseID,
				Model:              s.state.responseModel,
				Created:            s.state.created,
				PreviousResponseID: s.state.previousResponseID,
				Choices:            []llm.Choice{},
				Usage:              s.state.usage,
			}

			s.enqueue(resp)
			s.enqueue(usageResp)

			return nil
		}

	case StreamEventTypeResponseFailed:
		// Response failed
		finishReason := "error"
		resp.Choices = []llm.Choice{
			{
				Index:        0,
				FinishReason: &finishReason,
			},
		}

	case StreamEventTypeResponseIncomplete:
		// Response incomplete (e.g., max tokens)
		finishReason := "length"
		resp.Choices = []llm.Choice{
			{
				Index:        0,
				FinishReason: &finishReason,
			},
		}

	case StreamEventTypeError:
		return &llm.ResponseError{
			Detail: llm.ErrorDetail{
				Code:    streamEvent.Code,
				Message: streamEvent.Message,
				Param:   lo.FromPtr(streamEvent.Param),
			},
		}

	case StreamEventTypeWebSearchCallInProgress, StreamEventTypeWebSearchCallSearching, StreamEventTypeWebSearchCallCompleted:
		if streamEvent.ItemID != nil {
			if callID, ok := s.state.itemToCallID[*streamEvent.ItemID]; ok {
				if tc, ok := s.state.toolCalls[callID]; ok && tc.WebSearchToolCall != nil {
					switch streamEvent.Type {
					case StreamEventTypeWebSearchCallInProgress:
						tc.WebSearchToolCall.Status = "in_progress"
					case StreamEventTypeWebSearchCallSearching:
						tc.WebSearchToolCall.Status = "searching"
					case StreamEventTypeWebSearchCallCompleted:
						tc.WebSearchToolCall.Status = "completed"
					}
				}
			}
		}
		return nil

	case StreamEventTypeImageGenerationPartialImage,
		StreamEventTypeImageGenerationGenerating,
		StreamEventTypeImageGenerationInProgress,
		StreamEventTypeImageGenerationCompleted:
		// Handle image generation events
		if streamEvent.ItemID == nil || *streamEvent.ItemID == "" {
			return nil
		}

		tc, toolCallIdx := s.ensureImageGenerationToolCall(*streamEvent.ItemID)
		if tc == nil {
			return nil
		}
		metadata := s.metadataForImageGenerationEvent(&streamEvent)
		update := imageGenerationToolCallFromStreamEvent(&streamEvent, metadata)
		if update.ID == "" {
			update.ID = *streamEvent.ItemID
		}
		tc.ImageGenerationToolCall = update

		resp.Choices = []llm.Choice{
			{
				Index: 0,
				Delta: &llm.Message{
					ToolCalls: []llm.ToolCall{
						{
							ID:                      update.ID,
							Type:                    llm.ToolTypeImageGeneration,
							Index:                   toolCallIdx,
							ImageGenerationToolCall: update,
						},
					},
				},
			},
		}

	default:
		// Unknown event type - skip
		return nil // Intentionally skip this event
	}

	s.enqueue(resp)

	return nil
}

func (s *responsesOutboundStream) Current() *llm.Response {
	if s.queueIndex < len(s.eventQueue) {
		event := s.eventQueue[s.queueIndex]
		s.queueIndex++

		return event
	}

	return nil
}

func (s *responsesOutboundStream) Err() error {
	if s.err != nil {
		return s.err
	}

	return s.stream.Err()
}

func (s *responsesOutboundStream) Close() error {
	return s.stream.Close()
}

// AggregateStreamChunks aggregates OpenAI Responses API streaming chunks into a complete response.
func (t *OutboundTransformer) AggregateStreamChunks(
	ctx context.Context,
	chunks []*httpclient.StreamEvent,
) ([]byte, llm.ResponseMeta, error) {
	return AggregateStreamChunks(ctx, chunks)
}

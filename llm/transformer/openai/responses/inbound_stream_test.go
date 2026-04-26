package responses

import (
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/samber/lo"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/internal/pkg/xtest"
	"github.com/looplj/axonhub/llm/streams"
)

// Compare each event.
var ignoreFields = cmp.FilterPath(func(p cmp.Path) bool {
	// Ignore dynamic fields that are generated at runtime
	if sf, ok := p.Last().(cmp.StructField); ok {
		switch sf.Name() {
		case "ID", "ItemID", "Obfuscation", "Logprobs", "Response":
			return true
		}
	}
	return false
}, cmp.Ignore())

func TestInboundTransformer_StreamTransformation_WithTestData(t *testing.T) {
	trans := NewInboundTransformer()

	tests := []struct {
		name                 string
		inputStreamFile      string
		expectedStreamFile   string
		expectedResponseFile string
	}{
		{
			name:                 "stream transformation with text and multiple tool calls",
			inputStreamFile:      "llm-tool-2.stream.jsonl",
			expectedStreamFile:   "tool-2.stream.jsonl",
			expectedResponseFile: "tool-2.response.json",
		},
		{
			name:                 "stream transformation with custom tool call",
			inputStreamFile:      "llm-custom_tool.stream.jsonl",
			expectedStreamFile:   "custom_tool.stream.jsonl",
			expectedResponseFile: "custom_tool.stream.response.json",
		},
		{
			name:                 "stream transformation with encrypted reasoning only (no summary items)",
			inputStreamFile:      "llm-encrypted_only.stream.jsonl",
			expectedStreamFile:   "encrypted_only.stream.jsonl",
			expectedResponseFile: "encrypted_only.response.json",
		},
		{
			name:                 "stream transformation with image generation output",
			inputStreamFile:      "llm-image_generation.stream.jsonl",
			expectedStreamFile:   "image_generation.stream.jsonl",
			expectedResponseFile: "image_generation.response.json",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Load the input file (LLM format responses)
			llmResponses, err := xtest.LoadLlmResponses(t, tt.inputStreamFile)
			require.NoError(t, err)

			// Load expected events from the expected stream file
			expectedEvents, err := xtest.LoadStreamChunks(t, tt.expectedStreamFile)
			require.NoError(t, err)

			// Create a mock stream from LLM responses
			mockStream := streams.SliceStream(llmResponses)

			// Transform the stream (LLM -> OpenAI Responses API)
			transformedStream, err := trans.TransformStream(t.Context(), mockStream)
			require.NoError(t, err)

			// Collect all transformed events
			var actualEvents []StreamEvent

			for transformedStream.Next() {
				event := transformedStream.Current()

				var ev StreamEvent

				err := json.Unmarshal(event.Data, &ev)
				require.NoError(t, err)

				actualEvents = append(actualEvents, ev)
			}

			require.NoError(t, transformedStream.Err())

			// Verify event count
			require.Equal(t, len(expectedEvents), len(actualEvents), "Event count should match expected")

			for i, expectedEvent := range expectedEvents {
				var expected StreamEvent

				err := json.Unmarshal(expectedEvent.Data, &expected)
				require.NoError(t, err)

				actual := actualEvents[i]

				if !xtest.Equal(expected, actual, ignoreFields) {
					t.Fatalf("event %d mismatch:\n%s", i, cmp.Diff(expected, actual, ignoreFields))
				}
			}

			// Verify the last event is response.completed and compare with expectedResponseFile
			if tt.expectedResponseFile != "" {
				require.NotEmpty(t, actualEvents, "Expected at least one event")

				lastEvent := actualEvents[len(actualEvents)-1]
				require.Equal(t, StreamEventTypeResponseCompleted, lastEvent.Type,
					"Last event should be response.completed")
				require.NotNil(t, lastEvent.Response, "response.completed event should have Response")

				// Load expected response from file
				var expectedResponse Response

				err := xtest.LoadTestData(t, tt.expectedResponseFile, &expectedResponse)
				require.NoError(t, err)

				// Compare the response in the event with the expected response file
				// Ignore dynamic fields like ID, ItemID
				responseIgnoreFields := cmp.FilterPath(func(p cmp.Path) bool {
					if sf, ok := p.Last().(cmp.StructField); ok {
						switch sf.Name() {
						case "ID", "ItemID", "Obfuscation", "Logprobs":
							return true
						}
					}

					return false
				}, cmp.Ignore())

				if !xtest.Equal(expectedResponse, *lastEvent.Response, responseIgnoreFields) {
					t.Fatalf("response.completed response mismatch:\n%s",
						cmp.Diff(expectedResponse, *lastEvent.Response, responseIgnoreFields))
				}
			}
		})
	}
}

func TestInboundTransformer_StreamTransformation_ImageGenerationEmitsProtocolEvents(t *testing.T) {
	trans := NewInboundTransformer()

	llmResponses, err := xtest.LoadLlmResponses(t, "llm-image_generation.stream.jsonl")
	require.NoError(t, err)

	transformedStream, err := trans.TransformStream(t.Context(), streams.SliceStream(llmResponses))
	require.NoError(t, err)

	var actualEvents []StreamEvent
	for transformedStream.Next() {
		event := transformedStream.Current()

		var ev StreamEvent
		err := json.Unmarshal(event.Data, &ev)
		require.NoError(t, err)

		actualEvents = append(actualEvents, ev)
	}
	require.NoError(t, transformedStream.Err())

	var partialEvent *StreamEvent
	var outputItemDoneEvent *StreamEvent
	for i := range actualEvents {
		ev := &actualEvents[i]
		switch ev.Type {
		case StreamEventTypeImageGenerationPartialImage:
			partialEvent = ev
		case StreamEventTypeOutputItemDone:
			if ev.Item != nil && ev.Item.Type == "image_generation_call" {
				outputItemDoneEvent = ev
			}
		}
	}

	require.NotNil(t, partialEvent)
	require.Equal(t, "generating", partialEvent.Status)
	require.Equal(t, "opaque", partialEvent.Background)
	require.Equal(t, "png", partialEvent.OutputFormat)
	require.Equal(t, "high", partialEvent.Quality)
	require.Equal(t, "1024x1536", partialEvent.Size)
	require.Equal(t, "A watercolor fox under moonlight", partialEvent.RevisedPrompt)
	require.Equal(t, "base64data", partialEvent.PartialImageB64)

	require.NotNil(t, outputItemDoneEvent)
	require.NotNil(t, outputItemDoneEvent.Item)
	require.Equal(t, "generating", lo.FromPtr(outputItemDoneEvent.Item.Status))
	require.Equal(t, lo.ToPtr("opaque"), outputItemDoneEvent.Item.Background)
	require.Equal(t, lo.ToPtr("png"), outputItemDoneEvent.Item.OutputFormat)
	require.Equal(t, lo.ToPtr("high"), outputItemDoneEvent.Item.Quality)
	require.Equal(t, lo.ToPtr("1024x1536"), outputItemDoneEvent.Item.Size)
	require.Equal(t, lo.ToPtr("A watercolor fox under moonlight"), outputItemDoneEvent.Item.RevisedPrompt)
}

func TestInboundTransformer_StreamTransformation_ImageGenerationToolCallRoundTrip(t *testing.T) {
	trans := NewInboundTransformer()

	llmResponses := []*llm.Response{
		{
			Object:  "chat.completion.chunk",
			ID:      "resp_img_tool_call",
			Model:   "gpt-5.4",
			Created: 1700000000,
			Choices: []llm.Choice{
				{
					Index: 0,
					Delta: &llm.Message{
						ToolCalls: []llm.ToolCall{
							{
								ID:    "ig_123",
								Type:  llm.ToolTypeImageGeneration,
								Index: 0,
								ImageGenerationToolCall: &llm.ImageGenerationToolCall{
									ID:        "ig_123",
									EventType: string(StreamEventTypeOutputItemAdded),
									Status:    "in_progress",
								},
							},
						},
					},
				},
			},
		},
		imageGenerationToolCallResponse("resp_img_tool_call", "ig_123", string(StreamEventTypeImageGenerationInProgress), "in_progress", ""),
		imageGenerationToolCallResponse("resp_img_tool_call", "ig_123", string(StreamEventTypeImageGenerationGenerating), "generating", ""),
		{
			Object:  "chat.completion.chunk",
			ID:      "resp_img_tool_call",
			Model:   "gpt-5.4",
			Created: 1700000000,
			Choices: []llm.Choice{
				{
					Index: 0,
					Delta: &llm.Message{
						ToolCalls: []llm.ToolCall{
							{
								ID:    "ig_123",
								Type:  llm.ToolTypeImageGeneration,
								Index: 0,
								ImageGenerationToolCall: &llm.ImageGenerationToolCall{
									ID:                "ig_123",
									EventType:         string(StreamEventTypeImageGenerationPartialImage),
									Status:            "generating",
									PartialImageB64:   "base64data",
									PartialImageIndex: lo.ToPtr(0),
									Background:        "opaque",
									OutputFormat:      "png",
									Quality:           "high",
									Size:              "1024x1536",
									RevisedPrompt:     "A watercolor fox under moonlight",
								},
							},
						},
					},
				},
			},
		},
		{
			Object:  "chat.completion.chunk",
			ID:      "resp_img_tool_call",
			Model:   "gpt-5.4",
			Created: 1700000000,
			Choices: []llm.Choice{
				{
					Index: 0,
					Delta: &llm.Message{
						ToolCalls: []llm.ToolCall{
							{
								ID:    "ig_123",
								Type:  llm.ToolTypeImageGeneration,
								Index: 0,
								ImageGenerationToolCall: &llm.ImageGenerationToolCall{
									ID:            "ig_123",
									EventType:     string(StreamEventTypeOutputItemDone),
									Status:        "completed",
									Result:        "base64data",
									Background:    "opaque",
									OutputFormat:  "png",
									Quality:       "high",
									Size:          "1024x1536",
									RevisedPrompt: "A watercolor fox under moonlight",
								},
							},
						},
					},
				},
			},
		},
		{
			Object:  "chat.completion.chunk",
			ID:      "resp_img_tool_call",
			Model:   "gpt-5.4",
			Created: 1700000000,
			Choices: []llm.Choice{
				{
					Index:        0,
					Delta:        &llm.Message{},
					FinishReason: lo.ToPtr("stop"),
				},
			},
		},
		{
			Object:  "chat.completion.chunk",
			ID:      "resp_img_tool_call",
			Model:   "gpt-5.4",
			Created: 1700000000,
			Usage: &llm.Usage{
				PromptTokens:     1,
				CompletionTokens: 1,
				TotalTokens:      2,
			},
		},
	}

	transformedStream, err := trans.TransformStream(t.Context(), streams.SliceStream(llmResponses))
	require.NoError(t, err)

	var actualEvents []StreamEvent
	for transformedStream.Next() {
		event := transformedStream.Current()
		var ev StreamEvent
		err := json.Unmarshal(event.Data, &ev)
		require.NoError(t, err)
		actualEvents = append(actualEvents, ev)
	}
	require.NoError(t, transformedStream.Err())

	eventTypes := make([]StreamEventType, 0, len(actualEvents))
	var partialEvent *StreamEvent
	var doneEvent *StreamEvent
	var completedEvent *StreamEvent
	for i := range actualEvents {
		ev := &actualEvents[i]
		eventTypes = append(eventTypes, ev.Type)
		switch ev.Type {
		case StreamEventTypeImageGenerationPartialImage:
			partialEvent = ev
		case StreamEventTypeOutputItemDone:
			if ev.Item != nil && ev.Item.Type == "image_generation_call" {
				doneEvent = ev
			}
		case StreamEventTypeResponseCompleted:
			completedEvent = ev
		}
	}

	require.Contains(t, eventTypes, StreamEventTypeOutputItemAdded)
	require.Contains(t, eventTypes, StreamEventTypeImageGenerationInProgress)
	require.Contains(t, eventTypes, StreamEventTypeImageGenerationGenerating)
	require.Contains(t, eventTypes, StreamEventTypeImageGenerationPartialImage)
	require.Contains(t, eventTypes, StreamEventTypeOutputItemDone)
	require.NotContains(t, eventTypes, StreamEventTypeImageGenerationCompleted)

	require.NotNil(t, partialEvent)
	require.Equal(t, "ig_123", lo.FromPtr(partialEvent.ItemID))
	require.Equal(t, "base64data", partialEvent.PartialImageB64)
	require.Equal(t, lo.ToPtr(0), partialEvent.PartialImageIndex)
	require.Equal(t, "A watercolor fox under moonlight", partialEvent.RevisedPrompt)

	require.NotNil(t, doneEvent)
	require.NotNil(t, doneEvent.Item)
	require.Equal(t, lo.ToPtr("completed"), doneEvent.Item.Status)
	require.Equal(t, lo.ToPtr("base64data"), doneEvent.Item.Result)
	require.Equal(t, lo.ToPtr("A watercolor fox under moonlight"), doneEvent.Item.RevisedPrompt)

	require.NotNil(t, completedEvent)
	require.Len(t, completedEvent.Response.Output, 1)
	require.Equal(t, "image_generation_call", completedEvent.Response.Output[0].Type)
	require.Equal(t, lo.ToPtr("A watercolor fox under moonlight"), completedEvent.Response.Output[0].RevisedPrompt)
}

func imageGenerationToolCallResponse(responseID string, itemID string, eventType string, status string, partialImageB64 string) *llm.Response {
	imageCall := &llm.ImageGenerationToolCall{
		ID:              itemID,
		EventType:       eventType,
		Status:          status,
		PartialImageB64: partialImageB64,
	}

	return &llm.Response{
		Object:  "chat.completion.chunk",
		ID:      responseID,
		Model:   "gpt-5.4",
		Created: 1700000000,
		Choices: []llm.Choice{
			{
				Index: 0,
				Delta: &llm.Message{
					ToolCalls: []llm.ToolCall{
						{
							ID:                      itemID,
							Type:                    llm.ToolTypeImageGeneration,
							Index:                   0,
							ImageGenerationToolCall: imageCall,
						},
					},
				},
			},
		},
	}
}

func TestInboundTransformer_StreamTransformation_ImageGenerationLateMetadataUpdate(t *testing.T) {
	trans := NewInboundTransformer()

	llmResponses := []*llm.Response{
		{
			Object:  "chat.completion.chunk",
			ID:      "resp_img_late",
			Model:   "gpt-5.4",
			Created: 1700000000,
			Choices: []llm.Choice{
				{
					Index: 0,
					Delta: &llm.Message{
						Content: llm.MessageContent{
							MultipleContent: []llm.MessageContentPart{
								{
									Type: "image_url",
									ImageURL: &llm.ImageURL{
										URL: "data:image/png;base64,base64data",
									},
								},
							},
						},
					},
				},
			},
		},
		{
			Object:  "chat.completion.chunk",
			ID:      "resp_img_late",
			Model:   "gpt-5.4",
			Created: 1700000000,
			Choices: []llm.Choice{
				{
					Index: 0,
					Delta: &llm.Message{},
					TransformerMetadata: map[string]any{
						"image_generation_item_updates": map[string]any{
							"ig_345": map[string]any{
								"revised_prompt": "late revised prompt",
								"background":     "opaque",
								"output_format":  "png",
								"quality":        "high",
								"size":           "1536x1024",
							},
						},
					},
				},
			},
		},
		{
			Object:  "chat.completion.chunk",
			ID:      "resp_img_late",
			Model:   "gpt-5.4",
			Created: 1700000000,
			Choices: []llm.Choice{
				{
					Index:        0,
					Delta:        &llm.Message{},
					FinishReason: lo.ToPtr("stop"),
				},
			},
		},
		{
			Object:  "chat.completion.chunk",
			ID:      "resp_img_late",
			Model:   "gpt-5.4",
			Created: 1700000000,
			Choices: []llm.Choice{},
			Usage: &llm.Usage{
				PromptTokens:     1,
				CompletionTokens: 1,
				TotalTokens:      2,
			},
		},
	}

	transformedStream, err := trans.TransformStream(t.Context(), streams.SliceStream(llmResponses))
	require.NoError(t, err)

	var actualEvents []StreamEvent
	for transformedStream.Next() {
		event := transformedStream.Current()
		var ev StreamEvent
		err := json.Unmarshal(event.Data, &ev)
		require.NoError(t, err)
		actualEvents = append(actualEvents, ev)
	}
	require.NoError(t, transformedStream.Err())

	var partialEvent *StreamEvent
	var outputItemDoneEvent *StreamEvent
	for i := range actualEvents {
		ev := &actualEvents[i]
		switch ev.Type {
		case StreamEventTypeImageGenerationPartialImage:
			partialEvent = ev
		case StreamEventTypeOutputItemDone:
			if ev.Item != nil && ev.Item.Type == "image_generation_call" {
				outputItemDoneEvent = ev
			}
		}
	}

	require.NotNil(t, partialEvent)
	require.Equal(t, "late revised prompt", partialEvent.RevisedPrompt)
	require.Equal(t, "opaque", partialEvent.Background)
	require.Equal(t, "png", partialEvent.OutputFormat)
	require.Equal(t, "high", partialEvent.Quality)
	require.Equal(t, "1536x1024", partialEvent.Size)

	require.NotNil(t, outputItemDoneEvent)
	require.NotNil(t, outputItemDoneEvent.Item)
	require.Equal(t, lo.ToPtr("late revised prompt"), outputItemDoneEvent.Item.RevisedPrompt)
}

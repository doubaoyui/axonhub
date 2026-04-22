package responses

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"
	"github.com/samber/lo"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
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

func TestResponsesRealReplay_ImageGenerationDebugCapture(t *testing.T) {
	outbound, err := NewOutboundTransformer("https://api.openai.com", "test-api-key")
	require.NoError(t, err)

	inbound := NewInboundTransformer()

	upstreamPath := filepath.Join("..", "..", "..", "..", "..", "..", "refrence", "debug", "chunks-1776843642950.json")
	raw, err := os.ReadFile(upstreamPath)
	require.NoError(t, err)

	var captured []struct {
		Data  json.RawMessage `json:"data"`
		Event string          `json:"event"`
	}
	err = json.Unmarshal(raw, &captured)
	require.NoError(t, err)
	require.NotEmpty(t, captured)

	upstreamEvents := make([]*httpclient.StreamEvent, 0, len(captured))
	for _, item := range captured {
		if len(item.Data) == 0 || item.Event == "" {
			continue
		}
		upstreamEvents = append(upstreamEvents, &httpclient.StreamEvent{
			Type: item.Event,
			Data: item.Data,
		})
	}
	require.NotEmpty(t, upstreamEvents)

	llmStream, err := outbound.TransformStream(t.Context(), streams.SliceStream(upstreamEvents))
	require.NoError(t, err)
	llmResponses, err := streams.All(llmStream)
	require.NoError(t, err)
	require.NotEmpty(t, llmResponses)

	roundtripStream, err := inbound.TransformStream(t.Context(), streams.SliceStream(llmResponses))
	require.NoError(t, err)

	var actualEvents []StreamEvent
	for roundtripStream.Next() {
		event := roundtripStream.Current()
		var ev StreamEvent
		err := json.Unmarshal(event.Data, &ev)
		require.NoError(t, err)
		actualEvents = append(actualEvents, ev)
	}
	require.NoError(t, roundtripStream.Err())
	require.NotEmpty(t, actualEvents)

	imageStart := -1
	expectedOrder := []StreamEventType{
		StreamEventTypeOutputItemAdded,
		StreamEventTypeImageGenerationInProgress,
		StreamEventTypeImageGenerationGenerating,
		StreamEventTypeImageGenerationPartialImage,
		StreamEventTypeOutputItemDone,
	}

	for i := 0; i+len(expectedOrder) <= len(actualEvents); i++ {
		if actualEvents[i].Type == StreamEventTypeOutputItemAdded &&
			actualEvents[i].Item != nil &&
			actualEvents[i].Item.Type == "image_generation_call" {
			matched := true
			for offset, want := range expectedOrder {
				if actualEvents[i+offset].Type != want {
					matched = false
					break
				}
			}
			if matched {
				imageStart = i
				break
			}
		}
	}

	require.NotEqual(t, -1, imageStart, "image generation event sequence not found")

	for _, ev := range actualEvents {
		require.NotEqual(t, StreamEventTypeImageGenerationCompleted, ev.Type)
	}

	partialEvent := actualEvents[imageStart+3]
	require.Equal(t, "1536x1024", partialEvent.Size)
	require.Equal(t, "medium", partialEvent.Quality)
	require.Equal(t, "opaque", partialEvent.Background)
	require.Equal(t, "png", partialEvent.OutputFormat)
	require.Contains(t, partialEvent.RevisedPrompt, "Liu Yifei")
	require.NotEmpty(t, partialEvent.PartialImageB64)

	outputDoneEvent := actualEvents[imageStart+4]
	require.NotNil(t, outputDoneEvent.Item)
	require.Equal(t, "image_generation_call", outputDoneEvent.Item.Type)
	require.Equal(t, "generating", lo.FromPtr(outputDoneEvent.Item.Status))
	require.Equal(t, "1536x1024", lo.FromPtr(outputDoneEvent.Item.Size))
	require.Equal(t, "medium", lo.FromPtr(outputDoneEvent.Item.Quality))
	require.Equal(t, "opaque", lo.FromPtr(outputDoneEvent.Item.Background))
	require.Equal(t, "png", lo.FromPtr(outputDoneEvent.Item.OutputFormat))
	require.NotNil(t, outputDoneEvent.Item.RevisedPrompt)
	require.Contains(t, *outputDoneEvent.Item.RevisedPrompt, "Liu Yifei")
}

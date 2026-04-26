package responses

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/internal/pkg/xtest"
	"github.com/looplj/axonhub/llm/streams"
)

func TestOutboundTransformer_StreamTransformation_WithTestData(t *testing.T) {
	trans, err := NewOutboundTransformer("https://api.openai.com", "test-api-key")
	require.NoError(t, err)

	tests := []struct {
		name                 string
		inputStreamFile      string // OpenAI Responses API stream format
		expectedStreamFile   string // Expected LLM stream format
		expectedResponseFile string // Final LLM response format
	}{
		{
			name:                 "stream transformation with text and multiple tool calls",
			inputStreamFile:      "tool-2.stream.jsonl",
			expectedStreamFile:   "llm-tool-2.stream.jsonl",
			expectedResponseFile: "llm-tool-2.response.json",
		},
		{
			name:                 "stream transformation with encrypted reasoning",
			inputStreamFile:      "encrypted_content.stream.jsonl",
			expectedStreamFile:   "llm-encrypted_content.stream.jsonl",
			expectedResponseFile: "llm-encrypted_content.response.json",
		},
		{
			name:                 "stream transformation with custom tool call",
			inputStreamFile:      "custom_tool.stream.jsonl",
			expectedStreamFile:   "llm-custom_tool.stream.jsonl",
			expectedResponseFile: "llm-custom_tool.stream.response.json",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			expectedEvents, err := xtest.LoadLlmResponses(t, tt.expectedStreamFile)
			require.NoError(t, err)

			// Load the input file (OpenAI Responses API format events)
			responsesAPIEvents, err := xtest.LoadStreamChunks(t, tt.inputStreamFile)
			require.NoError(t, err)

			// Transform the stream (OpenAI Responses API -> LLM format)
			transformedStream, err := trans.TransformStream(t.Context(), streams.SliceStream(responsesAPIEvents))
			require.NoError(t, err)
			require.NoError(t, transformedStream.Err())

			// Collect all transformed events
			actualLLMResponses, err := streams.All(transformedStream)
			require.NoError(t, err)

			// Stream transformation may not be 1:1, so we verify key properties instead of exact count
			require.NotEmpty(t, actualLLMResponses, "Should have at least one response")

			// Verify the last event is DONE
			lastEvent := actualLLMResponses[len(actualLLMResponses)-1]
			require.Equal(t, llm.DoneResponse, lastEvent, "Last event should be DONE")

			// Verify non-DONE events have valid structure
			for _, resp := range actualLLMResponses {
				if resp != llm.DoneResponse {
					// Verify each response has the correct object type
					require.Contains(t, []string{"chat.completion", "chat.completion.chunk"}, resp.Object,
						"Response should be chat.completion or chat.completion.chunk")
				}
			}

			require.Len(t, actualLLMResponses, len(expectedEvents))

			// exclude the last DONE event
			for i, expectedEvent := range expectedEvents[:len(expectedEvents)-1] {
				if !xtest.Equal(expectedEvent, actualLLMResponses[i]) {
					t.Fatalf("event %d mismatch:\n%s", i, cmp.Diff(expectedEvent, actualLLMResponses[i]))
				}
			}

			// Verify the final response against expectedResponseFile
			if tt.expectedResponseFile != "" {
				// Find the last non-DONE response
				var lastResponse *llm.Response

				for i := len(actualLLMResponses) - 1; i >= 0; i-- {
					if actualLLMResponses[i] != llm.DoneResponse {
						lastResponse = actualLLMResponses[i]

						break
					}
				}

				require.NotNil(t, lastResponse, "Expected at least one non-DONE response")

				// Load expected final response from file
				var expectedFinalResponse llm.Response

				err := xtest.LoadTestData(t, tt.expectedResponseFile, &expectedFinalResponse)
				require.NoError(t, err)

				// Compare model and ID from the last response
				require.Equal(t, expectedFinalResponse.Model, lastResponse.Model,
					"Final response model should match")
				require.Equal(t, expectedFinalResponse.ID, lastResponse.ID,
					"Final response ID should match")
			}
		})
	}
}

func TestOutboundTransformer_StreamTransformation_ErrorEvent(t *testing.T) {
	trans, err := NewOutboundTransformer("https://api.openai.com", "test-api-key")
	require.NoError(t, err)

	responsesAPIEvents, err := xtest.LoadStreamChunks(t, "error.response.stream.jsonl")
	require.NoError(t, err)

	transformedStream, err := trans.TransformStream(t.Context(), streams.SliceStream(responsesAPIEvents))
	require.NoError(t, err)

	_, err = streams.All(transformedStream)
	require.Error(t, err)
	require.Contains(t, err.Error(), "Something went wrong")
}

func TestOutboundTransformer_TransformStream_PreservesPreviousResponseID(t *testing.T) {
	trans, err := NewOutboundTransformer("https://api.openai.com", "test-api-key")
	require.NoError(t, err)

	events := []*httpclient.StreamEvent{
		{
			Type: "response.created",
			Data: []byte(`{
				"type":"response.created",
				"response":{
					"id":"resp_stream_prev",
					"object":"response",
					"created_at":1700000000,
					"model":"gpt-5.4",
					"status":"in_progress",
					"previous_response_id":"resp_prev_123",
					"output":[]
				}
			}`),
		},
		{
			Type: "response.completed",
			Data: []byte(`{
				"type":"response.completed",
				"response":{
					"id":"resp_stream_prev",
					"object":"response",
					"created_at":1700000000,
					"model":"gpt-5.4",
					"status":"completed",
					"previous_response_id":"resp_prev_123",
					"output":[],
					"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}
				}
			}`),
		},
	}

	stream, err := trans.TransformStream(context.Background(), streams.SliceStream(events))
	require.NoError(t, err)

	actual, err := streams.All(stream)
	require.NoError(t, err)
	require.Len(t, actual, 4)

	require.NotNil(t, actual[0].PreviousResponseID)
	require.Equal(t, "resp_prev_123", *actual[0].PreviousResponseID)

	require.NotNil(t, actual[1].PreviousResponseID)
	require.Equal(t, "resp_prev_123", *actual[1].PreviousResponseID)

	require.NotNil(t, actual[2].PreviousResponseID)
	require.Equal(t, "resp_prev_123", *actual[2].PreviousResponseID)
	require.Equal(t, llm.DoneResponse, actual[3])
}

func TestOutboundTransformer_TransformStream_ImageGenerationPartialImageUsesToolCall(t *testing.T) {
	trans, err := NewOutboundTransformer("https://api.openai.com", "test-api-key")
	require.NoError(t, err)

	events := []*httpclient.StreamEvent{
		{
			Type: "response.created",
			Data: []byte(`{
				"type":"response.created",
				"response":{
					"id":"resp_img_stream",
					"object":"response",
					"created_at":1700000000,
					"model":"gpt-5.4",
					"status":"in_progress",
					"output":[]
				}
			}`),
		},
		{
			Type: "response.output_item.added",
			Data: []byte(`{
				"type":"response.output_item.added",
				"output_index":0,
				"item":{
					"id":"ig_123",
					"type":"image_generation_call",
					"status":"in_progress"
				}
			}`),
		},
		{
			Type: "response.image_generation_call.generating",
			Data: []byte(`{
				"type":"response.image_generation_call.generating",
				"output_index":0,
				"item_id":"ig_123"
			}`),
		},
		{
			Type: "response.image_generation_call.partial_image",
			Data: []byte(`{
				"type":"response.image_generation_call.partial_image",
				"output_index":0,
				"item_id":"ig_123",
				"background":"opaque",
				"output_format":"png",
				"quality":"high",
				"size":"1024x1024",
				"revised_prompt":"A watercolor fox under moonlight",
				"partial_image_b64":"base64data"
			}`),
		},
	}

	stream, err := trans.TransformStream(context.Background(), streams.SliceStream(events))
	require.NoError(t, err)

	actual, err := streams.All(stream)
	require.NoError(t, err)
	require.Len(t, actual, 5)

	added := findImageGenerationToolCall(actual, string(StreamEventTypeOutputItemAdded))
	require.NotNil(t, added)
	require.Equal(t, "ig_123", added.ID)
	require.Equal(t, "in_progress", added.Status)

	generating := findImageGenerationToolCall(actual, string(StreamEventTypeImageGenerationGenerating))
	require.NotNil(t, generating)
	require.Equal(t, "ig_123", generating.ID)
	require.Equal(t, "generating", generating.Status)

	partial := findImageGenerationToolCall(actual, string(StreamEventTypeImageGenerationPartialImage))
	require.NotNil(t, partial)
	require.Equal(t, "ig_123", partial.ID)
	require.Equal(t, "generating", partial.Status)
	require.Equal(t, "base64data", partial.PartialImageB64)
	require.Equal(t, "opaque", partial.Background)
	require.Equal(t, "png", partial.OutputFormat)
	require.Equal(t, "high", partial.Quality)
	require.Equal(t, "1024x1024", partial.Size)
	require.Equal(t, "A watercolor fox under moonlight", partial.RevisedPrompt)
	require.Equal(t, llm.DoneResponse, actual[4])
}

func TestOutboundTransformer_TransformStream_ImageGenerationOutputItemDoneUsesToolCall(t *testing.T) {
	trans, err := NewOutboundTransformer("https://api.openai.com", "test-api-key")
	require.NoError(t, err)

	events := []*httpclient.StreamEvent{
		{
			Type: "response.created",
			Data: []byte(`{
				"type":"response.created",
				"response":{
					"id":"resp_img_stream2",
					"object":"response",
					"created_at":1700000000,
					"model":"gpt-5.4",
					"status":"in_progress",
					"output":[]
				}
			}`),
		},
		{
			Type: "response.output_item.added",
			Data: []byte(`{
				"type":"response.output_item.added",
				"output_index":0,
				"item":{
					"id":"ig_234",
					"type":"image_generation_call",
					"status":"in_progress"
				}
			}`),
		},
		{
			Type: "response.image_generation_call.partial_image",
			Data: []byte(`{
				"type":"response.image_generation_call.partial_image",
				"output_index":0,
				"item_id":"ig_234",
				"status":"generating",
				"background":"opaque",
				"output_format":"png",
				"quality":"high",
				"size":"853x1844",
				"revised_prompt":"a revised prompt",
				"partial_image_b64":"base64data"
			}`),
		},
		{
			Type: "response.output_item.done",
			Data: []byte(`{
				"type":"response.output_item.done",
				"output_index":0,
				"item":{
					"id":"ig_234",
					"type":"image_generation_call",
					"status":"completed",
					"result":"base64data",
					"background":"opaque",
					"output_format":"png",
					"quality":"high",
					"size":"853x1844",
					"revised_prompt":"done revised prompt"
				}
			}`),
		},
	}

	stream, err := trans.TransformStream(context.Background(), streams.SliceStream(events))
	require.NoError(t, err)

	actual, err := streams.All(stream)
	require.NoError(t, err)
	require.Len(t, actual, 5)

	done := findImageGenerationToolCall(actual, string(StreamEventTypeOutputItemDone))
	require.NotNil(t, done)
	require.Equal(t, "ig_234", done.ID)
	require.Equal(t, "completed", done.Status)
	require.Equal(t, "base64data", done.Result)
	require.Equal(t, "opaque", done.Background)
	require.Equal(t, "png", done.OutputFormat)
	require.Equal(t, "high", done.Quality)
	require.Equal(t, "853x1844", done.Size)
	require.Equal(t, "done revised prompt", done.RevisedPrompt)
	require.Equal(t, llm.DoneResponse, actual[4])
}

func TestOutboundTransformer_TransformStream_ImageGenerationResponseCompletedUsesToolCall(t *testing.T) {
	trans, err := NewOutboundTransformer("https://api.openai.com", "test-api-key")
	require.NoError(t, err)

	events := []*httpclient.StreamEvent{
		{
			Type: "response.created",
			Data: []byte(`{"type":"response.created","response":{"id":"resp_img_stream4","object":"response","created_at":1700000000,"model":"gpt-5.4","status":"in_progress","output":[]}}`),
		},
		{
			Type: "response.output_item.added",
			Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"id":"ig_456","type":"image_generation_call","status":"in_progress"}}`),
		},
		{
			Type: "response.image_generation_call.partial_image",
			Data: []byte(`{"type":"response.image_generation_call.partial_image","output_index":0,"item_id":"ig_456","status":"generating","partial_image_b64":"base64data"}`),
		},
		{
			Type: "response.output_item.done",
			Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"id":"ig_456","type":"image_generation_call","status":"generating","result":"base64data","background":"opaque","output_format":"png","quality":"high","size":"1024x1024"}}`),
		},
		{
			Type: "response.completed",
			Data: []byte(`{"type":"response.completed","response":{"id":"resp_img_stream4","object":"response","created_at":1700000000,"model":"gpt-5.4","status":"completed","output":[{"id":"ig_456","type":"image_generation_call","status":"generating","result":"base64data","background":"opaque","output_format":"png","quality":"high","size":"1024x1024","revised_prompt":"completed revised prompt"}]}}`),
		},
	}

	stream, err := trans.TransformStream(context.Background(), streams.SliceStream(events))
	require.NoError(t, err)

	actual, err := streams.All(stream)
	require.NoError(t, err)
	require.Len(t, actual, 7)

	completed := findImageGenerationToolCall(actual, string(StreamEventTypeResponseCompleted))
	require.NotNil(t, completed)
	require.Equal(t, "ig_456", completed.ID)
	require.Equal(t, "generating", completed.Status)
	require.Equal(t, "base64data", completed.Result)
	require.Equal(t, "opaque", completed.Background)
	require.Equal(t, "png", completed.OutputFormat)
	require.Equal(t, "high", completed.Quality)
	require.Equal(t, "1024x1024", completed.Size)
	require.Equal(t, "completed revised prompt", completed.RevisedPrompt)
	require.Equal(t, llm.DoneResponse, actual[6])
}

func findImageGenerationToolCall(responses []*llm.Response, eventType string) *llm.ImageGenerationToolCall {
	for _, resp := range responses {
		if resp == nil || resp == llm.DoneResponse {
			continue
		}
		for _, choice := range resp.Choices {
			if choice.Delta == nil {
				continue
			}
			for i := range choice.Delta.ToolCalls {
				tc := &choice.Delta.ToolCalls[i]
				if tc.ImageGenerationToolCall != nil && tc.ImageGenerationToolCall.EventType == eventType {
					return tc.ImageGenerationToolCall
				}
			}
		}
	}

	return nil
}

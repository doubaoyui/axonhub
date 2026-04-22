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

func TestOutboundTransformer_TransformStream_ImageGenerationPartialImagePreservesRevisedPrompt(t *testing.T) {
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
			Type: "response.output_item.done",
			Data: []byte(`{
				"type":"response.output_item.done",
				"output_index":0,
				"item":{
					"id":"ig_123",
					"type":"image_generation_call",
					"status":"completed",
					"background":"opaque",
					"output_format":"png",
					"quality":"high",
					"size":"1024x1024",
					"revised_prompt":"A watercolor fox under moonlight"
				}
			}`),
		},
		{
			Type: "response.image_generation_call.partial_image",
			Data: []byte(`{
				"type":"response.image_generation_call.partial_image",
				"output_index":0,
				"item_id":"ig_123",
				"partial_image_b64":"base64data"
			}`),
		},
	}

	stream, err := trans.TransformStream(context.Background(), streams.SliceStream(events))
	require.NoError(t, err)

	actual, err := streams.All(stream)
	require.NoError(t, err)
	require.Len(t, actual, 3)

	var part llm.MessageContentPart
	var foundPart bool
	for _, resp := range actual {
		if resp == nil || resp == llm.DoneResponse || len(resp.Choices) == 0 {
			continue
		}
		choice := resp.Choices[0]
		if choice.Delta != nil && len(choice.Delta.Content.MultipleContent) > 0 {
			part = choice.Delta.Content.MultipleContent[0]
			foundPart = true
		}
	}

	require.True(t, foundPart)
	require.NotNil(t, part.ImageURL)
	require.Equal(t, "data:image/png;base64,base64data", part.ImageURL.URL)
	require.NotNil(t, part.TransformerMetadata)
	require.Equal(t, "opaque", part.TransformerMetadata["background"])
	require.Equal(t, "png", part.TransformerMetadata["output_format"])
	require.Equal(t, "high", part.TransformerMetadata["quality"])
	require.Equal(t, "1024x1024", part.TransformerMetadata["size"])
	require.Equal(t, "A watercolor fox under moonlight", part.TransformerMetadata["revised_prompt"])

	if actual[1].Choices[0].TransformerMetadata != nil {
		if rawUpdates, ok := actual[1].Choices[0].TransformerMetadata["image_generation_item_updates"]; ok && rawUpdates != nil {
			updates := rawUpdates.(map[string]any)
			require.Equal(t, "A watercolor fox under moonlight", updates["ig_123"].(map[string]any)["revised_prompt"])
		}
	}
	require.Equal(t, llm.DoneResponse, actual[2])
}

func TestOutboundTransformer_TransformStream_ImageGenerationPartialImageUsesTopLevelFields(t *testing.T) {
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
	}

	stream, err := trans.TransformStream(context.Background(), streams.SliceStream(events))
	require.NoError(t, err)

	actual, err := streams.All(stream)
	require.NoError(t, err)
	require.Len(t, actual, 3)

	part := actual[1].Choices[0].Delta.Content.MultipleContent[0]
	require.NotNil(t, part.TransformerMetadata)
	require.Equal(t, "opaque", part.TransformerMetadata["background"])
	require.Equal(t, "png", part.TransformerMetadata["output_format"])
	require.Equal(t, "high", part.TransformerMetadata["quality"])
	require.Equal(t, "853x1844", part.TransformerMetadata["size"])
	require.Equal(t, "a revised prompt", part.TransformerMetadata["revised_prompt"])
	require.Equal(t, llm.DoneResponse, actual[2])
}

func TestOutboundTransformer_TransformStream_ImageGenerationOutputItemDoneEmitsLateMetadataUpdate(t *testing.T) {
	trans, err := NewOutboundTransformer("https://api.openai.com", "test-api-key")
	require.NoError(t, err)

	events := []*httpclient.StreamEvent{
		{
			Type: "response.created",
			Data: []byte(`{"type":"response.created","response":{"id":"resp_img_stream3","object":"response","created_at":1700000000,"model":"gpt-5.4","status":"in_progress","output":[]}}`),
		},
		{
			Type: "response.output_item.added",
			Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"id":"ig_345","type":"image_generation_call","status":"in_progress"}}`),
		},
		{
			Type: "response.image_generation_call.partial_image",
			Data: []byte(`{"type":"response.image_generation_call.partial_image","output_index":0,"item_id":"ig_345","status":"generating","partial_image_b64":"base64data"}`),
		},
		{
			Type: "response.output_item.done",
			Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"id":"ig_345","type":"image_generation_call","status":"completed","revised_prompt":"late revised prompt"}}`),
		},
	}

	stream, err := trans.TransformStream(context.Background(), streams.SliceStream(events))
	require.NoError(t, err)

	actual, err := streams.All(stream)
	require.NoError(t, err)
	require.Len(t, actual, 4)

	updates := actual[2].Choices[0].TransformerMetadata["image_generation_item_updates"].(map[string]any)
	require.Equal(t, "late revised prompt", updates["ig_345"].(map[string]any)["revised_prompt"])
	require.Equal(t, llm.DoneResponse, actual[3])
}

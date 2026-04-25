package responses

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
)

func TestResponsesWebSearchCallStreamToLLMToolCall(t *testing.T) {
	events := []*httpclient.StreamEvent{
		{
			Type: "response.created",
			Data: []byte(`{"type":"response.created","response":{"id":"resp_ws","model":"gpt-5.4","created_at":1700000000}}`),
		},
		{
			Type: "response.output_item.added",
			Data: []byte(`{"type":"response.output_item.added","output_index":0,"item":{"id":"ws_123","type":"web_search_call","status":"in_progress"}}`),
		},
		{
			Type: "response.web_search_call.searching",
			Data: []byte(`{"type":"response.web_search_call.searching","output_index":0,"item_id":"ws_123"}`),
		},
		{
			Type: "response.web_search_call.completed",
			Data: []byte(`{"type":"response.web_search_call.completed","output_index":0,"item_id":"ws_123"}`),
		},
		{
			Type: "response.output_item.done",
			Data: []byte(`{"type":"response.output_item.done","output_index":0,"item":{"id":"ws_123","type":"web_search_call","status":"completed","action":{"type":"search","query":"Manchester United latest results","queries":["Manchester United latest results"]}}}`),
		},
		{
			Type: "response.completed",
			Data: []byte(`{"type":"response.completed","response":{"id":"resp_ws","model":"gpt-5.4","created_at":1700000000,"status":"completed"}}`),
		},
	}

	trans, err := NewOutboundTransformer("https://api.openai.com", "test-api-key")
	require.NoError(t, err)

	stream, err := trans.TransformStream(t.Context(), streams.SliceStream(events))
	require.NoError(t, err)

	responses, err := streams.All(stream)
	require.NoError(t, err)

	var webSearchCall *llm.ToolCall
	var finishReason *string
	for _, resp := range responses {
		for _, choice := range resp.Choices {
			if choice.FinishReason != nil {
				finishReason = choice.FinishReason
			}
			if choice.Delta == nil {
				continue
			}
			for i := range choice.Delta.ToolCalls {
				tc := &choice.Delta.ToolCalls[i]
				if tc.WebSearchToolCall != nil {
					webSearchCall = tc
				}
			}
		}
	}

	require.NotNil(t, webSearchCall)
	require.Equal(t, llm.ToolTypeWebSearch, webSearchCall.Type)
	require.Equal(t, "ws_123", webSearchCall.WebSearchToolCall.ID)
	require.Equal(t, "completed", webSearchCall.WebSearchToolCall.Status)
	require.NotNil(t, webSearchCall.WebSearchToolCall.Action)
	require.NotNil(t, finishReason)
	require.Equal(t, "stop", *finishReason)
}

func TestLLMWebSearchToolCallStreamToResponsesEvents(t *testing.T) {
	action := map[string]any{
		"type":    "search",
		"query":   "Manchester United latest results",
		"queries": []any{"Manchester United latest results"},
	}
	llmResponses := []*llm.Response{
		{
			ID:      "resp_ws",
			Model:   "gpt-5.4",
			Created: 1700000000,
			Choices: []llm.Choice{
				{
					Index: 0,
					Delta: &llm.Message{
						ToolCalls: []llm.ToolCall{
							{
								ID:    "ws_123",
								Type:  llm.ToolTypeWebSearch,
								Index: 0,
								WebSearchToolCall: &llm.WebSearchToolCall{
									ID:     "ws_123",
									Status: "completed",
									Action: action,
								},
							},
						},
					},
				},
			},
		},
		{
			ID:      "resp_ws",
			Model:   "gpt-5.4",
			Created: 1700000000,
			Choices: []llm.Choice{
				{
					Index:        0,
					Delta:        &llm.Message{},
					FinishReason: loPtr("tool_calls"),
				},
			},
		},
	}

	stream, err := NewInboundTransformer().TransformStream(t.Context(), streams.SliceStream(llmResponses))
	require.NoError(t, err)

	events, err := streams.All(stream)
	require.NoError(t, err)

	eventTypes := make([]string, 0, len(events))
	var doneEvent StreamEvent
	for _, event := range events {
		eventTypes = append(eventTypes, event.Type)
		if event.Type == string(StreamEventTypeOutputItemDone) {
			require.NoError(t, json.Unmarshal(event.Data, &doneEvent))
		}
	}

	require.Contains(t, eventTypes, string(StreamEventTypeOutputItemAdded))
	require.Contains(t, eventTypes, string(StreamEventTypeWebSearchCallInProgress))
	require.Contains(t, eventTypes, string(StreamEventTypeWebSearchCallSearching))
	require.Contains(t, eventTypes, string(StreamEventTypeWebSearchCallCompleted))
	require.Contains(t, eventTypes, string(StreamEventTypeOutputItemDone))
	require.NotNil(t, doneEvent.Item)
	require.Equal(t, "web_search_call", doneEvent.Item.Type)
	require.Equal(t, action, doneEvent.Item.Action)
}

func loPtr[T any](v T) *T {
	return &v
}

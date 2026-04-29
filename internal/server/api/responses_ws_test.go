package api

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/streams"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

type fallbackExecutor struct{}

func (fallbackExecutor) Do(context.Context, *httpclient.Request) (*httpclient.Response, error) {
	return nil, errors.New("fallback Do should not be used")
}

func (fallbackExecutor) DoStream(context.Context, *httpclient.Request) (streams.Stream[*httpclient.StreamEvent], error) {
	return nil, errors.New("fallback DoStream should not be used")
}

func TestResponsesWSExecutor_ReusesUpstreamConnectionAndPreservesIncrementalFields(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}
	frames := make(chan map[string]any, 2)
	var connections atomic.Int32

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		require.Equal(t, "/v1/responses", r.URL.Path)
		require.Equal(t, "Bearer upstream-key", r.Header.Get("Authorization"))
		connections.Add(1)

		conn, err := upgrader.Upgrade(w, r, nil)
		require.NoError(t, err)
		defer conn.Close()

		responseID := 1
		for {
			_, payload, err := conn.ReadMessage()
			if err != nil {
				return
			}

			var frame map[string]any
			require.NoError(t, json.Unmarshal(payload, &frame))
			frames <- frame

			id := "resp-1"
			if responseID > 1 {
				id = "resp-2"
			}
			responseID++

			require.NoError(t, conn.WriteJSON(map[string]any{
				"type": "response.created",
				"response": map[string]any{
					"id":     id,
					"status": "in_progress",
					"output": []any{},
				},
			}))
			require.NoError(t, conn.WriteJSON(map[string]any{
				"type": "response.completed",
				"response": map[string]any{
					"id":     id,
					"status": "completed",
					"output": []any{},
				},
			}))
		}
	}))
	defer upstream.Close()

	executor := newResponsesWSExecutor(fallbackExecutor{})
	defer executor.Close()

	first := responsesWSRequestForTest(upstream.URL+"/v1/responses", []byte(`{"model":"gpt-5","stream":true,"generate":false,"input":[]}`))
	stream, err := executor.DoStream(context.Background(), first)
	require.NoError(t, err)
	require.NoError(t, drainResponsesWSStream(stream))

	second := responsesWSRequestForTest(upstream.URL+"/v1/responses", []byte(`{"model":"gpt-5","stream":true,"previous_response_id":"resp-1","input":[]}`))
	stream, err = executor.DoStream(context.Background(), second)
	require.NoError(t, err)
	require.NoError(t, drainResponsesWSStream(stream))

	require.Equal(t, int32(1), connections.Load())

	firstFrame := receiveResponsesWSFrame(t, frames)
	require.Equal(t, "response.create", firstFrame["type"])
	require.Equal(t, false, firstFrame["generate"])
	require.Nil(t, firstFrame["previous_response_id"])

	secondFrame := receiveResponsesWSFrame(t, frames)
	require.Equal(t, "response.create", secondFrame["type"])
	require.Equal(t, "resp-1", secondFrame["previous_response_id"])
}

func TestResponsesWSExecutor_MapsWrappedUpstreamError(t *testing.T) {
	upgrader := websocket.Upgrader{CheckOrigin: func(*http.Request) bool { return true }}

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		require.NoError(t, err)
		defer conn.Close()

		_, _, err = conn.ReadMessage()
		require.NoError(t, err)
		require.NoError(t, conn.WriteJSON(map[string]any{
			"type":   "error",
			"status": http.StatusBadRequest,
			"error": map[string]any{
				"type":    "invalid_request_error",
				"message": "Model does not support image inputs",
			},
			"headers": map[string]any{
				"x-request-id": "req_123",
				"x-retry":      1,
			},
		}))
	}))
	defer upstream.Close()

	executor := newResponsesWSExecutor(fallbackExecutor{})
	defer executor.Close()

	request := responsesWSRequestForTest(upstream.URL+"/v1/responses", []byte(`{"model":"gpt-5","stream":true,"input":[]}`))
	stream, err := executor.DoStream(context.Background(), request)
	require.NoError(t, err)

	require.False(t, stream.Next())
	err = stream.Err()
	require.Error(t, err)

	var httpErr *httpclient.Error
	require.ErrorAs(t, err, &httpErr)
	require.Equal(t, http.StatusBadRequest, httpErr.StatusCode)
	require.Contains(t, string(httpErr.Body), "Model does not support image inputs")
	require.Equal(t, "req_123", httpErr.Headers.Get("x-request-id"))
	require.Equal(t, "1", httpErr.Headers.Get("x-retry"))
}

func responsesWSRequestForTest(url string, body []byte) *httpclient.Request {
	return &httpclient.Request{
		Method:  http.MethodPost,
		URL:     url,
		Headers: http.Header{"Authorization": []string{"Bearer upstream-key"}},
		Body:    body,
		Metadata: map[string]string{
			responses.ResponsesWebSocketMetadataKey: "true",
		},
	}
}

func drainResponsesWSStream(stream streams.Stream[*httpclient.StreamEvent]) error {
	for stream.Next() {
		if stream.Current().Type == responsesWSCompleteEvent {
			return stream.Close()
		}
	}
	if err := stream.Err(); err != nil {
		return err
	}
	return stream.Close()
}

func receiveResponsesWSFrame(t *testing.T, frames <-chan map[string]any) map[string]any {
	t.Helper()

	select {
	case frame := <-frames:
		return frame
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for upstream websocket frame")
		return nil
	}
}

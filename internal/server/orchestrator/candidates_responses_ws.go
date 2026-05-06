package orchestrator

import (
	"context"
	"net/http"

	"github.com/samber/lo"

	"github.com/looplj/axonhub/internal/log"
	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

type ResponsesWebSocketSelector struct {
	wrapped CandidateSelector
}

func WithResponsesWebSocketSelector(wrapped CandidateSelector) *ResponsesWebSocketSelector {
	return &ResponsesWebSocketSelector{wrapped: wrapped}
}

func (s *ResponsesWebSocketSelector) Select(ctx context.Context, req *llm.Request) ([]*ChannelModelsCandidate, error) {
	candidates, err := s.wrapped.Select(ctx, req)
	if err != nil {
		return nil, err
	}

	if !isResponsesWebSocketRequest(req) {
		if !isPlainResponsesRequest(req) {
			return candidates, nil
		}

		filtered := lo.Filter(candidates, func(c *ChannelModelsCandidate, _ int) bool {
			return !hasResponsesWebSocketEnabled(c)
		})

		if log.DebugEnabled(ctx) {
			log.Debug(ctx, "filtered responses websocket candidates for plain responses request",
				log.Int("total_candidates", len(candidates)),
				log.Int("compatible_candidates", len(filtered)),
			)
		}

		return filtered, nil
	}

	filtered := lo.Filter(candidates, func(c *ChannelModelsCandidate, _ int) bool {
		return supportsResponsesWebSocket(c)
	})
	if len(filtered) == 0 {
		return nil, responsesWebSocketUnsupportedChannelError(req)
	}

	if log.DebugEnabled(ctx) {
		log.Debug(ctx, "filtered candidates for responses websocket",
			log.Int("total_candidates", len(candidates)),
			log.Int("compatible_candidates", len(filtered)),
		)
	}

	return filtered, nil
}

func isPlainResponsesRequest(req *llm.Request) bool {
	if req == nil {
		return false
	}

	return req.APIFormat == llm.APIFormatOpenAIResponse
}

func hasResponsesWebSocketEnabled(candidate *ChannelModelsCandidate) bool {
	return candidate != nil &&
		candidate.Channel != nil &&
		candidate.Channel.Settings != nil &&
		candidate.Channel.Settings.SupportsResponsesWebSocket
}

func supportsResponsesWebSocket(candidate *ChannelModelsCandidate) bool {
	if !hasResponsesWebSocketEnabled(candidate) ||
		candidate.Channel.Outbound == nil {
		return false
	}

	return candidate.Channel.Outbound.APIFormat() == llm.APIFormatOpenAIResponse
}

func isResponsesWebSocketRequest(req *llm.Request) bool {
	if req == nil || req.TransformerMetadata == nil {
		return false
	}

	switch value := req.TransformerMetadata[responses.ResponsesWebSocketMetadataKey].(type) {
	case bool:
		return value
	case string:
		return value == "true"
	default:
		return false
	}
}

func responsesWebSocketUnsupportedChannelError(req *llm.Request) *httpclient.Error {
	url := ""
	if req != nil && req.RawRequest != nil {
		url = req.RawRequest.URL
	}

	return &httpclient.Error{
		Method:     http.MethodGet,
		URL:        url,
		StatusCode: http.StatusUpgradeRequired,
		Status:     http.StatusText(http.StatusUpgradeRequired),
		Body: []byte(
			`{"error":{"type":"invalid_request_error","message":"selected upstream does not support Responses WebSocket"}}`,
		),
	}
}

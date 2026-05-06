package orchestrator

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/internal/ent"
	"github.com/looplj/axonhub/internal/objects"
	"github.com/looplj/axonhub/internal/server/biz"
	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

func TestResponsesWebSocketSelector_FiltersExplicitlySupportedChannels(t *testing.T) {
	candidates := []*ChannelModelsCandidate{
		responseWSCandidateForTest(1, false),
		responseWSCandidateForTest(2, true),
		responseWSCandidateForTest(3, false),
	}
	selector := WithResponsesWebSocketSelector(&staticChannelSelector{candidates: candidates})

	req := &llm.Request{
		Model: "gpt-5",
		TransformerMetadata: map[string]any{
			responses.ResponsesWebSocketMetadataKey: true,
		},
	}

	got, err := selector.Select(context.Background(), req)
	require.NoError(t, err)
	require.Len(t, got, 1)
	require.Equal(t, 2, got[0].Channel.ID)
}

func TestResponsesWebSocketSelector_DoesNotFilterPlainResponsesRequests(t *testing.T) {
	candidates := []*ChannelModelsCandidate{
		responseWSCandidateForTest(1, false),
		responseWSCandidateForTest(2, true),
	}
	selector := WithResponsesWebSocketSelector(&staticChannelSelector{candidates: candidates})

	got, err := selector.Select(context.Background(), &llm.Request{Model: "gpt-5"})
	require.NoError(t, err)
	require.Len(t, got, 2)
}

func TestResponsesWebSocketSelector_FiltersWSChannelsForPlainResponsesRequests(t *testing.T) {
	candidates := []*ChannelModelsCandidate{
		responseWSCandidateForTest(1, false),
		responseWSCandidateForTest(2, true),
		responseWSCandidateForTest(3, false),
	}
	selector := WithResponsesWebSocketSelector(&staticChannelSelector{candidates: candidates})

	got, err := selector.Select(context.Background(), &llm.Request{
		Model:     "gpt-5",
		APIFormat: llm.APIFormatOpenAIResponse,
	})

	require.NoError(t, err)
	require.Len(t, got, 2)
	require.Equal(t, 1, got[0].Channel.ID)
	require.Equal(t, 3, got[1].Channel.ID)
}

func TestResponsesWebSocketSelector_FiltersMisconfiguredWSChannelsForPlainResponsesRequests(t *testing.T) {
	candidates := []*ChannelModelsCandidate{
		responseWSCandidateForTest(1, false),
		responseWSCandidateForTest(2, true),
	}
	candidates[1].Channel.Outbound = &mockTransformer{apiFormat: llm.APIFormatOpenAIChatCompletion}
	selector := WithResponsesWebSocketSelector(&staticChannelSelector{candidates: candidates})

	got, err := selector.Select(context.Background(), &llm.Request{
		Model:     "gpt-5",
		APIFormat: llm.APIFormatOpenAIResponse,
	})

	require.NoError(t, err)
	require.Len(t, got, 1)
	require.Equal(t, 1, got[0].Channel.ID)
}

func TestResponsesWebSocketSelector_ReturnsEmptyWhenOnlyWSChannelsForPlainResponsesRequests(t *testing.T) {
	candidates := []*ChannelModelsCandidate{
		responseWSCandidateForTest(1, true),
		responseWSCandidateForTest(2, true),
	}
	selector := WithResponsesWebSocketSelector(&staticChannelSelector{candidates: candidates})

	got, err := selector.Select(context.Background(), &llm.Request{
		Model:     "gpt-5",
		APIFormat: llm.APIFormatOpenAIResponse,
	})

	require.NoError(t, err)
	require.Empty(t, got)
}

func TestResponsesWebSocketSelector_ReturnsUpgradeRequiredWhenUnsupported(t *testing.T) {
	candidates := []*ChannelModelsCandidate{
		responseWSCandidateForTest(1, false),
		responseWSCandidateForTest(2, false),
	}
	selector := WithResponsesWebSocketSelector(&staticChannelSelector{candidates: candidates})

	_, err := selector.Select(context.Background(), &llm.Request{
		Model: "gpt-5",
		RawRequest: &httpclient.Request{
			URL: "https://example.com/v1/responses",
		},
		TransformerMetadata: map[string]any{
			responses.ResponsesWebSocketMetadataKey: true,
		},
	})

	var httpErr *httpclient.Error
	require.True(t, errors.As(err, &httpErr))
	require.Equal(t, http.StatusUpgradeRequired, httpErr.StatusCode)
	require.Contains(t, string(httpErr.Body), "selected upstream does not support Responses WebSocket")
}

func TestResponsesWebSocketSelector_RejectsNonResponsesOutboundEvenWhenEnabled(t *testing.T) {
	candidates := []*ChannelModelsCandidate{
		responseWSCandidateForTest(1, true),
	}
	candidates[0].Channel.Outbound = &mockTransformer{apiFormat: llm.APIFormatOpenAIChatCompletion}
	selector := WithResponsesWebSocketSelector(&staticChannelSelector{candidates: candidates})

	_, err := selector.Select(context.Background(), &llm.Request{
		Model: "gpt-5",
		TransformerMetadata: map[string]any{
			responses.ResponsesWebSocketMetadataKey: true,
		},
	})

	var httpErr *httpclient.Error
	require.True(t, errors.As(err, &httpErr))
	require.Equal(t, http.StatusUpgradeRequired, httpErr.StatusCode)
}

func TestResponsesWebSocketSelector_RejectsCompactResponsesOutbound(t *testing.T) {
	candidates := []*ChannelModelsCandidate{
		responseWSCandidateForTest(1, true),
	}
	candidates[0].Channel.Outbound = &mockTransformer{apiFormat: llm.APIFormatOpenAIResponseCompact}
	selector := WithResponsesWebSocketSelector(&staticChannelSelector{candidates: candidates})

	_, err := selector.Select(context.Background(), &llm.Request{
		Model: "gpt-5",
		TransformerMetadata: map[string]any{
			responses.ResponsesWebSocketMetadataKey: true,
		},
	})

	var httpErr *httpclient.Error
	require.True(t, errors.As(err, &httpErr))
	require.Equal(t, http.StatusUpgradeRequired, httpErr.StatusCode)
}

func responseWSCandidateForTest(id int, supportsResponsesWebSocket bool) *ChannelModelsCandidate {
	return &ChannelModelsCandidate{
		Channel: &biz.Channel{
			Channel: &ent.Channel{
				ID:   id,
				Name: "channel",
				Settings: &objects.ChannelSettings{
					SupportsResponsesWebSocket: supportsResponsesWebSocket,
				},
			},
			Outbound: &mockTransformer{apiFormat: llm.APIFormatOpenAIResponse},
		},
		Models: []biz.ChannelModelEntry{{RequestModel: "gpt-5", ActualModel: "gpt-5"}},
	}
}

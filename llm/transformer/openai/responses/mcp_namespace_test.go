package responses

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
	"github.com/looplj/axonhub/llm/transformer/shared"
)

func TestResponsesMCPNamespaceToolFlattensToLLMFunction(t *testing.T) {
	tools, err := convertToolsToLLM([]Tool{
		{
			Type: "namespace",
			Name: "mcp__arthas__",
			Tools: []Tool{
				{
					Type:        "function",
					Name:        "excelPeek",
					Description: "Peek Excel.",
					Parameters:  map[string]any{"type": "object"},
				},
			},
		},
	})
	require.NoError(t, err)
	require.Len(t, tools, 1)
	require.Equal(t, "function", tools[0].Type)
	require.Equal(t, "mcp__arthas__responses_unit__excelPeek", tools[0].Function.Name)
}

func TestResponsesMCPEncodedFunctionToolRestoresNamespace(t *testing.T) {
	params := json.RawMessage(`{"type":"object"}`)
	tool := convertFunctionToTool(llm.Tool{
		Type: "function",
		Function: llm.Function{
			Name:        "mcp__arthas__responses_unit__excelPeek",
			Description: "Peek Excel.",
			Parameters:  params,
		},
	})

	require.Equal(t, "namespace", tool.Type)
	require.Equal(t, "mcp__arthas__", tool.Name)
	require.Len(t, tool.Tools, 1)
	require.Equal(t, "function", tool.Tools[0].Type)
	require.Equal(t, "excelPeek", tool.Tools[0].Name)
}

func TestResponsesMCPFunctionCallRoundTrip(t *testing.T) {
	msg := convertOutputToMessage([]Item{
		{
			Type:      "function_call",
			CallID:    "call_1",
			Name:      "excelPeek",
			Namespace: "mcp__arthas__",
			Arguments: `{"path":"a.xlsx"}`,
		},
	}, shared.TransportScope{}, nil)

	require.Len(t, msg.ToolCalls, 1)
	require.Equal(t, "mcp__arthas__responses_unit__excelPeek", msg.ToolCalls[0].Function.Name)

	resp := convertToResponsesAPIResponse(&llm.Response{
		ID: "resp_1",
		Choices: []llm.Choice{
			{
				Message: &llm.Message{
					Role:      "assistant",
					ToolCalls: msg.ToolCalls,
				},
			},
		},
	})

	require.Len(t, resp.Output, 1)
	require.Equal(t, "function_call", resp.Output[0].Type)
	require.Equal(t, "mcp__arthas__", resp.Output[0].Namespace)
	require.Equal(t, "excelPeek", resp.Output[0].Name)
}

func TestResponsesMCPNamespaceToolsCoalesce(t *testing.T) {
	tools := coalesceResponsesNamespaceTools([]Tool{
		{Type: "namespace", Name: "mcp__arthas__", Tools: []Tool{{Type: "function", Name: "excelPeek"}}},
		{Type: "namespace", Name: "mcp__arthas__", Tools: []Tool{{Type: "function", Name: "datasetRegister"}}},
	})

	require.Len(t, tools, 1)
	require.Equal(t, "namespace", tools[0].Type)
	require.Equal(t, "mcp__arthas__", tools[0].Name)
	require.Len(t, tools[0].Tools, 2)
	require.Equal(t, "excelPeek", tools[0].Tools[0].Name)
	require.Equal(t, "datasetRegister", tools[0].Tools[1].Name)
}

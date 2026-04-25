package responses

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/looplj/axonhub/llm"
)

func TestResponsesWebSearchToolRoundTrip(t *testing.T) {
	externalWebAccess := true

	tools, err := convertToolsToLLM([]Tool{
		{
			Type:               "web_search",
			ExternalWebAccess:  &externalWebAccess,
			SearchContentTypes: []string{"text", "image"},
		},
	})
	require.NoError(t, err)
	require.Len(t, tools, 1)
	require.Equal(t, llm.ToolTypeWebSearch, tools[0].Type)
	require.NotNil(t, tools[0].WebSearch)
	require.NotNil(t, tools[0].WebSearch.ExternalWebAccess)
	require.True(t, *tools[0].WebSearch.ExternalWebAccess)
	require.Equal(t, []string{"text", "image"}, tools[0].WebSearch.SearchContentTypes)

	tool := convertWebSearchToTool(tools[0])
	require.Equal(t, "web_search", tool.Type)
	require.NotNil(t, tool.ExternalWebAccess)
	require.True(t, *tool.ExternalWebAccess)
	require.Equal(t, []string{"text", "image"}, tool.SearchContentTypes)
}

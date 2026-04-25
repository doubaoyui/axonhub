package responses

import "strings"

const responsesMCPNamespaceUnitMarker = "responses_unit__"

func encodeResponsesMCPToolName(namespace, name string) string {
	if namespace == "" || name == "" {
		return name
	}

	return namespace + responsesMCPNamespaceUnitMarker + name
}

func decodeResponsesMCPToolName(name string) (namespace string, toolName string, ok bool) {
	idx := strings.Index(name, responsesMCPNamespaceUnitMarker)
	if idx <= 0 {
		return "", name, false
	}

	namespace = name[:idx]
	toolName = name[idx+len(responsesMCPNamespaceUnitMarker):]
	if namespace == "" || toolName == "" || !strings.HasPrefix(namespace, "mcp__") {
		return "", name, false
	}

	return namespace, toolName, true
}

func encodeResponsesFunctionCallName(namespace, name string) string {
	if namespace == "" {
		return name
	}

	return encodeResponsesMCPToolName(namespace, name)
}

func decodeResponsesFunctionCallName(name string) (namespace string, toolName string) {
	if namespace, toolName, ok := decodeResponsesMCPToolName(name); ok {
		return namespace, toolName
	}

	return "", name
}

func coalesceResponsesNamespaceTools(tools []Tool) []Tool {
	if len(tools) == 0 {
		return tools
	}

	result := make([]Tool, 0, len(tools))
	namespaceIndexes := map[string]int{}
	for _, tool := range tools {
		if tool.Type != "namespace" || tool.Name == "" {
			result = append(result, tool)
			continue
		}

		if idx, ok := namespaceIndexes[tool.Name]; ok {
			result[idx].Tools = append(result[idx].Tools, tool.Tools...)
			if result[idx].Description == "" {
				result[idx].Description = tool.Description
			}
			continue
		}

		namespaceIndexes[tool.Name] = len(result)
		result = append(result, tool)
	}

	return result
}

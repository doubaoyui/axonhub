package xmap

import (
	"encoding/json"

	"github.com/samber/lo"
)

// GetStringPtr extracts a *string value from a map[string]any.
func GetStringPtr(m map[string]any, key string) *string {
	if m == nil {
		return nil
	}

	if v, ok := m[key]; ok {
		switch vv := v.(type) {
		case string:
			return lo.ToPtr(vv)
		case *string:
			return vv
		default:
			return nil
		}
	}

	return nil
}

// GetInt64Ptr extracts a *int64 value from a map[string]any.
func GetInt64Ptr(m map[string]any, key string) *int64 {
	if m == nil {
		return nil
	}

	if v, ok := m[key]; ok {
		switch vv := v.(type) {
		case int64:
			return lo.ToPtr(vv)
		case *int64:
			return vv
		default:
			return nil
		}
	}

	return nil
}

// GetBoolPtr extracts a *bool value from a map[string]any.
func GetBoolPtr(m map[string]any, key string) *bool {
	if m == nil {
		return nil
	}

	if v, ok := m[key]; ok {
		switch vv := v.(type) {
		case bool:
			return lo.ToPtr(vv)
		case *bool:
			return vv
		default:
			return nil
		}
	}

	return nil
}

// GetStringSlice extracts a []string value from a map[string]any.
func GetStringSlice(m map[string]any, key string) []string {
	if m == nil {
		return nil
	}

	if v, ok := m[key]; ok {
		if slice, ok := v.([]string); ok {
			return slice
		}
	}

	return nil
}

// GetStringMap extracts a map[string]string value from a map[string]any.
func GetStringMap(m map[string]any, key string) map[string]string {
	if m == nil {
		return nil
	}

	if v, ok := m[key]; ok {
		switch stringMap := v.(type) {
		case map[string]string:
			return stringMap
		case map[string]any:
			result := make(map[string]string, len(stringMap))
			for k, rawValue := range stringMap {
				if value, ok := rawValue.(string); ok {
					result[k] = value
				}
			}
			if len(result) == 0 {
				return nil
			}
			return result
		case map[string]json.RawMessage:
			result := make(map[string]string, len(stringMap))
			for k, rawValue := range stringMap {
				var value string
				if err := json.Unmarshal(rawValue, &value); err == nil {
					result[k] = value
				}
			}
			if len(result) == 0 {
				return nil
			}
			return result
		}
	}

	return nil
}

func GetSlice[T any](m map[string]any, key string) []T {
	if m == nil {
		return nil
	}

	if v, ok := m[key]; ok {
		switch vv := v.(type) {
		case []T:
			return vv
		case []*T:
			return lo.FromSlicePtr(vv)
		default:
			return []T{}
		}
	}

	return nil
}

func GetSlicePtr[T any](m map[string]any, key string) []*T {
	if m == nil {
		return nil
	}

	if v, ok := m[key]; ok {
		switch vv := v.(type) {
		case []*T:
			return vv
		case []T:
			return lo.ToSlicePtr(vv)
		default:
			return nil
		}
	}

	return nil
}

func GetPtr[T any](m map[string]any, key string) *T {
	if m == nil {
		return nil
	}

	if v, ok := m[key]; ok {
		switch vv := v.(type) {
		case T:
			return lo.ToPtr(vv)
		case *T:
			return vv
		default:
			return nil
		}
	}

	return nil
}

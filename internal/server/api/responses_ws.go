package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"

	"github.com/looplj/axonhub/internal/log"
	"github.com/looplj/axonhub/llm/httpclient"
	"github.com/looplj/axonhub/llm/pipeline"
	"github.com/looplj/axonhub/llm/streams"
	"github.com/looplj/axonhub/llm/transformer/openai/codex"
	"github.com/looplj/axonhub/llm/transformer/openai/responses"
)

const (
	responsesWSCompleteEvent    = "response.completed"
	responsesWSFailedEvent      = "response.failed"
	responsesWSIncompleteEvent  = "response.incomplete"
	responsesWSWrappedErrorType = "error"
)

var responsesWSUpgrader = websocket.Upgrader{
	CheckOrigin: func(*http.Request) bool {
		return true
	},
	EnableCompression: true,
}

type responsesWSRequest struct {
	Type string `json:"type"`
}

type responsesWSCreateLogSummary struct {
	Type               string          `json:"type"`
	Generate           *bool           `json:"generate"`
	PreviousResponseID *string         `json:"previous_response_id"`
	Input              json.RawMessage `json:"input"`
}

type responsesWSClientMetadata struct {
	ClientMetadata map[string]string `json:"client_metadata"`
}

type responsesWSWrappedError struct {
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
	Type    string `json:"type,omitempty"`
}

type responsesWSWrappedErrorEvent struct {
	Type       string                     `json:"type"`
	Status     *int                       `json:"status,omitempty"`
	StatusCode *int                       `json:"status_code,omitempty"`
	Error      *responsesWSWrappedError   `json:"error,omitempty"`
	Headers    map[string]json.RawMessage `json:"headers,omitempty"`
}

func (handlers *OpenAIHandlers) CreateResponseWebSocket(c *gin.Context) {
	if !c.IsWebsocket() {
		c.JSON(http.StatusUpgradeRequired, gin.H{
			"error": gin.H{
				"type":    "invalid_request_error",
				"message": "websocket upgrade required",
			},
		})
		return
	}

	conn, err := responsesWSUpgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Warn(c.Request.Context(), "failed to upgrade responses websocket", log.Cause(err))
		return
	}
	defer func() {
		if err := conn.Close(); err != nil {
			log.Debug(c.Request.Context(), "failed to close responses websocket", log.Cause(err))
		}
	}()

	executor := newResponsesWSExecutor(handlers.ResponseCompletionHandlers.ChatCompletionOrchestrator.PipelineFactory.Executor)
	defer executor.Close()

	for {
		messageType, payload, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) ||
				websocket.IsUnexpectedCloseError(err) {
				log.Debug(c.Request.Context(), "responses websocket closed", log.Cause(err))
			}
			return
		}
		if messageType != websocket.TextMessage {
			if err := writeResponsesWSError(conn, http.StatusBadRequest, "invalid_request_error", "expected text websocket frame"); err != nil {
				return
			}
			continue
		}

		logResponsesWSCreateFrame(c.Request.Context(), payload)
		if err := handlers.handleResponseCreateWSMessage(c, conn, executor, payload); err != nil {
			if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
				log.Warn(c.Request.Context(), "responses websocket request failed", log.Cause(err))
			}
			return
		}
	}
}

func (handlers *OpenAIHandlers) handleResponseCreateWSMessage(c *gin.Context, conn *websocket.Conn, executor *responsesWSExecutor, payload []byte) error {
	var envelope responsesWSRequest
	if err := json.Unmarshal(payload, &envelope); err != nil {
		return writeResponsesWSError(conn, http.StatusBadRequest, "invalid_request_error", "invalid websocket request json")
	}
	if envelope.Type != "response.create" {
		return writeResponsesWSError(conn, http.StatusBadRequest, "invalid_request_error", "unsupported websocket request type")
	}

	body, err := stripResponsesWSType(payload)
	if err != nil {
		return writeResponsesWSError(conn, http.StatusBadRequest, "invalid_request_error", err.Error())
	}

	headers := cloneHeaders(c.Request.Header)
	mergeResponsesWSClientMetadataHeaders(headers, payload)

	req := &httpclient.Request{
		Method:      http.MethodPost,
		URL:         c.Request.URL.String(),
		Path:        c.Request.URL.Path,
		Query:       c.Request.URL.Query(),
		Headers:     headers,
		ContentType: "application/json",
		Body:        body,
		RawRequest:  c.Request,
		ClientIP:    c.ClientIP(),
		Metadata: map[string]string{
			responses.ResponsesWebSocketMetadataKey: "true",
		},
	}
	req.Headers.Set("Content-Type", "application/json")

	result, err := handlers.ResponseCompletionHandlers.ChatCompletionOrchestrator.ProcessWithExecutor(c.Request.Context(), req, executor)
	if err != nil {
		httpErr := handlers.ResponseCompletionHandlers.ChatCompletionOrchestrator.Inbound.TransformError(c.Request.Context(), err)
		return writeResponsesWSError(conn, httpErr.StatusCode, "api_error", strings.TrimSpace(string(httpErr.Body)))
	}
	if result.ChatCompletionStream == nil {
		return writeResponsesWSError(conn, http.StatusInternalServerError, "internal_error", "responses websocket expected a stream result")
	}
	defer func() {
		if err := result.ChatCompletionStream.Close(); err != nil {
			log.Debug(c.Request.Context(), "failed to close responses websocket stream", log.Cause(err))
		}
	}()

	return writeResponsesWSStream(c.Request.Context(), conn, result.ChatCompletionStream)
}

func writeResponsesWSStream(ctx context.Context, conn *websocket.Conn, stream streams.Stream[*httpclient.StreamEvent]) error {
	for stream.Next() {
		event := stream.Current()
		if event == nil || len(event.Data) == 0 {
			continue
		}
		if err := conn.WriteMessage(websocket.TextMessage, event.Data); err != nil {
			return err
		}
		if isResponsesWSTerminalEvent(event.Type) {
			logResponsesWSTerminalEvent(ctx, event.Type, event.Data)
			return nil
		}
	}
	if err := stream.Err(); err != nil && !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
		_ = writeResponsesWSError(conn, http.StatusBadGateway, "api_error", err.Error())
		return err
	}
	log.Debug(ctx, "responses websocket stream ended without explicit error")
	return nil
}

func stripResponsesWSType(payload []byte) ([]byte, error) {
	var body map[string]json.RawMessage
	if err := json.Unmarshal(payload, &body); err != nil {
		return nil, err
	}
	delete(body, "type")
	out, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("failed to encode response.create body: %w", err)
	}
	return out, nil
}

func mergeResponsesWSClientMetadataHeaders(headers http.Header, payload []byte) {
	var metadata responsesWSClientMetadata
	if err := json.Unmarshal(payload, &metadata); err != nil {
		return
	}
	if len(metadata.ClientMetadata) == 0 {
		return
	}
	for key, value := range metadata.ClientMetadata {
		trimmedValue := strings.TrimSpace(value)
		if trimmedValue == "" {
			continue
		}
		if strings.HasPrefix(strings.ToLower(key), "x-") {
			headers.Set(key, value)
		}
		if strings.EqualFold(key, codex.TurnMetadataHeader) {
			sessionID := codex.ExtractSessionIDFromTurnMetadata(trimmedValue)
			if sessionID != "" {
				headers.Set(codex.SessionHeader, sessionID)
			}
		}
	}
}

func writeResponsesWSError(conn *websocket.Conn, status int, code, message string) error {
	if message == "" {
		message = http.StatusText(status)
	}
	payload := map[string]any{
		"type":   "error",
		"status": status,
		"error": map[string]any{
			"type":    code,
			"message": message,
		},
	}
	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.TextMessage, data)
}

func cloneHeaders(headers http.Header) http.Header {
	cloned := make(http.Header, len(headers))
	for key, values := range headers {
		cloned[key] = append([]string(nil), values...)
	}
	return cloned
}

type responsesWSExecutor struct {
	base    pipeline.Executor
	conn    *websocket.Conn
	connKey string
	mu      sync.Mutex
}

func newResponsesWSExecutor(base pipeline.Executor) *responsesWSExecutor {
	return &responsesWSExecutor{base: base}
}

func (e *responsesWSExecutor) WithBase(base pipeline.Executor) pipeline.Executor {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.base = base
	return e
}

func (e *responsesWSExecutor) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.conn != nil {
		_ = e.conn.Close()
		e.conn = nil
	}
	e.connKey = ""
}

func (e *responsesWSExecutor) Do(ctx context.Context, request *httpclient.Request) (*httpclient.Response, error) {
	return e.base.Do(ctx, request)
}

func (e *responsesWSExecutor) DoStream(ctx context.Context, request *httpclient.Request) (streams.Stream[*httpclient.StreamEvent], error) {
	if request == nil || request.Metadata == nil || request.Metadata[responses.ResponsesWebSocketMetadataKey] != "true" {
		return e.base.DoStream(ctx, request)
	}
	conn, err := e.upstreamConn(ctx, request)
	if err != nil {
		return nil, err
	}
	requestBody := addResponsesWSType(request.Body)
	if err := conn.WriteMessage(websocket.TextMessage, requestBody); err != nil {
		e.Close()
		conn, err = e.upstreamConn(ctx, request)
		if err != nil {
			return nil, err
		}
		if err := conn.WriteMessage(websocket.TextMessage, requestBody); err != nil {
			e.Close()
			return nil, err
		}
	}
	return &upstreamResponsesWSStream{
		ctx:         ctx,
		executor:    e,
		conn:        conn,
		upstreamURL: request.URL,
	}, nil
}

func (e *responsesWSExecutor) upstreamConn(ctx context.Context, request *httpclient.Request) (*websocket.Conn, error) {
	wsURL, err := responsesWebSocketURL(request.URL)
	if err != nil {
		return nil, err
	}
	connKey := wsURL + "\n" + request.Headers.Get("Authorization")

	e.mu.Lock()
	defer e.mu.Unlock()
	if e.conn != nil && e.connKey == connKey {
		return e.conn, nil
	}
	if e.conn != nil {
		_ = e.conn.Close()
		e.conn = nil
		e.connKey = ""
	}

	headers := cloneHeaders(request.Headers)
	headers.Del("Accept")
	headers.Del("Content-Type")
	headers.Del("Connection")
	headers.Del("Upgrade")
	headers.Del("Sec-Websocket-Key")
	headers.Del("Sec-Websocket-Version")
	headers.Del("Sec-Websocket-Extensions")
	headers.Del("Sec-Websocket-Protocol")

	dialer := responsesWSDialer(e.base)
	conn, resp, err := dialer.DialContext(ctx, wsURL, headers)
	if err != nil {
		if resp != nil {
			defer resp.Body.Close()
			return nil, &httpclient.Error{
				Method:     http.MethodGet,
				URL:        wsURL,
				StatusCode: resp.StatusCode,
				Status:     resp.Status,
				Headers:    resp.Header,
			}
		}
		return nil, err
	}

	e.conn = conn
	e.connKey = connKey
	return conn, nil
}

func responsesWSDialer(base pipeline.Executor) websocket.Dialer {
	dialer := *websocket.DefaultDialer
	dialer.EnableCompression = true

	httpClient, ok := base.(*httpclient.HttpClient)
	if !ok || httpClient == nil || httpClient.GetNativeClient() == nil {
		return dialer
	}

	transport, ok := httpClient.GetNativeClient().Transport.(*http.Transport)
	if !ok || transport == nil {
		return dialer
	}

	if transport.Proxy != nil {
		dialer.Proxy = transport.Proxy
	}
	if transport.TLSClientConfig != nil {
		dialer.TLSClientConfig = transport.TLSClientConfig.Clone()
	}

	return dialer
}

type upstreamResponsesWSStream struct {
	ctx         context.Context
	executor    *responsesWSExecutor
	conn        *websocket.Conn
	upstreamURL string
	current     *httpclient.StreamEvent
	err         error
	closed      bool
	terminal    bool
}

func (s *upstreamResponsesWSStream) Next() bool {
	for {
		if s.err != nil || s.closed {
			return false
		}
		select {
		case <-s.ctx.Done():
			s.err = s.ctx.Err()
			_ = s.Close()
			return false
		default:
		}

		messageType, payload, err := s.conn.ReadMessage()
		if err != nil {
			if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				s.executor.Close()
				return false
			}
			s.err = err
			s.executor.Close()
			return false
		}
		if messageType != websocket.TextMessage {
			continue
		}

		if wrappedErr := mapResponsesWSWrappedErrorEvent(s.ctx, payload, s.upstreamURL); wrappedErr != nil {
			s.err = wrappedErr
			s.executor.Close()
			return false
		}

		eventType := responsesWSEventType(payload)
		s.current = &httpclient.StreamEvent{
			Type: eventType,
			Data: payload,
		}
		if isResponsesWSTerminalEvent(eventType) {
			s.closed = true
			s.terminal = true
		}
		return true
	}
}

func (s *upstreamResponsesWSStream) Current() *httpclient.StreamEvent {
	return s.current
}

func (s *upstreamResponsesWSStream) Err() error {
	return s.err
}

func (s *upstreamResponsesWSStream) Close() error {
	if s.closed {
		return nil
	}
	s.closed = true
	if !s.terminal {
		s.executor.Close()
	}
	return nil
}

func responsesWebSocketURL(rawURL string) (string, error) {
	parsed, err := url.Parse(rawURL)
	if err != nil {
		return "", err
	}
	switch parsed.Scheme {
	case "https":
		parsed.Scheme = "wss"
	case "http":
		parsed.Scheme = "ws"
	case "ws", "wss":
	default:
		return "", fmt.Errorf("unsupported websocket upstream scheme: %s", parsed.Scheme)
	}
	return parsed.String(), nil
}

func addResponsesWSType(body []byte) []byte {
	if len(body) == 0 {
		return body
	}
	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return body
	}
	if _, ok := payload["type"]; ok {
		return body
	}
	payload["type"] = json.RawMessage(`"response.create"`)
	out, err := json.Marshal(payload)
	if err != nil {
		return body
	}
	return out
}

func responsesWSEventType(payload []byte) string {
	var envelope struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(payload, &envelope); err != nil {
		return "message"
	}
	return envelope.Type
}

func mapResponsesWSWrappedErrorEvent(ctx context.Context, payload []byte, upstreamURL string) *httpclient.Error {
	var event responsesWSWrappedErrorEvent
	if err := json.Unmarshal(payload, &event); err != nil || event.Type != responsesWSWrappedErrorType {
		return nil
	}

	status := 0
	switch {
	case event.Status != nil:
		status = *event.Status
	case event.StatusCode != nil:
		status = *event.StatusCode
	}
	if status < http.StatusBadRequest {
		return nil
	}

	log.Warn(ctx, "responses websocket upstream error event",
		log.Int("status", status),
		log.String("error_type", responsesWSWrappedErrorTypeValue(event.Error)),
		log.String("error_code", responsesWSWrappedErrorCode(event.Error)),
		log.String("message", responsesWSWrappedErrorMessage(event.Error)),
		log.Int("payload_bytes", len(payload)),
	)

	return &httpclient.Error{
		Method:     http.MethodGet,
		URL:        upstreamURL,
		StatusCode: status,
		Status:     strconv.Itoa(status) + " " + http.StatusText(status),
		Body:       payload,
		Headers:    responsesWSWrappedHeaders(event.Headers),
	}
}

func responsesWSWrappedHeaders(headers map[string]json.RawMessage) http.Header {
	out := make(http.Header, len(headers))
	for name, raw := range headers {
		value := responsesWSHeaderValue(raw)
		if value == "" {
			continue
		}
		out.Set(name, value)
	}
	return out
}

func responsesWSHeaderValue(raw json.RawMessage) string {
	var value string
	if err := json.Unmarshal(raw, &value); err == nil {
		return value
	}
	var number json.Number
	if err := json.Unmarshal(raw, &number); err == nil {
		return number.String()
	}
	var boolean bool
	if err := json.Unmarshal(raw, &boolean); err == nil {
		return strconv.FormatBool(boolean)
	}
	return ""
}

func responsesWSWrappedErrorTypeValue(err *responsesWSWrappedError) string {
	if err == nil {
		return ""
	}
	return err.Type
}

func responsesWSWrappedErrorCode(err *responsesWSWrappedError) string {
	if err == nil {
		return ""
	}
	return err.Code
}

func responsesWSWrappedErrorMessage(err *responsesWSWrappedError) string {
	if err == nil {
		return ""
	}
	return err.Message
}

func logResponsesWSCreateFrame(ctx context.Context, payload []byte) {
	var summary responsesWSCreateLogSummary
	if err := json.Unmarshal(payload, &summary); err != nil || summary.Type != "response.create" {
		return
	}

	inputKind, inputItems := summarizeResponsesWSInput(summary.Input)
	log.Info(ctx, "responses websocket create frame",
		log.Bool("generate_set", summary.Generate != nil),
		log.Bool("generate", summary.Generate != nil && *summary.Generate),
		log.Bool("has_previous_response_id", summary.PreviousResponseID != nil && *summary.PreviousResponseID != ""),
		log.String("previous_response_id_prefix", responsesWSIDPrefix(summary.PreviousResponseID)),
		log.String("input_kind", inputKind),
		log.Int("input_items", inputItems),
		log.Int("input_bytes", len(summary.Input)),
		log.Int("payload_bytes", len(payload)),
	)
}

func logResponsesWSTerminalEvent(ctx context.Context, eventType string, payload []byte) {
	var summary struct {
		Response *struct {
			ID string `json:"id"`
		} `json:"response"`
	}
	_ = json.Unmarshal(payload, &summary)

	responseID := ""
	if summary.Response != nil {
		responseID = summary.Response.ID
	}
	log.Info(ctx, "responses websocket terminal event",
		log.String("event_type", eventType),
		log.String("response_id_prefix", responsesWSShortIDPrefix(responseID)),
		log.Int("payload_bytes", len(payload)),
	)
}

func summarizeResponsesWSInput(input json.RawMessage) (string, int) {
	if len(input) == 0 || string(input) == "null" {
		return "none", 0
	}

	var items []json.RawMessage
	if err := json.Unmarshal(input, &items); err == nil {
		return "array", len(items)
	}

	var text string
	if err := json.Unmarshal(input, &text); err == nil {
		return "string", 1
	}

	return "other", 1
}

func responsesWSIDPrefix(id *string) string {
	if id == nil {
		return ""
	}
	return responsesWSShortIDPrefix(*id)
}

func responsesWSShortIDPrefix(id string) string {
	if len(id) <= 12 {
		return id
	}
	return id[:12]
}

func isResponsesWSTerminalEvent(eventType string) bool {
	return eventType == responsesWSCompleteEvent ||
		eventType == responsesWSFailedEvent ||
		eventType == responsesWSIncompleteEvent ||
		eventType == responsesWSWrappedErrorType
}

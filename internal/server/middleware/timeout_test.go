package middleware

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestWithTimeoutSkipsWebSocketUpgradeRequests(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(WithTimeout(time.Nanosecond))
	router.GET("/v1/responses", func(c *gin.Context) {
		time.Sleep(time.Millisecond)
		require.NoError(t, c.Request.Context().Err())
		c.Status(http.StatusSwitchingProtocols)
	})

	req := httptest.NewRequest(http.MethodGet, "/v1/responses", nil)
	req.Header.Set("Connection", "Upgrade")
	req.Header.Set("Upgrade", "websocket")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	require.Equal(t, http.StatusSwitchingProtocols, w.Code)
}

func TestWithTimeoutAppliesToNormalRequests(t *testing.T) {
	gin.SetMode(gin.TestMode)

	router := gin.New()
	router.Use(WithTimeout(time.Nanosecond))
	router.GET("/v1/responses", func(c *gin.Context) {
		time.Sleep(time.Millisecond)
		require.ErrorIs(t, c.Request.Context().Err(), context.DeadlineExceeded)
		c.Status(http.StatusNoContent)
	})

	req := httptest.NewRequest(http.MethodGet, "/v1/responses", nil)
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	require.Equal(t, http.StatusNoContent, w.Code)
}

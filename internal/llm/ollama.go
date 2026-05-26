package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// OllamaClient handles all communication with local Ollama server
type OllamaClient struct {
	baseURL    string
	chatModel  string
	embedModel string
	http       *http.Client
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
	Options  Options   `json:"options,omitempty"`
}

type Options struct {
	Temperature float64 `json:"temperature,omitempty"`
	NumCtx      int     `json:"num_ctx,omitempty"`
}

type ChatResponse struct {
	Message Message `json:"message"`
	Done    bool    `json:"done"`
}

type EmbedRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type EmbedResponse struct {
	Embedding []float64 `json:"embedding"`
}

func NewOllamaClient(baseURL, chatModel, embedModel string) *OllamaClient {
	return &OllamaClient{
		baseURL:    baseURL,
		chatModel:  chatModel,
		embedModel: embedModel,
		http: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
}

// Chat sends messages to Ollama and returns the response
func (c *OllamaClient) Chat(ctx context.Context, messages []Message) (string, error) {
	req := ChatRequest{
		Model:    c.chatModel,
		Messages: messages,
		Stream:   false,
		Options: Options{
			Temperature: 0.8,
			NumCtx:      4096,
		},
	}

	body, err := json.Marshal(req)
	if err != nil {
		return "", err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/chat", bytes.NewBuffer(body))
	if err != nil {
		return "", err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("ollama chat request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("ollama returned %d: %s", resp.StatusCode, string(b))
	}

	var chatResp ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return "", err
	}

	return chatResp.Message.Content, nil
}

// Embed generates an embedding vector for the given text
func (c *OllamaClient) Embed(ctx context.Context, text string) ([]float64, error) {
	req := EmbedRequest{
		Model:  c.embedModel,
		Prompt: text,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/embeddings", bytes.NewBuffer(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama embed request failed: %w", err)
	}
	defer resp.Body.Close()

	var embedResp EmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&embedResp); err != nil {
		return nil, err
	}

	return embedResp.Embedding, nil
}

// IsAvailable checks if Ollama is running
func (c *OllamaClient) IsAvailable(ctx context.Context) bool {
	req, _ := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/api/tags", nil)
	resp, err := c.http.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

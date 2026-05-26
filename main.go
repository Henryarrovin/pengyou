package main

import (
	"context"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/Henryarrovin/pengyou/internal/bot"
	"github.com/Henryarrovin/pengyou/internal/llm"
	"github.com/Henryarrovin/pengyou/internal/memory"
	"github.com/Henryarrovin/pengyou/internal/rag"
)

func main() {
	log.Println("🐼 朋友 (Péngyǒu) is waking up...")

	// Load config
	cfg := loadConfig()

	// Init memory store (SQLite + vector embeddings)
	memStore, err := memory.NewStore(cfg.DBPath)
	if err != nil {
		log.Fatalf("Failed to init memory store: %v", err)
	}
	defer memStore.Close()

	// Init Ollama LLM client
	llmClient := llm.NewOllamaClient(cfg.OllamaURL, cfg.ChatModel, cfg.EmbedModel)

	// Init RAG engine
	ragEngine := rag.NewEngine(memStore, llmClient)

	// Init Discord bot
	discordBot, err := bot.New(cfg.DiscordToken, ragEngine, llmClient, memStore)
	if err != nil {
		log.Fatalf("Failed to create bot: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := discordBot.Start(ctx); err != nil {
		log.Fatalf("Failed to start bot: %v", err)
	}

	log.Println("✅ 朋友 is online! 你好！")

	// Wait for interrupt
	sc := make(chan os.Signal, 1)
	signal.Notify(sc, syscall.SIGINT, syscall.SIGTERM, os.Interrupt)
	<-sc

	log.Println("👋 朋友 is shutting down... 再见！")
	discordBot.Stop()
}

type Config struct {
	DiscordToken string
	OllamaURL    string
	ChatModel    string
	EmbedModel   string
	DBPath       string
}

func loadConfig() Config {
	cfg := Config{
		DiscordToken: mustEnv("DISCORD_TOKEN"),
		OllamaURL:    getEnv("OLLAMA_URL", "http://localhost:11434"),
		ChatModel:    getEnv("CHAT_MODEL", "qwen2.5:7b"), // Great for Chinese!
		EmbedModel:   getEnv("EMBED_MODEL", "nomic-embed-text"),
		DBPath:       getEnv("DB_PATH", "./data/pengyou.db"),
	}
	return cfg
}

func mustEnv(key string) string {
	v := os.Getenv(key)
	if v == "" {
		log.Fatalf("Required env var %s is not set", key)
	}
	return v
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

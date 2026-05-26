package bot

import (
	"context"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/Henryarrovin/pengyou/internal/llm"
	"github.com/Henryarrovin/pengyou/internal/memory"
	"github.com/Henryarrovin/pengyou/internal/rag"
	"github.com/bwmarrin/discordgo"
)

type Bot struct {
	session   *discordgo.Session
	ragEngine *rag.Engine
	llm       *llm.OllamaClient
	store     *memory.Store
	// Track which channels the bot is actively responding to
	// (DMs always active; guilds only when mentioned or in designated channels)
}

// New creates and configures the Discord bot
func New(token string, ragEngine *rag.Engine, llmClient *llm.OllamaClient, store *memory.Store) (*Bot, error) {
	dg, err := discordgo.New("Bot " + token)
	if err != nil {
		return nil, fmt.Errorf("create discord session: %w", err)
	}

	dg.Identify.Intents = discordgo.IntentsGuildMessages |
		discordgo.IntentsDirectMessages |
		discordgo.IntentsMessageContent |
		discordgo.IntentsGuilds

	b := &Bot{
		session:   dg,
		ragEngine: ragEngine,
		llm:       llmClient,
		store:     store,
	}

	dg.AddHandler(b.onReady)
	dg.AddHandler(b.onMessage)
	dg.AddHandler(b.onInteraction)

	return b, nil
}

func (b *Bot) Start(ctx context.Context) error {
	if err := b.session.Open(); err != nil {
		return fmt.Errorf("open discord connection: %w", err)
	}

	// Register slash commands
	b.registerCommands()

	return nil
}

func (b *Bot) Stop() {
	b.session.Close()
}

func (b *Bot) onReady(s *discordgo.Session, r *discordgo.Ready) {
	log.Printf("✅ Logged in as %s#%s", r.User.Username, r.User.Discriminator)
	s.UpdateGameStatus(0, "学中文 | /learn to start!")
}

// onMessage handles all incoming Discord messages
func (b *Bot) onMessage(s *discordgo.Session, m *discordgo.MessageCreate) {
	// Ignore bot messages
	if m.Author.ID == s.State.User.ID || m.Author.Bot {
		return
	}

	// Respond in DMs always, in guilds only when mentioned
	isDM := m.GuildID == ""
	isMentioned := false
	for _, mention := range m.Mentions {
		if mention.ID == s.State.User.ID {
			isMentioned = true
			break
		}
	}

	if !isDM && !isMentioned {
		return
	}

	// Clean the message content (remove bot mention)
	content := cleanMessage(m.Content, s.State.User.ID)
	if strings.TrimSpace(content) == "" {
		return
	}

	// Show typing indicator
	s.ChannelTyping(m.ChannelID)

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	userID := m.Author.ID
	response, err := b.generateResponse(ctx, userID, content)
	if err != nil {
		log.Printf("Error generating response for %s: %v", userID, err)
		s.ChannelMessageSend(m.ChannelID, "哎呀 (āi yā)! Something went wrong on my end 😅 Try again!")
		return
	}

	// Send response (handle Discord's 2000 char limit)
	sendChunked(s, m.ChannelID, response)

	// Background: extract facts and save messages
	go func() {
		bgCtx, bgCancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer bgCancel()

		// Save messages to memory
		userEmb, _ := b.llm.Embed(bgCtx, content)
		b.store.SaveMessage(userID, "user", content, userEmb)

		botEmb, _ := b.llm.Embed(bgCtx, response)
		b.store.SaveMessage(userID, "assistant", response, botEmb)

		// Extract and save facts about the user
		b.ragEngine.ExtractAndSaveFacts(bgCtx, userID, content, response)

		// Occasionally update level estimate
		b.ragEngine.UpdateUserLevel(bgCtx, userID, content+" "+response)

		// Update session count
		profile, err := b.store.GetOrCreateProfile(userID)
		if err == nil {
			profile.TotalSessions++
			b.store.UpdateProfile(profile)
		}
	}()
}

// generateResponse is the core RAG pipeline
func (b *Bot) generateResponse(ctx context.Context, userID, message string) (string, error) {
	// 1. Retrieve relevant context
	rc, err := b.ragEngine.Retrieve(ctx, userID, message)
	if err != nil {
		return "", fmt.Errorf("retrieve context: %w", err)
	}

	// 2. Build personalized system prompt
	systemPrompt := b.ragEngine.BuildSystemPrompt(rc)

	// 3. Build full message array with history
	messages := b.ragEngine.BuildMessages(rc, systemPrompt, message)

	// 4. Generate response
	response, err := b.llm.Chat(ctx, messages)
	if err != nil {
		return "", fmt.Errorf("llm chat: %w", err)
	}

	return response, nil
}

// registerCommands registers Discord slash commands
func (b *Bot) registerCommands() {
	commands := []*discordgo.ApplicationCommand{
		{
			Name:        "learn",
			Description: "Start or continue your Chinese learning journey with 朋友!",
		},
		{
			Name:        "profile",
			Description: "See your Chinese learning profile and stats",
		},
		{
			Name:        "practice",
			Description: "Do a quick Chinese practice session",
			Options: []*discordgo.ApplicationCommandOption{
				{
					Type:        discordgo.ApplicationCommandOptionString,
					Name:        "topic",
					Description: "What to practice (e.g. greetings, numbers, food)",
					Required:    false,
				},
			},
		},
		{
			Name:        "word",
			Description: "Learn a specific Chinese word or phrase",
			Options: []*discordgo.ApplicationCommandOption{
				{
					Type:        discordgo.ApplicationCommandOptionString,
					Name:        "word",
					Description: "The word or phrase to learn",
					Required:    true,
				},
			},
		},
		{
			Name:        "quiz",
			Description: "Take a quick quiz on words you've learned",
		},
		{
			Name:        "reset",
			Description: "Reset your learning profile (fresh start)",
		},
	}

	for _, cmd := range commands {
		_, err := b.session.ApplicationCommandCreate(b.session.State.User.ID, "", cmd)
		if err != nil {
			log.Printf("Cannot create command %s: %v", cmd.Name, err)
		}
	}
	log.Println("✅ Slash commands registered")
}

// onInteraction handles slash commands
func (b *Bot) onInteraction(s *discordgo.Session, i *discordgo.InteractionCreate) {
	if i.Type != discordgo.InteractionApplicationCommand {
		return
	}

	// Acknowledge immediately
	s.InteractionRespond(i.Interaction, &discordgo.InteractionResponse{
		Type: discordgo.InteractionResponseDeferredChannelMessageWithSource,
	})

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	userID := i.Member.User.ID
	if i.Member == nil && i.User != nil {
		userID = i.User.ID
	}

	data := i.ApplicationCommandData()
	var response string
	var err error

	switch data.Name {
	case "learn":
		response, err = b.generateResponse(ctx, userID,
			"Hey! I'm ready to learn Chinese. Give me a fun intro and let's get started!")

	case "profile":
		response = b.buildProfileResponse(ctx, userID)

	case "practice":
		topic := "general conversation"
		if len(data.Options) > 0 {
			topic = data.Options[0].StringValue()
		}
		response, err = b.generateResponse(ctx, userID,
			fmt.Sprintf("Let's do a practice session focused on: %s. Make it fun and interactive!", topic))

	case "word":
		word := data.Options[0].StringValue()
		response, err = b.generateResponse(ctx, userID,
			fmt.Sprintf("Teach me the Chinese word/phrase for: %s. Use your baby-learning style!", word))

	case "quiz":
		response, err = b.generateResponse(ctx, userID,
			"Give me a fun quick quiz on Chinese words I've been learning. Make it feel like a game!")

	case "reset":
		response = b.handleReset(userID)
	}

	if err != nil {
		response = "哎呀! Something went wrong 😅 Try again!"
	}

	// Send followup (split if needed)
	chunks := splitMessage(response, 1900)
	for idx, chunk := range chunks {
		if idx == 0 {
			s.InteractionResponseEdit(i.Interaction, &discordgo.WebhookEdit{
				Content: &chunk,
			})
		} else {
			s.FollowupMessageCreate(i.Interaction, false, &discordgo.WebhookParams{
				Content: chunk,
			})
		}
	}
}

func (b *Bot) buildProfileResponse(ctx context.Context, userID string) string {
	profile, err := b.store.GetOrCreateProfile(userID)
	if err != nil {
		return "Couldn't load your profile 😅"
	}

	var sb strings.Builder
	sb.WriteString("## 📊 Your 朋友 Profile\n\n")
	sb.WriteString(fmt.Sprintf("**Level:** %s\n", strings.Title(profile.Level)))
	sb.WriteString(fmt.Sprintf("**Sessions:** %d\n", profile.TotalSessions))

	if len(profile.LearnedWords) > 0 {
		sb.WriteString(fmt.Sprintf("**Words learned:** %d\n", len(profile.LearnedWords)))
		last := profile.LearnedWords
		if len(last) > 5 {
			last = last[len(last)-5:]
		}
		sb.WriteString(fmt.Sprintf("**Recent vocab:** %s\n", strings.Join(last, ", ")))
	}

	if len(profile.Interests) > 0 {
		sb.WriteString(fmt.Sprintf("**Your interests:** %s\n", strings.Join(profile.Interests, ", ")))
	}

	sb.WriteString("\n*Keep going! 加油 (jiā yóu)! 💪*")
	return sb.String()
}

func (b *Bot) handleReset(userID string) string {
	// Soft reset — clear learned words and level but keep facts
	profile, err := b.store.GetOrCreateProfile(userID)
	if err != nil {
		return "Couldn't reset 😅"
	}
	profile.Level = "beginner"
	profile.LearnedWords = nil
	profile.WeakTopics = nil
	profile.TotalSessions = 0
	b.store.UpdateProfile(profile)
	return "✅ Fresh start! 重新开始 (chóngxīn kāishǐ) — 'start over'! Your first new word 😄"
}

func cleanMessage(content, botID string) string {
	content = strings.ReplaceAll(content, "<@"+botID+">", "")
	content = strings.ReplaceAll(content, "<@!"+botID+">", "")
	return strings.TrimSpace(content)
}

func sendChunked(s *discordgo.Session, channelID, content string) {
	chunks := splitMessage(content, 1900)
	for _, chunk := range chunks {
		s.ChannelMessageSend(channelID, chunk)
	}
}

func splitMessage(content string, maxLen int) []string {
	if len(content) <= maxLen {
		return []string{content}
	}

	var chunks []string
	for len(content) > maxLen {
		// Try to split at newline
		idx := strings.LastIndex(content[:maxLen], "\n")
		if idx < maxLen/2 {
			idx = maxLen
		}
		chunks = append(chunks, content[:idx])
		content = content[idx:]
	}
	if content != "" {
		chunks = append(chunks, content)
	}
	return chunks
}

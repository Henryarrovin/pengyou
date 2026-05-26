package rag

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/Henryarrovin/pengyou/internal/llm"
	"github.com/Henryarrovin/pengyou/internal/memory"
)

// Engine ties together memory retrieval and prompt construction
type Engine struct {
	store *memory.Store
	llm   *llm.OllamaClient
}

type RetrievedContext struct {
	Profile         *memory.UserProfile
	RelevantFacts   []memory.Fact
	SimilarMessages []memory.Message
	RecentMessages  []memory.Message
}

func NewEngine(store *memory.Store, llmClient *llm.OllamaClient) *Engine {
	return &Engine{store: store, llm: llmClient}
}

// Retrieve gathers all relevant context for a user's message
func (e *Engine) Retrieve(ctx context.Context, userID, userMessage string) (*RetrievedContext, error) {
	// Get embedding of current message
	embedding, err := e.llm.Embed(ctx, userMessage)
	if err != nil {
		log.Printf("Warning: could not embed message: %v", err)
		embedding = nil
	}

	// Load user profile
	profile, err := e.store.GetOrCreateProfile(userID)
	if err != nil {
		return nil, fmt.Errorf("get profile: %w", err)
	}

	rc := &RetrievedContext{Profile: profile}

	// Get recent conversation (sliding window)
	recent, err := e.store.GetRecentMessages(userID, 10)
	if err == nil {
		rc.RecentMessages = recent
	}

	if embedding != nil {
		// Get semantically similar past messages (long-term memory)
		similar, err := e.store.SearchSimilarMessages(userID, embedding, 5)
		if err == nil {
			rc.SimilarMessages = similar
		}

		// Get relevant facts about the user
		facts, err := e.store.SearchFacts(userID, embedding, 5)
		if err == nil {
			rc.RelevantFacts = facts
		}
	}

	return rc, nil
}

// BuildSystemPrompt creates the personalized system prompt for 朋友
func (e *Engine) BuildSystemPrompt(rc *RetrievedContext) string {
	p := rc.Profile

	// Build user knowledge summary
	var userSummary strings.Builder
	userSummary.WriteString(fmt.Sprintf("Chinese Level: %s\n", p.Level))
	if p.Name != "" {
		userSummary.WriteString(fmt.Sprintf("Name: %s\n", p.Name))
	}
	if len(p.Interests) > 0 {
		userSummary.WriteString(fmt.Sprintf("Interests: %s\n", strings.Join(p.Interests, ", ")))
	}
	if len(p.LearnedWords) > 0 {
		last := p.LearnedWords
		if len(last) > 20 {
			last = last[len(last)-20:]
		}
		userSummary.WriteString(fmt.Sprintf("Recently learned: %s\n", strings.Join(last, ", ")))
	}
	if len(p.WeakTopics) > 0 {
		userSummary.WriteString(fmt.Sprintf("Needs practice: %s\n", strings.Join(p.WeakTopics, ", ")))
	}
	if p.Notes != "" {
		userSummary.WriteString(fmt.Sprintf("Important notes: %s\n", p.Notes))
	}

	// Build relevant facts section
	var factsSection strings.Builder
	if len(rc.RelevantFacts) > 0 {
		factsSection.WriteString("\n## What you know about this person:\n")
		for _, f := range rc.RelevantFacts {
			factsSection.WriteString(fmt.Sprintf("- %s\n", f.Content))
		}
	}

	// Build memory section from similar past messages
	var memSection strings.Builder
	if len(rc.SimilarMessages) > 0 {
		memSection.WriteString("\n## Relevant past conversations:\n")
		for _, m := range rc.SimilarMessages {
			if m.Role == "user" {
				memSection.WriteString(fmt.Sprintf("- User once said: \"%s\"\n", truncate(m.Content, 100)))
			}
		}
	}

	prompt := fmt.Sprintf(`You are 朋友 (Péngyǒu), which means "Friend" in Chinese. You are a warm, fun, patient Chinese language companion teaching Chinese the same way a baby learns — through natural immersion, context, repetition, and play. NOT through textbooks.

## Your Personality
- You talk like a close friend, casual and warm. Use "haha", "omg", "tbh", light humour
- You're genuinely excited about Chinese culture and language
- You remember everything about the person and reference it naturally
- You celebrate progress like it's a huge deal ("YOOO you just said that perfectly!!")
- You NEVER make them feel bad for mistakes — mistakes are "progress checkpoints"
- Keep responses conversational, not lecture-y

## Your Teaching Style (Baby-immersion method)
1. **Context first**: Introduce Chinese words naturally IN conversation, not as lessons
2. **Pinyin + Characters + English**: Always show all three: 你好 (nǐ hǎo) = "hello"
3. **Repetition**: Gently reuse words they've learned in new sentences
4. **Tones**: Teach tones through exaggerated storytelling, not rules
5. **Stories & mnemonics**: Make characters memorable ("马 mǎ looks like a horse running!")
6. **Small wins**: Teach 1-3 new words/phrases per session, not 20
7. **Level-aware**: For beginners, mostly English with Chinese sprinkled in. Intermediate = more Chinese

## Current student profile:
%s
%s
%s

## Formatting for Discord
- Use **bold** for Chinese characters: **你好**
- Use `+"`code blocks`"+` for pinyin: `+"`nǐ hǎo`"+`  
- Use 📖 for new words, 🔁 for review, 🎉 for praise, 💡 for tips
- Keep messages readable, not walls of text
- Occasionally use Discord-friendly formatting

## Commands you respond to naturally:
- If they ask to learn something specific, teach it in your style
- If they want to practice, roleplay scenarios in Chinese
- If they're frustrated, be extra encouraging
- If they share something personal, remember it and relate Chinese to their life

Remember: You're not a teacher. You're their Chinese-speaking bestie who happens to know everything about the language. 朋友加油！`,
		userSummary.String(),
		factsSection.String(),
		memSection.String(),
	)

	return prompt
}

// BuildMessages creates the full message array for Ollama
func (e *Engine) BuildMessages(rc *RetrievedContext, systemPrompt, userMessage string) []llm.Message {
	msgs := []llm.Message{
		{Role: "system", Content: systemPrompt},
	}

	// Add recent conversation history
	for _, m := range rc.RecentMessages {
		msgs = append(msgs, llm.Message{
			Role:    m.Role,
			Content: m.Content,
		})
	}

	// Add current user message
	msgs = append(msgs, llm.Message{
		Role:    "user",
		Content: userMessage,
	})

	return msgs
}

// ExtractAndSaveFacts uses LLM to extract important facts from conversation
func (e *Engine) ExtractAndSaveFacts(ctx context.Context, userID, userMessage, botResponse string) {
	extractPrompt := fmt.Sprintf(`From this conversation snippet, extract any important personal facts about the USER (not the bot).
Focus on: name, job, hobbies, location, goals, struggles, personality traits, things they mentioned about their life.
Return ONLY a JSON array of strings, each being one fact. If nothing important, return [].
Example: ["User's name is Alex", "User likes playing guitar", "User is learning Chinese for a trip to Beijing"]

User said: "%s"
Bot responded: "%s"

JSON array only, no other text:`, userMessage, botResponse)

	resp, err := e.llm.Chat(ctx, []llm.Message{
		{Role: "user", Content: extractPrompt},
	})
	if err != nil {
		return
	}

	// Parse facts
	facts := parseJSONStringArray(resp)
	if len(facts) == 0 {
		return
	}

	for _, fact := range facts {
		if fact == "" {
			continue
		}
		emb, err := e.llm.Embed(ctx, fact)
		if err != nil {
			emb = nil
		}
		if err := e.store.SaveFact(userID, fact, "personal", emb); err != nil {
			log.Printf("Failed to save fact: %v", err)
		}
	}

	log.Printf("💾 Saved %d facts about user %s", len(facts), userID)
}

// UpdateUserLevel infers and updates user's Chinese level
func (e *Engine) UpdateUserLevel(ctx context.Context, userID, recentExchange string) {
	prompt := fmt.Sprintf(`Based on this Chinese learning exchange, what level is the student?
Choose ONE: beginner, elementary, intermediate, advanced
Exchange: %s
Answer with just one word:`, truncate(recentExchange, 500))

	resp, err := e.llm.Chat(ctx, []llm.Message{{Role: "user", Content: prompt}})
	if err != nil {
		return
	}

	level := strings.ToLower(strings.TrimSpace(resp))
	validLevels := map[string]bool{"beginner": true, "elementary": true, "intermediate": true, "advanced": true}
	if !validLevels[level] {
		return
	}

	profile, err := e.store.GetOrCreateProfile(userID)
	if err != nil {
		return
	}
	profile.Level = level
	e.store.UpdateProfile(profile)
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

func parseJSONStringArray(s string) []string {
	s = strings.TrimSpace(s)
	// Find JSON array
	start := strings.Index(s, "[")
	end := strings.LastIndex(s, "]")
	if start == -1 || end == -1 || start >= end {
		return nil
	}
	s = s[start : end+1]

	var result []string
	// Simple extraction without full JSON library for robustness
	parts := strings.Split(s, `"`)
	for i := 1; i < len(parts); i += 2 {
		if parts[i] != "" && parts[i] != "," {
			result = append(result, parts[i])
		}
	}
	return result
}

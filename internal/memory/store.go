package memory

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"sort"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

// Store handles all persistent memory: user profiles, conversation history, embeddings
type Store struct {
	db *sql.DB
}

type Message struct {
	ID        int64
	UserID    string
	Role      string // "user" or "assistant"
	Content   string
	Embedding []float64
	CreatedAt time.Time
}

type UserProfile struct {
	UserID         string
	Name           string
	Level          string // beginner, elementary, intermediate
	NativeLanguage string
	Interests      []string
	LearnedWords   []string
	WeakTopics     []string
	TotalSessions  int
	LastSeen       time.Time
	Notes          string // free-form notes about the user
}

type Fact struct {
	ID        int64
	UserID    string
	Content   string
	Embedding []float64
	Category  string // "personal", "progress", "preference"
	CreatedAt time.Time
}

func NewStore(dbPath string) (*Store, error) {
	db, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL")
	if err != nil {
		return nil, fmt.Errorf("open db: %w", err)
	}

	s := &Store{db: db}
	if err := s.migrate(); err != nil {
		return nil, fmt.Errorf("migrate: %w", err)
	}

	log.Println("📦 Memory store initialized")
	return s, nil
}

func (s *Store) migrate() error {
	queries := []string{
		`CREATE TABLE IF NOT EXISTS messages (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id    TEXT NOT NULL,
			role       TEXT NOT NULL,
			content    TEXT NOT NULL,
			embedding  TEXT,
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE TABLE IF NOT EXISTS user_profiles (
			user_id         TEXT PRIMARY KEY,
			name            TEXT,
			level           TEXT DEFAULT 'beginner',
			native_language TEXT DEFAULT 'english',
			interests       TEXT DEFAULT '[]',
			learned_words   TEXT DEFAULT '[]',
			weak_topics     TEXT DEFAULT '[]',
			total_sessions  INTEGER DEFAULT 0,
			last_seen       DATETIME,
			notes           TEXT DEFAULT ''
		)`,
		`CREATE TABLE IF NOT EXISTS facts (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			user_id    TEXT NOT NULL,
			content    TEXT NOT NULL,
			embedding  TEXT,
			category   TEXT DEFAULT 'personal',
			created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		)`,
		`CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id)`,
		`CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id)`,
	}

	for _, q := range queries {
		if _, err := s.db.Exec(q); err != nil {
			return fmt.Errorf("exec %q: %w", q[:30], err)
		}
	}
	return nil
}

// SaveMessage persists a message with its embedding
func (s *Store) SaveMessage(userID, role, content string, embedding []float64) error {
	embJSON, _ := json.Marshal(embedding)
	_, err := s.db.Exec(
		`INSERT INTO messages (user_id, role, content, embedding) VALUES (?, ?, ?, ?)`,
		userID, role, content, string(embJSON),
	)
	return err
}

// GetRecentMessages returns the last N messages for a user (for conversation context)
func (s *Store) GetRecentMessages(userID string, limit int) ([]Message, error) {
	rows, err := s.db.Query(
		`SELECT id, user_id, role, content, created_at 
		 FROM messages WHERE user_id = ? 
		 ORDER BY created_at DESC LIMIT ?`,
		userID, limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var msgs []Message
	for rows.Next() {
		var m Message
		if err := rows.Scan(&m.ID, &m.UserID, &m.Role, &m.Content, &m.CreatedAt); err != nil {
			continue
		}
		msgs = append(msgs, m)
	}

	// Reverse so oldest first
	for i, j := 0, len(msgs)-1; i < j; i, j = i+1, j-1 {
		msgs[i], msgs[j] = msgs[j], msgs[i]
	}
	return msgs, nil
}

// SearchSimilarMessages finds semantically similar past messages using cosine similarity
func (s *Store) SearchSimilarMessages(userID string, queryEmbedding []float64, topK int) ([]Message, error) {
	rows, err := s.db.Query(
		`SELECT id, user_id, role, content, embedding, created_at 
		 FROM messages WHERE user_id = ? AND embedding IS NOT NULL`,
		userID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	type scored struct {
		msg   Message
		score float64
	}
	var candidates []scored

	for rows.Next() {
		var m Message
		var embStr string
		if err := rows.Scan(&m.ID, &m.UserID, &m.Role, &m.Content, &embStr, &m.CreatedAt); err != nil {
			continue
		}
		var emb []float64
		if err := json.Unmarshal([]byte(embStr), &emb); err != nil {
			continue
		}
		m.Embedding = emb
		score := cosineSimilarity(queryEmbedding, emb)
		candidates = append(candidates, scored{m, score})
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	var result []Message
	for i, c := range candidates {
		if i >= topK {
			break
		}
		if c.score > 0.6 { // threshold
			result = append(result, c.msg)
		}
	}
	return result, nil
}

// SaveFact stores an extracted fact about the user
func (s *Store) SaveFact(userID, content, category string, embedding []float64) error {
	embJSON, _ := json.Marshal(embedding)
	_, err := s.db.Exec(
		`INSERT INTO facts (user_id, content, category, embedding) VALUES (?, ?, ?, ?)`,
		userID, content, category, string(embJSON),
	)
	return err
}

// SearchFacts finds relevant facts about the user
func (s *Store) SearchFacts(userID string, queryEmbedding []float64, topK int) ([]Fact, error) {
	rows, err := s.db.Query(
		`SELECT id, user_id, content, embedding, category, created_at 
		 FROM facts WHERE user_id = ?`,
		userID,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	type scored struct {
		fact  Fact
		score float64
	}
	var candidates []scored

	for rows.Next() {
		var f Fact
		var embStr string
		if err := rows.Scan(&f.ID, &f.UserID, &f.Content, &embStr, &f.Category, &f.CreatedAt); err != nil {
			continue
		}
		var emb []float64
		json.Unmarshal([]byte(embStr), &emb)
		f.Embedding = emb
		score := cosineSimilarity(queryEmbedding, emb)
		candidates = append(candidates, scored{f, score})
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score > candidates[j].score
	})

	var result []Fact
	for i, c := range candidates {
		if i >= topK {
			break
		}
		result = append(result, c.fact)
	}
	return result, nil
}

// GetOrCreateProfile returns the user profile, creating one if needed
func (s *Store) GetOrCreateProfile(userID string) (*UserProfile, error) {
	var p UserProfile
	var interestsJSON, learnedJSON, weakJSON string

	err := s.db.QueryRow(
		`SELECT user_id, name, level, native_language, interests, learned_words, weak_topics, total_sessions, last_seen, notes
		 FROM user_profiles WHERE user_id = ?`, userID,
	).Scan(&p.UserID, &p.Name, &p.Level, &p.NativeLanguage, &interestsJSON, &learnedJSON, &weakJSON, &p.TotalSessions, &p.LastSeen, &p.Notes)

	if err == sql.ErrNoRows {
		// Create new profile
		p = UserProfile{
			UserID:         userID,
			Level:          "beginner",
			NativeLanguage: "english",
			LastSeen:       time.Now(),
		}
		_, err = s.db.Exec(
			`INSERT INTO user_profiles (user_id, last_seen) VALUES (?, ?)`,
			userID, time.Now(),
		)
		if err != nil {
			return nil, err
		}
		return &p, nil
	}
	if err != nil {
		return nil, err
	}

	json.Unmarshal([]byte(interestsJSON), &p.Interests)
	json.Unmarshal([]byte(learnedJSON), &p.LearnedWords)
	json.Unmarshal([]byte(weakJSON), &p.WeakTopics)

	return &p, nil
}

// UpdateProfile saves updated user profile fields
func (s *Store) UpdateProfile(p *UserProfile) error {
	interestsJSON, _ := json.Marshal(p.Interests)
	learnedJSON, _ := json.Marshal(p.LearnedWords)
	weakJSON, _ := json.Marshal(p.WeakTopics)

	_, err := s.db.Exec(
		`UPDATE user_profiles SET 
			name=?, level=?, native_language=?, interests=?, learned_words=?, 
			weak_topics=?, total_sessions=?, last_seen=?, notes=?
		 WHERE user_id=?`,
		p.Name, p.Level, p.NativeLanguage, string(interestsJSON),
		string(learnedJSON), string(weakJSON), p.TotalSessions,
		time.Now(), p.Notes, p.UserID,
	)
	return err
}

func (s *Store) Close() error {
	return s.db.Close()
}

// cosineSimilarity computes cosine similarity between two vectors
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

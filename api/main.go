package handler

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"
)

// TokenCount defines the structure for token counting
type TokenCount struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

const (
	MaxContextTokens = 2000 // Maximum context token count
)

// YouChatResponse defines the structure for a single token received from You.com API.
type YouChatResponse struct {
	YouChatToken string `json:"youChatToken"`
}

// OpenAIStreamResponse defines the structure for OpenAI API streaming response.
type OpenAIStreamResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
}

// Choice defines the structure for a single element in the choices array of OpenAI streaming response.
type Choice struct {
	Delta        Delta  `json:"delta"`
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason"`
}

// Delta defines the structure representing incremental content in streaming response.
type Delta struct {
	Content string `json:"content"`
}

// OpenAIRequest defines the structure for OpenAI API request body.
type OpenAIRequest struct {
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
	Model    string    `json:"model"`
}

// Message defines the structure for OpenAI chat messages.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// OpenAIResponse defines the structure for OpenAI API non-streaming response.
type OpenAIResponse struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []OpenAIChoice `json:"choices"`
}

// OpenAIChoice defines the structure for a single element in the choices array of OpenAI non-streaming response.
type OpenAIChoice struct {
	Message      Message `json:"message"`
	Index        int     `json:"index"`
	FinishReason string  `json:"finish_reason"`
}

// ModelResponse defines the structure for /v1/models response.
type ModelResponse struct {
	Object string        `json:"object"`
	Data   []ModelDetail `json:"data"`
}

// ModelDetail defines the detailed information of a single model in the model list.
type ModelDetail struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// modelMap stores the mapping from OpenAI model names to You.com model names.
var modelMap = map[string]string{
	"deepseek-reasoner":       "deepseek_r1",
	"deepseek-chat":           "deepseek_v3",
	"o3-mini-high":            "openai_o3_mini_high",
	"o3-mini-medium":          "openai_o3_mini_medium",
	"o1":                      "openai_o1",
	"o1-mini":                 "openai_o1_mini",
	"o1-preview":              "openai_o1_preview",
	"gpt-4o":                  "gpt_4o",
	"gpt-4o-mini":             "gpt_4o_mini",
	"gpt-4-turbo":             "gpt_4_turbo",
	"gpt-3.5-turbo":           "gpt_3.5",
	"claude-3-opus":           "claude_3_opus",
	"claude-3-sonnet":         "claude_3_sonnet",
	"claude-3.5-sonnet":       "claude_3_5_sonnet",
	"claude-3.5-haiku":        "claude_3_5_haiku",
	"gemini-1.5-pro":          "gemini_1_5_pro",
	"gemini-1.5-flash":        "gemini_1_5_flash",
	"llama-3.2-90b":           "llama3_2_90b",
	"llama-3.1-405b":          "llama3_1_405b",
	"mistral-large-2":         "mistral_large_2",
	"qwen-2.5-72b":            "qwen2p5_72b",
	"qwen-2.5-coder-32b":      "qwen2p5_coder_32b",
	"command-r-plus":          "command_r_plus",
	"claude-3-7-sonnet":       "claude_3_7_sonnet",
	"claude-3-7-sonnet-think": "claude_3_7_sonnet_thinking",
}

// getReverseModelMap creates and returns the reverse mapping of modelMap (You.com model name -> OpenAI model name).
func getReverseModelMap() map[string]string {
	reverse := make(map[string]string, len(modelMap))
	for k, v := range modelMap {
		reverse[v] = k
	}
	return reverse
}

// mapModelName maps OpenAI model name to You.com model name.
func mapModelName(openAIModel string) string {
	if mappedModel, exists := modelMap[openAIModel]; exists {
		return mappedModel
	}
	return "deepseek_v3" // Default model
}

// reverseMapModelName maps You.com model name back to OpenAI model name.
func reverseMapModelName(youModel string) string {
	reverseMap := getReverseModelMap()
	if mappedModel, exists := reverseMap[youModel]; exists {
		return mappedModel
	}
	return "deepseek-chat" // Default model
}

// originalModel stores the original OpenAI model name.
var originalModel string

// NonceResponse defines the structure for the response of getting nonce.
type NonceResponse struct {
	Uuid string
}

// UploadResponse defines the structure for the response of file upload.
type UploadResponse struct {
	Filename     string `json:"filename"`
	UserFilename string `json:"user_filename"`
}

// Define the maximum query length
const MaxQueryLength = 2000

// Handler is the main handler function for all incoming HTTP requests.
func Handler(w http.ResponseWriter, r *http.Request) {
	// Handle /v1/models request (list available models)
	if r.URL.Path == "/v1/models" || r.URL.Path == "/api/v1/models" {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "*")

		if r.Method == "OPTIONS" {
			w.WriteHeader(http.StatusOK)
			return
		}

		models := make([]ModelDetail, 0, len(modelMap))
		created := time.Now().Unix()
		for modelID := range modelMap {
			models = append(models, ModelDetail{
				ID:      modelID,
				Object:  "model",
				Created: created,
				OwnedBy: "organization-owner",
			})
		}

		response := ModelResponse{
			Object: "list",
			Data:   models,
		}

		json.NewEncoder(w).Encode(response)
		return
	}

	// Handle non /v1/chat/completions request (service status check)
	if r.URL.Path != "/v1/chat/completions" && r.URL.Path != "/none/v1/chat/completions" && r.URL.Path != "/such/chat/completions" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status":  "You2Api Service Running...",
			"message": "MoLoveSze...",
		})
		return
	}

	// Set CORS headers
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "*")

	if r.Method == "OPTIONS" {
		w.WriteHeader(http.StatusOK)
		return
	}

	// Load allowed tokens from environment if configured
	allowedTokens := strings.Split(os.Getenv("ALLOWED_TOKENS"), ",")
	requireAuth := len(allowedTokens) > 0 && !(len(allowedTokens) == 1 && allowedTokens[0] == "")

	// Validate Authorization header if auth is required
	if requireAuth {
		authHeader := r.Header.Get("Authorization")
		if !strings.HasPrefix(authHeader, "Bearer ") {
			http.Error(w, "Missing or invalid authorization header", http.StatusUnauthorized)
			return
		}
		token := strings.TrimPrefix(authHeader, "Bearer ")

		// Check if token is in whitelist
		validToken := false
		for _, allowedToken := range allowedTokens {
			if token == strings.TrimSpace(allowedToken) {
				validToken = true
				break
			}
		}

		if !validToken {
			http.Error(w, "Unauthorized token", http.StatusUnauthorized)
			return
		}
	}

	// Get real DS token from environment
	dsToken := os.Getenv("DS_TOKEN")
	if dsToken == "" {
		http.Error(w, "DS token not configured", http.StatusInternalServerError)
		return
	}

	// Parse OpenAI request body
	var openAIReq OpenAIRequest
	if err := json.NewDecoder(r.Body).Decode(&openAIReq); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	originalModel = openAIReq.Model

	// Convert system messages to user messages
	openAIReq.Messages = convertSystemToUser(openAIReq.Messages)

	// Build You.com chat history
	var chatHistory []map[string]interface{}
	var sources []map[string]interface{}
	var lastAssistantMessage string

	// Handle historical messages (excluding the last one)
	for _, msg := range openAIReq.Messages[:len(openAIReq.Messages)-1] {
		if msg.Role == "user" {
			tokens, err := countTokens([]Message{msg})
			if err != nil {
				http.Error(w, "Failed to count tokens", http.StatusInternalServerError)
				return
			}

			if tokens > MaxContextTokens {
				// Get nonce
				nonceResp, err := getNonce(dsToken)
				if err != nil {
					fmt.Printf("Failed to get nonce: %v\n", err)
					http.Error(w, "Failed to get nonce", http.StatusInternalServerError)
					return
				}

				// Create temporary file
				tempFile := fmt.Sprintf("temp_%s.txt", nonceResp.Uuid)
				if err := os.WriteFile(tempFile, []byte(msg.Content), 0644); err != nil {
					fmt.Printf("Failed to create temp file: %v\n", err)
					http.Error(w, "Failed to create temp file", http.StatusInternalServerError)
					return
				}
				defer os.Remove(tempFile)

				// Upload file
				uploadResp, err := uploadFile(dsToken, tempFile)
				if err != nil {
					fmt.Printf("Failed to upload file: %v\n", err)
					http.Error(w, "Failed to upload file", http.StatusInternalServerError)
					return
				}

				// Add file source information
				sources = append(sources, map[string]interface{}{
					"source_type":   "user_file",
					"filename":      uploadResp.Filename,
					"user_filename": uploadResp.UserFilename,
					"size_bytes":    len(msg.Content),
				})

				// Use file reference in chat history
				chatHistory = append(chatHistory, map[string]interface{}{
					"question": fmt.Sprintf("Please review the attached file: %s", uploadResp.UserFilename),
					"answer":   "",
				})
			} else {
				chatHistory = append(chatHistory, map[string]interface{}{
					"question": msg.Content,
					"answer":   "",
				})
			}
		} else if msg.Role == "assistant" {
			// Only save the last assistant message
			lastAssistantMessage = msg.Content
		}
	}

	// If there is a last assistant message, add it to the chat history
	if lastAssistantMessage != "" {
		chatHistory = append(chatHistory, map[string]interface{}{
			"question": "",
			"answer":   lastAssistantMessage,
		})
	}

	chatHistoryJSON, _ := json.Marshal(chatHistory)

	// Create You.com API request
	youReq, _ := http.NewRequest("GET", "https://you.com/api/streamingSearch", nil)

	// Generate necessary IDs
	chatId := uuid.New().String()
	conversationTurnId := uuid.New().String()
	traceId := fmt.Sprintf("%s|%s|%s", chatId, conversationTurnId, time.Now().Format(time.RFC3339))

	// Handle the last message
	lastMessage := openAIReq.Messages[len(openAIReq.Messages)-1]
	lastMessageTokens, err := countTokens([]Message{lastMessage})
	if err != nil {
		http.Error(w, "Failed to count tokens", http.StatusInternalServerError)
		return
	}

	// Build query parameters
	q := youReq.URL.Query()

	// Set basic parameters
	q.Add("page", "1")
	q.Add("count", "10")
	q.Add("safeSearch", "Off")
	q.Add("mkt", "en-US")
	q.Add("enable_worklow_generation_ux", "true")
	q.Add("domain", "youchat")
	q.Add("use_personalization_extraction", "true")
	q.Add("queryTraceId", chatId)
	q.Add("chatId", chatId)
	q.Add("conversationTurnId", conversationTurnId)
	q.Add("pastChatLength", fmt.Sprintf("%d", len(chatHistory)))
	q.Add("selectedChatMode", "custom")
	q.Add("selectedAiModel", mapModelName(openAIReq.Model))
	q.Add("enable_agent_clarification_questions", "true")
	q.Add("traceId", traceId)
	q.Add("use_nested_youchat_updates", "true")

	// If the last message exceeds the limit, use file upload
	if lastMessageTokens > MaxContextTokens {
		// Get nonce
		nonceResp, err := getNonce(dsToken)
		if err != nil {
			fmt.Printf("Failed to get nonce: %v\n", err)
			http.Error(w, "Failed to get nonce", http.StatusInternalServerError)
			return
		}

		// Create temporary file
		tempFile := fmt.Sprintf("temp_%s.txt", nonceResp.Uuid)
		if err := os.WriteFile(tempFile, []byte(lastMessage.Content), 0644); err != nil {
			fmt.Printf("Failed to create temp file: %v\n", err)
			http.Error(w, "Failed to create temp file", http.StatusInternalServerError)
			return
		}
		defer os.Remove(tempFile)

		// Upload file
		uploadResp, err := uploadFile(dsToken, tempFile)
		if err != nil {
			fmt.Printf("Failed to upload file: %v\n", err)
			http.Error(w, "Failed to upload file", http.StatusInternalServerError)
			return
		}

		// Add file source information
		sources = append(sources, map[string]interface{}{
			"source_type":   "user_file",
			"filename":      uploadResp.Filename,
			"user_filename": uploadResp.UserFilename,
			"size_bytes":    len(lastMessage.Content),
		})

		// Add sources parameter
		sourcesJSON, _ := json.Marshal(sources)
		q.Add("sources", string(sourcesJSON))

		// Use file reference as query
		q.Add("q", fmt.Sprintf("Please review the attached file: %s", uploadResp.UserFilename))
	} else {
		// If there are previously uploaded files, add sources
		if len(sources) > 0 {
			sourcesJSON, _ := json.Marshal(sources)
			q.Add("sources", string(sourcesJSON))
		}
		q.Add("q", lastMessage.Content)
	}

	q.Add("chat", string(chatHistoryJSON))
	youReq.URL.RawQuery = q.Encode()

	fmt.Printf("\n=== Full Request Information ===\n")
	fmt.Printf("Request URL: %s\n", youReq.URL.String())
	fmt.Printf("Request Headers:\n")
	for key, values := range youReq.Header {
		fmt.Printf("%s: %v\n", key, values)
	}

	// Set request headers
	youReq.Header = http.Header{
		"Cache-Control": {"no-cache"},
		"Accept":        {"text/event-stream"},
		"User-Agent":    {"Mozi1la/5.0 (compatible; YouMobile/1.0; iOS 18.3.1) Version/3.11.0 Build/2656"},
		"Host":          {"you.com"},
	}

	// Set cookies
	cookies := getCookies(dsToken)
	var cookieStrings []string
	for name, value := range cookies {
		cookieStrings = append(cookieStrings, fmt.Sprintf("%s=%s", name, value))
	}
	youReq.Header.Add("Cookie", strings.Join(cookieStrings, ";"))
	fmt.Printf("Cookie: %s\n", strings.Join(cookieStrings, ";"))
	fmt.Printf("===================\n\n")

	// Send request and get response
	client := &http.Client{}
	resp, err := client.Do(youReq)
	if err != nil {
		fmt.Printf("Failed to send request: %v\n", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	// Print response status code
	fmt.Printf("Response Status Code: %d\n", resp.StatusCode)

	// If status code is not 200, print response content
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		fmt.Printf("Error Response Content: %s\n", string(body))
		http.Error(w, fmt.Sprintf("API returned status %d", resp.StatusCode), resp.StatusCode)
		return
	}

	// Choose handler function based on OpenAI request's stream parameter
	if !openAIReq.Stream {
		handleNonStreamingResponse(w, youReq) // Handle non-streaming response
		return
	}

	handleStreamingResponse(w, youReq) // Handle streaming response
}

// getCookies generates the required cookies based on the provided DS token.
func getCookies(dsToken string) map[string]string {
	return map[string]string{
		"guest_has_seen_legal_disclaimer": "true",
		"youchat_personalization":         "true",
		"DS":                              dsToken,                // Key DS token
		"you_subscription":                "youpro_standard_year", // Example subscription information
		"youpro_subscription":             "true",
		"ai_model":                        "deepseek_r1", // Example AI model
		"youchat_smart_learn":             "true",
	}
}

// handleNonStreamingResponse handles non-streaming requests.
func handleNonStreamingResponse(w http.ResponseWriter, youReq *http.Request) {
	client := &http.Client{
		Timeout: 60 * time.Second, // Set timeout
	}
	resp, err := client.Do(youReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	var fullResponse strings.Builder
	scanner := bufio.NewScanner(resp.Body)

	// Set scanner buffer size (optional but important for large responses)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	// Scan response line by line, looking for youChatToken events
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: youChatToken") {
			scanner.Scan() // Read the next line (data line)
			data := scanner.Text()
			if !strings.HasPrefix(data, "data: ") {
				continue // If not a data line, skip
			}
			var token YouChatResponse
			if err := json.Unmarshal([]byte(strings.TrimPrefix(data, "data: ")), &token); err != nil {
				continue // If parsing fails, skip
			}
			fullResponse.WriteString(token.YouChatToken) // Add token to full response
		}
	}

	if scanner.Err() != nil {
		http.Error(w, "Error reading response", http.StatusInternalServerError)
		return
	}

	// Build OpenAI format non-streaming response
	openAIResp := OpenAIResponse{
		ID:      "chatcmpl-" + fmt.Sprintf("%d", time.Now().Unix()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   reverseMapModelName(mapModelName(originalModel)), // Map back to OpenAI model name
		Choices: []OpenAIChoice{
			{
				Message: Message{
					Role:    "assistant",
					Content: fullResponse.String(), // Full response content
				},
				Index:        0,
				FinishReason: "stop", // Stop reason
			},
		},
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(openAIResp); err != nil {
		http.Error(w, "Error encoding response", http.StatusInternalServerError)
		return
	}
}

// handleStreamingResponse handles streaming requests.
func handleStreamingResponse(w http.ResponseWriter, youReq *http.Request) {
	client := &http.Client{} // Streaming requests do not need a timeout as they will continuously receive data
	resp, err := client.Do(youReq)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	// Set headers for streaming response
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	scanner := bufio.NewScanner(resp.Body)
	// Scan response line by line, looking for youChatToken events
	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "event: youChatToken") {
			scanner.Scan()         // Read the next line (data line)
			data := scanner.Text() // Get data line

			var token YouChatResponse
			json.Unmarshal([]byte(strings.TrimPrefix(data, "data: ")), &token) // Parse JSON

			// Build OpenAI format streaming response chunk
			openAIResp := OpenAIStreamResponse{
				ID:      "chatcmpl-" + fmt.Sprintf("%d", time.Now().Unix()),
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   reverseMapModelName(mapModelName(originalModel)), // Map back to OpenAI model name
				Choices: []Choice{
					{
						Delta: Delta{
							Content: token.YouChatToken, // Incremental content
						},
						Index:        0,
						FinishReason: "", // Usually empty in streaming response
					},
				},
			}

			respBytes, _ := json.Marshal(openAIResp)          // Serialize response chunk to JSON
			fmt.Fprintf(w, "data: %s\n\n", string(respBytes)) // Write response data
			w.(http.Flusher).Flush()                          // Flush output immediately
		}
	}

}

// getNonce retrieves the nonce required for file upload.
func getNonce(dsToken string) (*NonceResponse, error) {
	req, _ := http.NewRequest("GET", "https://you.com/api/get_nonce", nil)
	req.Header.Set("Cookie", fmt.Sprintf("DS=%s", dsToken))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Read the full response content
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("Failed to read response: %v", err)
	}

	// Directly use the response content as UUID
	return &NonceResponse{
		Uuid: strings.TrimSpace(string(body)),
	}, nil
}

// uploadFile uploads a file.
func uploadFile(dsToken, filePath string) (*UploadResponse, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	body := &bytes.Buffer{}
	writer := multipart.NewWriter(body)
	part, err := writer.CreateFormFile("file", filepath.Base(filePath))
	if err != nil {
		return nil, err
	}

	if _, err := io.Copy(part, file); err != nil {
		return nil, err
	}
	writer.Close()

	req, _ := http.NewRequest("POST", "https://you.com/api/upload", body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	req.Header.Set("Cookie", fmt.Sprintf("DS=%s", dsToken))

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var uploadResp UploadResponse
	if err := json.NewDecoder(resp.Body).Decode(&uploadResp); err != nil {
		return nil, err
	}
	return &uploadResp, nil
}

// countTokens calculates the number of tokens in messages (using character estimation method).
func countTokens(messages []Message) (int, error) {
	totalTokens := 0
	for _, msg := range messages {
		content := msg.Content
		englishCount := 0
		chineseCount := 0

		// Traverse each character
		for _, r := range content {
			if r <= 127 { // ASCII characters (English and symbols)
				englishCount++
			} else { // Non-ASCII characters (Chinese, etc.)
				chineseCount++
			}
		}

		// Calculate tokens: English characters * 0.3 + Chinese characters * 0.6
		tokens := int(float64(englishCount)*0.3 + float64(chineseCount)*1)

		// Add tokens for role name (about 2 tokens)
		totalTokens += tokens + 2
	}
	return totalTokens, nil
}

// convertSystemToUser converts system messages to the first user message.
func convertSystemToUser(messages []Message) []Message {
	if len(messages) == 0 {
		return messages
	}

	var systemContent strings.Builder
	var newMessages []Message
	var systemFound bool

	// Collect all system messages
	for _, msg := range messages {
		if msg.Role == "system" {
			if systemContent.Len() > 0 {
				systemContent.WriteString("\n")
			}
			systemContent.WriteString(msg.Content)
			systemFound = true
		} else {
			newMessages = append(newMessages, msg)
		}
	}

	// If there are system messages, use them as the first user message
	if systemFound {
		newMessages = append([]Message{{
			Role:    "user",
			Content: systemContent.String(),
		}}, newMessages...)
	}

	return newMessages
}

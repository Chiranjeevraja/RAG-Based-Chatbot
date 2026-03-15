import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { Send, Trash2 } from "lucide-react";

const SAMPLE_QUESTIONS = [
  "What is this video about?",
  "What are the main points discussed?",
  "What do viewers think about this video?",
  "Summarize the key takeaways.",
];

function TypingIndicator() {
  return (
    <div className="message assistant">
      <div className="avatar bot-avatar">🤖</div>
      <div className="bubble">
        <div className="typing-indicator">
          <div className="typing-dot" />
          <div className="typing-dot" />
          <div className="typing-dot" />
        </div>
      </div>
    </div>
  );
}

function Sources({ sources }) {
  if (!sources?.length) return null;
  return (
    <div className="sources">
      <p className="sources-label">Sources</p>
      {sources.map((s, i) => (
        <span key={i} className="source-chip">
          <span className={`dot ${s.source === "transcript" ? "transcript" : "comments"}`} />
          {s.source === "transcript" ? "Transcript" : "Comments"} · {Math.round(s.score * 100)}%
        </span>
      ))}
    </div>
  );
}

function Message({ msg }) {
  return (
    <div className={`message ${msg.role}`}>
      <div className={`avatar ${msg.role === "user" ? "user-avatar" : "bot-avatar"}`}>
        {msg.role === "user" ? "👤" : "🤖"}
      </div>
      <div className="bubble">
        {msg.role === "assistant" ? (
          <ReactMarkdown>{msg.content}</ReactMarkdown>
        ) : (
          msg.content
        )}
        {msg.sources && <Sources sources={msg.sources} />}
      </div>
    </div>
  );
}

export default function ChatInterface({ videoId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleSend = async (question) => {
    const q = (question || input).trim();
    if (!q || loading) return;

    const userMsg = { role: "user", content: q };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    // Build history (exclude sources metadata for API)
    const history = messages.map((m) => ({ role: m.role, content: m.content }));

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          video_id: videoId || null,
          history,
          stream: true,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        setMessages((prev) => [
          ...prev,
          { role: "assistant", content: `Error: ${err.detail || "Something went wrong."}` },
        ]);
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let fullText = "";
      let sources = null;

      // Add a placeholder for the streaming message
      setMessages((prev) => [...prev, { role: "assistant", content: "", sources: null }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        // __SOURCES__ line and first content token may arrive in the same chunk
        if (chunk.includes("__SOURCES__")) {
          const nlIdx = chunk.indexOf("\n");
          const sourcesLine = nlIdx !== -1 ? chunk.substring(0, nlIdx) : chunk;
          const rest = nlIdx !== -1 ? chunk.substring(nlIdx + 1) : "";

          try {
            sources = JSON.parse(sourcesLine.replace("__SOURCES__", "").trim());
          } catch (_) {}

          if (rest) {
            fullText += rest;
            setMessages((prev) => {
              const updated = [...prev];
              updated[updated.length - 1] = { role: "assistant", content: fullText, sources };
              return updated;
            });
          }
        } else {
          fullText += chunk;
          setMessages((prev) => {
            const updated = [...prev];
            updated[updated.length - 1] = { role: "assistant", content: fullText, sources };
            return updated;
          });
        }
      }

      // Finalize with sources
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: fullText || "No response generated.",
          sources,
        };
        return updated;
      });
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Network error. Please check the backend is running." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const placeholder = videoId
    ? "Ask anything about this video..."
    : "Ask anything about the indexed videos...";

  return (
    <div className="chat-area">
      <div className="chat-header">
        <div className="chat-title">
          💬 Chat
          {videoId && <span className="chat-subtitle">· {videoId}</span>}
        </div>
        {messages.length > 0 && (
          <button className="clear-chat-btn" onClick={() => setMessages([])}>
            <Trash2 size={12} style={{ display: "inline", marginRight: 4 }} />
            Clear
          </button>
        )}
      </div>

      <div className="messages">
        {messages.length === 0 ? (
          <div className="welcome">
            <div className="welcome-icon">🎬</div>
            <h3>Ask about any indexed video</h3>
            <p>
              Index a YouTube video on the left, then ask questions about its
              transcript or what viewers are saying.
            </p>
            <div className="welcome-tips">
              {SAMPLE_QUESTIONS.map((q, i) => (
                <button key={i} className="tip" onClick={() => handleSend(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg, i) => <Message key={i} msg={msg} />)
        )}

        {loading && messages[messages.length - 1]?.role !== "assistant" && <TypingIndicator />}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-area">
        <div className="chat-input-form">
          <textarea
            ref={textareaRef}
            className="chat-input"
            rows={1}
            placeholder={placeholder}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              e.target.style.height = "auto";
              e.target.style.height = Math.min(e.target.scrollHeight, 120) + "px";
            }}
            onKeyDown={handleKeyDown}
            disabled={loading}
          />
          <button
            className="send-btn"
            onClick={() => handleSend()}
            disabled={loading || !input.trim()}
            title="Send (Enter)"
          >
            <Send size={16} />
          </button>
        </div>
        <p className="input-hint">Press Enter to send · Shift+Enter for new line</p>
      </div>
    </div>
  );
}

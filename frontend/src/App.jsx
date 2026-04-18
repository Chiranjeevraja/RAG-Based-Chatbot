import { useState, useEffect } from "react";
import YouTubeInput from "./components/YouTubeInput";
import StoredVideos from "./components/StoredVideos";
import ChatInterface from "./components/ChatInterface";
import AnalysisPanel from "./components/AnalysisPanel";

export default function App() {
  const [activeVideoId, setActiveVideoId] = useState(null);
  const [activeVideoTitle, setActiveVideoTitle] = useState("");
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [activeTab, setActiveTab] = useState("chat"); // "chat" | "analysis"

  const handleProcessed = () => {
    setRefreshTrigger((t) => t + 1);
  };

  // When the active video changes, reset to "chat" tab
  useEffect(() => {
    setActiveTab("chat");
  }, [activeVideoId]);

  const handleVideoSelect = (videoId, videoTitle) => {
    setActiveVideoId(videoId);
    setActiveVideoTitle(videoTitle || "");
  };

  return (
    <div className="app">
      <header className="header">
        <img src="/bits-logo.png" alt="BITS Pilani" style={{ height: 48, width: "auto", objectFit: "contain", flexShrink: 0 }} />
        <div style={{ borderLeft: "1px solid var(--border)", paddingLeft: 16, marginLeft: 4 }}>
          <h1>AI-Driven Automobile Insights Chatbot</h1>
          <p>Audio Intelligence · Document Analysis · RAG-Based Q&amp;A</p>
        </div>
      </header>

      <main className="main">
        <aside className="sidebar">
          <YouTubeInput
            onProcessed={handleProcessed}
            onVideoSelect={(id) => handleVideoSelect(id, "")}
          />
          <StoredVideos
            activeVideoId={activeVideoId}
            onSelect={(videoId, videoTitle) => handleVideoSelect(videoId, videoTitle)}
            refreshTrigger={refreshTrigger}
          />
        </aside>

        {/* Right panel */}
        <div style={{ display: "flex", flexDirection: "column", flex: 1, minWidth: 0, height: "100%" }}>
          {/* Tabs */}
          <div
            style={{
              display: "flex",
              gap: 4,
              padding: "10px 16px 0",
              background: "var(--bg-card)",
              borderBottom: "1px solid var(--border)",
              flexShrink: 0,
            }}
          >
            <button
              onClick={() => setActiveTab("chat")}
              style={{
                background: activeTab === "chat" ? "var(--accent, #5865f2)" : "transparent",
                color: activeTab === "chat" ? "#fff" : "var(--text-muted)",
                border: "none",
                borderRadius: "6px 6px 0 0",
                padding: "7px 18px",
                fontSize: 13,
                fontWeight: 600,
                cursor: "pointer",
                transition: "background 0.15s, color 0.15s",
              }}
            >
              💬 Chat
            </button>
            <button
              onClick={() => setActiveTab("analysis")}
              style={{
                background: activeTab === "analysis" ? "var(--accent, #5865f2)" : "transparent",
                color: activeTab === "analysis" ? "#fff" : "var(--text-muted)",
                border: "none",
                borderRadius: "6px 6px 0 0",
                padding: "7px 18px",
                fontSize: 13,
                fontWeight: 600,
                cursor: "pointer",
                transition: "background 0.15s, color 0.15s",
              }}
            >
              📊 Analysis
            </button>
          </div>

          {/* Tab content — both always mounted to preserve state */}
          <div style={{ flex: 1, minHeight: 0, display: "flex", flexDirection: "column", overflow: "hidden" }}>
            <div style={{ flex: 1, minHeight: 0, display: activeTab === "chat" ? "flex" : "none", flexDirection: "column" }}>
              <ChatInterface videoId={activeVideoId} />
            </div>
            <div style={{ flex: 1, minHeight: 0, display: activeTab === "analysis" ? "flex" : "none", flexDirection: "column" }}>
              <AnalysisPanel videoId={activeVideoId} videoTitle={activeVideoTitle} />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

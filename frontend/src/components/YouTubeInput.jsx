import { useState } from "react";
import { Link, MessageSquare, Loader2, CheckCircle2, AlertCircle } from "lucide-react";

export default function YouTubeInput({ onProcessed, onVideoSelect }) {
  const [url, setUrl] = useState("");
  const [includeComments, setIncludeComments] = useState(true);
  const [useWhisper, setUseWhisper] = useState(false);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null); // null | { type, data }

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!url.trim() || loading) return;

    setLoading(true);
    setStatus(null);

    try {
      const res = await fetch("/api/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url.trim(), include_comments: includeComments, use_whisper: useWhisper }),
      });

      const data = await res.json();

      if (!res.ok) {
        setStatus({ type: "error", message: data.detail || "Processing failed." });
        return;
      }

      setStatus({ type: "success", data });
      onProcessed?.(data);
      onVideoSelect?.(data.video_id);
    } catch (err) {
      setStatus({ type: "error", message: "Network error. Is the backend running?" });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <p className="card-title">Add YouTube Video</p>
      <form className="url-form" onSubmit={handleSubmit}>
        <div className="input-wrapper">
          <Link size={15} className="input-icon" />
          <input
            type="url"
            className="url-input"
            placeholder="https://www.youtube.com/watch?v=..."
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            disabled={loading}
            required
          />
        </div>

        <label className="options-row">
          <input
            type="checkbox"
            checked={includeComments}
            onChange={(e) => setIncludeComments(e.target.checked)}
            disabled={loading}
          />
          Include viewer comments
        </label>

        <label className="options-row">
          <input
            type="checkbox"
            checked={useWhisper}
            onChange={(e) => setUseWhisper(e.target.checked)}
            disabled={loading}
          />
          Force Whisper transcription
        </label>

        <button type="submit" className="process-btn" disabled={loading || !url.trim()}>
          {loading ? (
            <>
              <span className="spinner" />
              Processing...
            </>
          ) : (
            <>
              <MessageSquare size={15} />
              Extract & Index
            </>
          )}
        </button>
      </form>

      {status && (
        <div style={{ marginTop: 12 }}>
          {status.type === "error" && (
            <div className="error-banner">
              <AlertCircle size={14} />
              {status.message}
            </div>
          )}

          {status.type === "success" && (
            <div className={`card status-card success`} style={{ marginTop: 0, padding: "12px" }}>
              <p className="status-title" style={{ color: "var(--green)" }}>
                <CheckCircle2 size={14} style={{ display: "inline", marginRight: 6 }} />
                Indexed successfully
              </p>
              <div className="status-row">
                <span>Video</span>
                <span className="status-val" style={{ maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {status.data.title || status.data.video_id}
                </span>
              </div>
              <div className="status-row">
                <span>Transcript via</span>
                <span className="status-val">{status.data.transcript_method === "whisper" ? "🎙 Whisper" : "📄 Captions"}</span>
              </div>
              <div className="status-row">
                <span>Transcript chunks</span>
                <span className="status-val">{status.data.transcript_chunks}</span>
              </div>
              <div className="status-row">
                <span>Comment chunks</span>
                <span className="status-val">{status.data.comment_chunks}</span>
              </div>
              <div className="status-row">
                <span>Total chunks</span>
                <span className="status-val" style={{ color: "var(--accent)" }}>
                  {status.data.total_chunks}
                </span>
              </div>

              {status.data.warnings?.length > 0 && (
                <div className="warning-list">
                  {status.data.warnings.map((w, i) => (
                    <div key={i} className="warning-item">⚠ {w}</div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

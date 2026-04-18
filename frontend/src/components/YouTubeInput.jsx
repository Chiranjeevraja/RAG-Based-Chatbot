import { useState, useRef } from "react";
import { Link, MessageSquare, CheckCircle2, AlertCircle, Upload, Music } from "lucide-react";

function isTeamBHPUrl(url) {
  return url.toLowerCase().includes("team-bhp.com");
}

const AUDIO_ACCEPT = ".mp3,.wav,.m4a,.ogg,.flac,.webm,.mp4,.aac,.wma";

export default function YouTubeInput({ onProcessed, onVideoSelect }) {
  const [mode, setMode] = useState("url"); // "url" | "audio"

  // URL mode state
  const [url, setUrl] = useState("");
  const [includeComments, setIncludeComments] = useState(true);
  const [useWhisper, setUseWhisper] = useState(false);

  // Audio mode state
  const [audioFile, setAudioFile] = useState(null);
  const [audioTitle, setAudioTitle] = useState("");
  const fileInputRef = useRef(null);

  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState(null);

  const isTBHP = isTeamBHPUrl(url);

  // ── URL submit ────────────────────────────────────────────────────────────────

  const handleUrlSubmit = async (e) => {
    e.preventDefault();
    if (!url.trim() || loading) return;

    setLoading(true);
    setStatus(null);

    try {
      const body = { url: url.trim() };
      if (!isTBHP) {
        body.include_comments = includeComments;
        body.use_whisper = useWhisper;
      }

      const res = await fetch("/api/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await res.json();
      if (!res.ok) {
        setStatus({ type: "error", message: data.detail || "Processing failed." });
        return;
      }

      setStatus({ type: "success", data });
      onProcessed?.(data);
      onVideoSelect?.(data.video_id);
    } catch {
      setStatus({ type: "error", message: "Network error. Is the backend running?" });
    } finally {
      setLoading(false);
    }
  };

  // ── Audio submit ──────────────────────────────────────────────────────────────

  const handleAudioSubmit = async (e) => {
    e.preventDefault();
    if (!audioFile || loading) return;

    setLoading(true);
    setStatus(null);

    try {
      const form = new FormData();
      form.append("file", audioFile);
      if (audioTitle.trim()) form.append("title", audioTitle.trim());

      const res = await fetch("/api/upload-audio", { method: "POST", body: form });
      const data = await res.json();

      if (!res.ok) {
        setStatus({ type: "error", message: data.detail || "Upload failed." });
        return;
      }

      setStatus({ type: "success", data });
      onProcessed?.(data);
      onVideoSelect?.(data.video_id);
    } catch {
      setStatus({ type: "error", message: "Network error. Is the backend running?" });
    } finally {
      setLoading(false);
    }
  };

  // ── Drag-and-drop ─────────────────────────────────────────────────────────────

  const [dragging, setDragging] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) {
      setAudioFile(f);
      if (!audioTitle.trim()) setAudioTitle(f.name.replace(/\.[^.]+$/, ""));
    }
  };

  // ── Mode switch (clear status) ────────────────────────────────────────────────

  const switchMode = (m) => {
    setMode(m);
    setStatus(null);
  };

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div className="card">
      {/* Mode toggle */}
      <div style={{ display: "flex", gap: 4, marginBottom: 12 }}>
        {[
          { key: "url",   label: "URL",        icon: <Link size={13} /> },
          { key: "audio", label: "Audio File",  icon: <Music size={13} /> },
        ].map(({ key, label, icon }) => (
          <button
            key={key}
            onClick={() => switchMode(key)}
            style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 5,
              padding: "6px 0",
              fontSize: 12,
              fontWeight: 600,
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-sm)",
              background: mode === key ? "var(--accent)" : "var(--bg-input)",
              color: mode === key ? "#fff" : "var(--text-secondary)",
              cursor: "pointer",
              transition: "background 0.15s, color 0.15s",
            }}
          >
            {icon} {label}
          </button>
        ))}
      </div>

      {/* ── URL mode ── */}
      {mode === "url" && (
        <>
          <p className="card-title">Add YouTube Video / TeamBHP Website</p>
          <form className="url-form" onSubmit={handleUrlSubmit}>
            <div className="input-wrapper">
              <Link size={15} className="input-icon" />
              <input
                type="url"
                className="url-input"
                placeholder={
                  isTBHP
                    ? "https://www.team-bhp.com/forum/..."
                    : "https://www.youtube.com/watch?v=... or team-bhp.com/..."
                }
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                disabled={loading}
                required
              />
            </div>

            {!isTBHP && (
              <>
                <label className="options-row">
                  <input
                    type="checkbox"
                    checked={includeComments}
                    onChange={(e) => setIncludeComments(e.target.checked)}
                    disabled={loading}
                  />
                  Comments
                </label>
                <label className="options-row">
                  <input
                    type="checkbox"
                    checked={useWhisper}
                    onChange={(e) => setUseWhisper(e.target.checked)}
                    disabled={loading}
                  />
                  Transcription
                </label>
              </>
            )}

            <button type="submit" className="process-btn" disabled={loading || !url.trim()}>
              {loading ? (
                <><span className="spinner" />{isTBHP ? "Scraping..." : "Processing..."}</>
              ) : (
                <><MessageSquare size={15} />{isTBHP ? "Scrape & Index" : "Extract & Index"}</>
              )}
            </button>
          </form>
        </>
      )}

      {/* ── Audio mode ── */}
      {mode === "audio" && (
        <>
          <p className="card-title">Upload Audio File</p>
          <form className="url-form" onSubmit={handleAudioSubmit}>

            {/* Drop zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: `2px dashed ${dragging ? "var(--accent)" : "var(--border)"}`,
                borderRadius: "var(--radius-sm)",
                padding: "20px 12px",
                textAlign: "center",
                cursor: "pointer",
                background: dragging ? "var(--accent-light)" : "var(--bg-input)",
                transition: "all 0.15s",
              }}
            >
              <Upload size={22} style={{ color: "var(--accent)", marginBottom: 6 }} />
              {audioFile ? (
                <p style={{ fontSize: 13, color: "var(--text-primary)", fontWeight: 600 }}>
                  {audioFile.name}
                  <span style={{ display: "block", fontSize: 11, color: "var(--text-muted)", fontWeight: 400, marginTop: 2 }}>
                    {(audioFile.size / 1024 / 1024).toFixed(1)} MB
                  </span>
                </p>
              ) : (
                <>
                  <p style={{ fontSize: 13, color: "var(--text-secondary)", fontWeight: 500 }}>
                    Drop audio here or click to browse
                  </p>
                  <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>
                    mp3 · wav · m4a · ogg · flac · webm · aac
                  </p>
                </>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept={AUDIO_ACCEPT}
                style={{ display: "none" }}
                onChange={(e) => {
                  const f = e.target.files[0];
                  if (f) {
                    setAudioFile(f);
                    if (!audioTitle.trim()) setAudioTitle(f.name.replace(/\.[^.]+$/, ""));
                  }
                }}
              />
            </div>

            {/* Optional title */}
            <input
              type="text"
              className="url-input"
              placeholder="Title (optional — defaults to filename)"
              value={audioTitle}
              onChange={(e) => setAudioTitle(e.target.value)}
              disabled={loading}
              style={{ marginTop: 2 }}
            />

            <button type="submit" className="process-btn" disabled={loading || !audioFile}>
              {loading ? (
                <><span className="spinner" />Transcribing...</>
              ) : (
                <><Upload size={15} />Transcribe & Index</>
              )}
            </button>
          </form>
        </>
      )}

      {/* ── Status ── */}
      {status && (
        <div style={{ marginTop: 12 }}>
          {status.type === "error" && (
            <div className="error-banner">
              <AlertCircle size={14} />
              {status.message}
            </div>
          )}

          {status.type === "success" && status.data.source_type === "teambhp" && (
            <div className="card status-card success" style={{ marginTop: 0, padding: "12px" }}>
              <p className="status-title" style={{ color: "var(--green)" }}>
                <CheckCircle2 size={14} style={{ display: "inline", marginRight: 6 }} />
                Thread indexed successfully
              </p>
              <div className="status-row"><span>Thread</span>
                <span className="status-val" style={{ maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {status.data.title}
                </span>
              </div>
              <div className="status-row"><span>Source</span><span className="status-val">TeamBHP</span></div>
              <div className="status-row"><span>Post chunks</span>
                <span className="status-val" style={{ color: "var(--accent)" }}>
                  {status.data.post_chunks ?? status.data.total_chunks}
                </span>
              </div>
            </div>
          )}

          {status.type === "success" && status.data.source_type === "audio" && (
            <div className="card status-card success" style={{ marginTop: 0, padding: "12px" }}>
              <p className="status-title" style={{ color: "var(--green)" }}>
                <CheckCircle2 size={14} style={{ display: "inline", marginRight: 6 }} />
                Audio indexed successfully
              </p>
              <div className="status-row"><span>Title</span>
                <span className="status-val" style={{ maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {status.data.title}
                </span>
              </div>
              <div className="status-row"><span>Transcribed via</span><span className="status-val">🎙 Whisper</span></div>
              <div className="status-row"><span>Transcript chunks</span>
                <span className="status-val" style={{ color: "var(--accent)" }}>{status.data.transcript_chunks}</span>
              </div>
            </div>
          )}

          {status.type === "success" && status.data.source_type === "youtube" && (
            <div className="card status-card success" style={{ marginTop: 0, padding: "12px" }}>
              <p className="status-title" style={{ color: "var(--green)" }}>
                <CheckCircle2 size={14} style={{ display: "inline", marginRight: 6 }} />
                Indexed successfully
              </p>
              <div className="status-row"><span>Video</span>
                <span className="status-val" style={{ maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {status.data.title || status.data.video_id}
                </span>
              </div>
              <div className="status-row"><span>Transcript via</span>
                <span className="status-val">
                  {status.data.transcript_method === "whisper" ? "🎙 Whisper" : "📄 Captions"}
                </span>
              </div>
              <div className="status-row"><span>Transcript chunks</span><span className="status-val">{status.data.transcript_chunks}</span></div>
              <div className="status-row"><span>Comment chunks</span><span className="status-val">{status.data.comment_chunks}</span></div>
              <div className="status-row"><span>Total chunks</span>
                <span className="status-val" style={{ color: "var(--accent)" }}>{status.data.total_chunks}</span>
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

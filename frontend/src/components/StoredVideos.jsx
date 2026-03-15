import { useEffect, useState } from "react";
import { Trash2, RefreshCw, Video } from "lucide-react";

export default function StoredVideos({ activeVideoId, onSelect, refreshTrigger }) {
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchVideos = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/videos");
      if (res.ok) {
        const data = await res.json();
        setVideos(data.videos || []);
      }
    } catch (_) {}
    setLoading(false);
  };

  useEffect(() => {
    fetchVideos();
  }, [refreshTrigger]);

  const handleDelete = async (e, videoId) => {
    e.stopPropagation();
    try {
      await fetch(`/api/videos/${videoId}`, { method: "DELETE" });
      setVideos((prev) => prev.filter((v) => v.video_id !== videoId));
      if (activeVideoId === videoId) onSelect(null);
    } catch (_) {}
  };

  return (
    <div className="card">
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 12 }}>
        <p className="card-title" style={{ marginBottom: 0 }}>Indexed Videos</p>
        <button
          onClick={fetchVideos}
          style={{ background: "none", border: "none", cursor: "pointer", color: "var(--text-muted)", padding: "2px" }}
          title="Refresh"
        >
          <RefreshCw size={13} style={{ animation: loading ? "spin 1s linear infinite" : "none" }} />
        </button>
      </div>

      {!activeVideoId && videos.length > 0 && (
        <div className="warning-item" style={{ marginBottom: 8, color: "var(--text-secondary)", background: "var(--accent-light)" }}>
          Select a video to filter chat responses
        </div>
      )}

      {activeVideoId && (
        <div
          className="warning-item"
          style={{ marginBottom: 8, color: "var(--accent)", background: "var(--accent-light)", cursor: "pointer" }}
          onClick={() => onSelect(null)}
        >
          ✓ Filtering by selected video — click to show all
        </div>
      )}

      <div className="video-list">
        {videos.length === 0 ? (
          <p className="empty-state">No videos indexed yet</p>
        ) : (
          videos.map((v) => (
            <div
              key={v.video_id}
              className={`video-item ${activeVideoId === v.video_id ? "active" : ""}`}
              onClick={() => onSelect(activeVideoId === v.video_id ? null : v.video_id, v.title || "")}
            >
              <Video size={12} style={{ color: "var(--text-muted)", flexShrink: 0 }} />
              <span className="video-item-title" title={v.title || v.video_id}>
                {v.title || v.video_id}
              </span>
              <span className="video-item-id">{v.video_id}</span>
              <button
                className="delete-btn"
                onClick={(e) => handleDelete(e, v.video_id)}
                title="Delete"
              >
                <Trash2 size={12} />
              </button>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

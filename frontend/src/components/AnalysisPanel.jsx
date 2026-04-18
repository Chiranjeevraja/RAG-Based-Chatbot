import { useState, useEffect } from "react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";

// ── Design tokens ──────────────────────────────────────────────────────────────

const C = {
  positive: "#16a34a",
  neutral:  "#5865f2",
  negative: "#dc2626",
  bg:       "#f4f5f9",
  card:     "#ffffff",
  border:   "#d4d6e3",
  accent:   "#5865f2",
  dim:      "#e8eaf0",
  muted:    "#8888a8",
  text:     "#4a4a6a",
  bright:   "#1a1a2e",
};

const glow = (color, px = 8) => `0 0 ${px}px ${color}55, 0 0 ${px * 2}px ${color}22`;

// ── Tiny helpers ───────────────────────────────────────────────────────────────

function sentColor(s) { return C[s] || C.neutral; }

function Pip({ sentiment }) {
  const col = sentColor(sentiment);
  return (
    <span style={{
      display: "inline-block", width: 8, height: 8, borderRadius: "50%",
      background: col, boxShadow: glow(col, 6), flexShrink: 0,
    }} />
  );
}

function Badge({ sentiment }) {
  const col = sentColor(sentiment);
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      padding: "3px 10px", borderRadius: 3,
      border: `1px solid ${col}44`,
      background: `${col}11`,
      color: col, fontSize: 10, fontWeight: 700,
      letterSpacing: "0.1em", textTransform: "uppercase",
      boxShadow: glow(col, 4),
      fontFamily: "monospace",
    }}>
      <Pip sentiment={sentiment} />
      {sentiment}
    </span>
  );
}

function GlowBar({ score, height = 4 }) {
  const col = score >= 0.6 ? C.positive : score <= 0.4 ? C.negative : C.neutral;
  const pct = Math.round(score * 100);
  return (
    <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{
        flex: 1, height, borderRadius: 2,
        background: C.dim, overflow: "hidden",
      }}>
        <div style={{
          width: `${pct}%`, height: "100%",
          background: `linear-gradient(90deg, ${col}88, ${col})`,
          boxShadow: glow(col, 4),
          borderRadius: 2,
          transition: "width 0.4s ease",
        }} />
      </div>
      <span style={{ fontSize: 10, fontFamily: "monospace", color: col, width: 28, textAlign: "right" }}>
        {pct}%
      </span>
    </div>
  );
}

// ── Feature row ────────────────────────────────────────────────────────────────

function FeatureRow({ f }) {
  const [open, setOpen] = useState(false);
  const col = sentColor(f.sentiment);
  const hasVerbatim = f.verbatim?.length > 0;

  return (
    <div style={{ borderBottom: `1px solid ${C.border}` }}>
      {/* Feature header row — clickable if verbatim exists */}
      <div
        onClick={() => hasVerbatim && setOpen(o => !o)}
        style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "7px 4px",
          cursor: hasVerbatim ? "pointer" : "default",
        }}
      >
        {/* Expand indicator */}
        <span style={{
          width: 14, fontSize: 8, color: hasVerbatim ? col : C.dim,
          flexShrink: 0, textAlign: "center",
        }}>
          {hasVerbatim ? (open ? "▼" : "▶") : "·"}
        </span>

        {/* Feature name */}
        <span style={{
          flex: 1, fontSize: 11, color: C.text,
          textTransform: "capitalize", fontFamily: "monospace",
        }}>
          {f.name}
        </span>

        {/* Mention count */}
        {f.mention_count > 0 && (
          <span style={{
            fontSize: 9, fontFamily: "monospace",
            color: C.muted, marginRight: 6,
          }}>
            {f.mention_count}×
          </span>
        )}

        <Badge sentiment={f.sentiment} />
      </div>

      {/* Verbatim excerpts — shown on expand */}
      {open && hasVerbatim && (
        <div style={{
          margin: "2px 0 8px 24px",
          padding: "10px 12px",
          background: `${col}08`,
          border: `1px solid ${col}22`,
          borderRadius: 4,
        }}>
          <p style={{
            margin: "0 0 8px", fontSize: 8, fontWeight: 700,
            letterSpacing: "0.15em", color: col, textTransform: "uppercase",
          }}>
            What was said
          </p>
          {f.verbatim.map((text, i) => (
            <div key={i} style={{
              display: "flex", gap: 8, marginBottom: i < f.verbatim.length - 1 ? 8 : 0,
              alignItems: "flex-start",
            }}>
              <span style={{ color: col, fontSize: 9, marginTop: 2, flexShrink: 0 }}>❝</span>
              <span style={{
                fontSize: 11, color: C.text, lineHeight: 1.6,
                fontStyle: "italic",
              }}>
                {text}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Model panel (expanded inside company) ─────────────────────────────────────

function ModelPanel({ modelName, data }) {
  const [open, setOpen] = useState(false);
  const col = sentColor(data.overall_sentiment);
  if (!data.features?.length) return null;
  const featureMentionTotal = (data.features || []).reduce((sum, f) => sum + (f.mention_count || 0), 0);

  return (
    <div style={{ marginBottom: 8 }}>
      {/* Model row */}
      <div
        onClick={() => setOpen(o => !o)}
        style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "8px 12px", borderRadius: 4, cursor: "pointer",
          background: open ? `${col}08` : "transparent",
          border: `1px solid ${open ? col + "33" : C.border}`,
          transition: "all 0.2s",
        }}
      >
        <span style={{
          width: 18, height: 18, display: "flex", alignItems: "center", justifyContent: "center",
          border: `1px solid ${col}55`, borderRadius: 3, color: col, fontSize: 9,
          flexShrink: 0,
        }}>
          {open ? "▼" : "▶"}
        </span>
        <span style={{ flex: 1, fontSize: 12, fontWeight: 600, color: C.bright, letterSpacing: "0.05em" }}>
          {modelName}
        </span>
        <Badge sentiment={data.overall_sentiment} />
        <span style={{ fontSize: 10, color: C.muted, fontFamily: "monospace" }}>
          {featureMentionTotal}×
        </span>
      </div>

      {/* Model details */}
      {open && (
        <div style={{
          marginTop: 6, padding: "12px 16px",
          background: `${col}06`,
          border: `1px solid ${col}22`,
          borderRadius: 4,
        }}>
          {/* Features */}
          {data.features?.length > 0 && (
            <div style={{ marginBottom: 12 }}>
              <p style={{
                margin: "0 0 8px", fontSize: 9, fontWeight: 700,
                letterSpacing: "0.15em", color: C.muted, textTransform: "uppercase",
              }}>
                Features
              </p>
              {data.features.map(f => <FeatureRow key={f.name} f={f} />)}
            </div>
          )}

        </div>
      )}
    </div>
  );
}

// ── Company card ───────────────────────────────────────────────────────────────

function CompanyCard({ company, data }) {
  const [open, setOpen] = useState(false);
  const col = sentColor(data.overall_sentiment);
  const modelsWithFeatures = Object.entries(data.models || {}).filter(([, m]) => m.features?.length > 0);
  if (modelsWithFeatures.length === 0) return null;
  const modelCount = modelsWithFeatures.length;
  const displayedMentionCount = modelsWithFeatures.reduce(
    (sum, [, m]) => sum + (m.features || []).reduce((s, f) => s + (f.mention_count || 0), 0), 0
  );

  return (
    <div style={{
      marginBottom: 12,
      border: `1px solid ${open ? col + "55" : C.border}`,
      borderRadius: 6,
      overflow: "hidden",
      boxShadow: open ? glow(col, 12) : "none",
      transition: "box-shadow 0.3s, border-color 0.3s",
    }}>
      {/* Company header — always visible */}
      <div
        onClick={() => setOpen(o => !o)}
        style={{
          display: "flex", alignItems: "center", gap: 14,
          padding: "14px 18px", cursor: "pointer",
          background: open
            ? `linear-gradient(90deg, ${col}12 0%, ${C.card} 100%)`
            : C.card,
          transition: "background 0.3s",
        }}
      >
        {/* Indicator bar */}
        <div style={{
          width: 3, height: 28, borderRadius: 2,
          background: col, boxShadow: glow(col, 8), flexShrink: 0,
        }} />

        {/* Company name */}
        <span style={{
          flex: 1, fontSize: 15, fontWeight: 700,
          color: C.bright, letterSpacing: "0.08em", textTransform: "uppercase",
        }}>
          {company}
        </span>

        {/* Meta */}
        <span style={{ fontSize: 10, color: C.muted, fontFamily: "monospace", marginRight: 8 }}>
          {modelCount} model{modelCount !== 1 ? "s" : ""} · {displayedMentionCount}×
        </span>

        <Badge sentiment={data.overall_sentiment} />

        <span style={{
          marginLeft: 10, width: 20, height: 20,
          display: "flex", alignItems: "center", justifyContent: "center",
          border: `1px solid ${col}55`, borderRadius: 3,
          color: col, fontSize: 9, flexShrink: 0,
        }}>
          {open ? "▼" : "▶"}
        </span>
      </div>

      {/* Models list */}
      {open && (
        <div style={{
          padding: "14px 18px",
          background: `${C.bg}cc`,
          borderTop: `1px solid ${col}22`,
        }}>
          {Object.entries(data.models || {}).map(([modelName, modelData]) => (
            <ModelPanel key={modelName} modelName={modelName} data={modelData} />
          ))}
        </div>
      )}
    </div>
  );
}

// ── Overview stat box ──────────────────────────────────────────────────────────

function StatBox({ label, children }) {
  return (
    <div style={{
      background: C.card, border: `1px solid ${C.border}`,
      borderRadius: 6, padding: "14px 16px", textAlign: "center",
    }}>
      <p style={{
        margin: "0 0 8px", fontSize: 9, color: C.muted,
        textTransform: "uppercase", letterSpacing: "0.15em", fontWeight: 700,
      }}>{label}</p>
      {children}
    </div>
  );
}

// ── Main panel ─────────────────────────────────────────────────────────────────

export default function AnalysisPanel({ videoId, videoTitle }) {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState("");

  useEffect(() => {
    if (!videoId) { setAnalysis(null); setError(""); return; }
    setAnalysis(null);
    fetch(`/api/analysis/${videoId}`)
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d) setAnalysis(d); })
      .catch(() => {});
  }, [videoId]);

  const handleRun = async () => {
    if (!videoId) return;
    setLoading(true); setError("");
    try {
      const res = await fetch(`/api/analyze/${videoId}`, { method: "POST" });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setError(body.detail || "Analysis failed."); return;
      }
      const cached = await fetch(`/api/analysis/${videoId}`);
      if (cached.ok) setAnalysis(await cached.json());
    } catch (_) { setError("Network error."); }
    finally { setLoading(false); }
  };

  if (!videoId) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: C.muted, fontSize: 13 }}>
        Select a video to run analysis.
      </div>
    );
  }

  const agg = analysis?.aggregated;
  const brandEntries = Object.entries(agg?.brand_analysis || {});

  const pieData = agg ? [
    { name: "Positive", value: agg.sentiment_distribution?.positive ?? 0 },
    { name: "Neutral",  value: agg.sentiment_distribution?.neutral  ?? 0 },
    { name: "Negative", value: agg.sentiment_distribution?.negative ?? 0 },
  ] : [];

  return (
    <div style={{
      height: "100%", overflowY: "auto", padding: "22px 24px",
      boxSizing: "border-box",
      background: C.bg,
      fontFamily: "'Inter', 'Segoe UI', sans-serif",
    }}>

      {/* Header */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 22 }}>
        <div>
          <h2 style={{
            margin: 0, fontSize: 13, fontWeight: 700, color: C.accent,
            textTransform: "uppercase", letterSpacing: "0.2em",
          }}>
            Brand Intelligence
          </h2>
          {videoTitle && (
            <p style={{ margin: "4px 0 0", fontSize: 11, color: C.muted, letterSpacing: "0.05em" }}>
              {videoTitle}
            </p>
          )}
        </div>
        <button
          onClick={handleRun} disabled={loading}
          style={{
            background: loading ? C.dim : "transparent",
            color: loading ? C.muted : C.accent,
            border: `1px solid ${loading ? C.dim : C.accent}`,
            borderRadius: 4, padding: "7px 18px",
            fontSize: 11, fontWeight: 700, cursor: loading ? "not-allowed" : "pointer",
            letterSpacing: "0.1em", textTransform: "uppercase",
            boxShadow: loading ? "none" : glow(C.accent, 6),
            transition: "all 0.2s",
          }}
        >
          {loading ? "Processing…" : agg ? "Re-run" : "Run Analysis"}
        </button>
      </div>

      {/* Loading state */}
      {loading && (
        <div style={{
          background: C.card, border: `1px solid ${C.border}`, borderRadius: 6,
          padding: "40px 20px", textAlign: "center", marginBottom: 16,
        }}>
          <div style={{ fontSize: 11, color: C.accent, letterSpacing: "0.2em", textTransform: "uppercase", marginBottom: 8 }}>
            Pipeline Active
          </div>
          <div style={{ fontSize: 12, color: C.muted }}>
            Extracting brands · Deduplicating · Scoring sentiment…
          </div>
        </div>
      )}

      {/* Error */}
      {!loading && error && (
        <div style={{
          background: `${C.negative}11`, border: `1px solid ${C.negative}44`,
          borderRadius: 6, padding: "12px 16px", marginBottom: 16,
          color: C.negative, fontSize: 12,
        }}>
          {error}
        </div>
      )}

      {/* Empty state */}
      {!loading && !agg && !error && (
        <div style={{
          background: C.card, border: `1px solid ${C.border}`, borderRadius: 6,
          padding: "48px 20px", textAlign: "center",
        }}>
          <div style={{ fontSize: 11, color: C.muted, letterSpacing: "0.15em", textTransform: "uppercase" }}>
            No data · Click Run Analysis
          </div>
        </div>
      )}

      {/* Results */}
      {!loading && agg && (
        <>
          {/* Overview row */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3,1fr)", gap: 10, marginBottom: 16 }}>
            <StatBox label="Overall Sentiment">
              <Badge sentiment={agg.overall_sentiment} />
            </StatBox>
            <StatBox label="Companies">
              <span style={{ fontSize: 24, fontWeight: 700, color: C.accent, fontFamily: "monospace" }}>
                {brandEntries.length}
              </span>
            </StatBox>
            <StatBox label="Chunks Analyzed">
              <span style={{ fontSize: 24, fontWeight: 700, color: C.accent, fontFamily: "monospace" }}>
                {agg.total_chunks_analyzed}
              </span>
            </StatBox>
          </div>

          {/* Pie chart */}
          <div style={{
            background: C.card, border: `1px solid ${C.border}`,
            borderRadius: 6, padding: "14px 18px", marginBottom: 16,
            display: "flex", alignItems: "center", gap: 24,
          }}>
            <div style={{ flexShrink: 0 }}>
              <ResponsiveContainer width={120} height={120}>
                <PieChart>
                  <Pie data={pieData} cx="50%" cy="50%" innerRadius={36} outerRadius={54} dataKey="value" labelLine={false}>
                    {pieData.map((_, i) => (
                      <Cell key={i} fill={[C.positive, C.neutral, C.negative][i]}
                        style={{ filter: `drop-shadow(0 0 6px ${[C.positive, C.neutral, C.negative][i]}88)` }} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={v => `${v}%`}
                    contentStyle={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 6, fontSize: 11 }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div style={{ flex: 1 }}>
              <p style={{ margin: "0 0 10px", fontSize: 9, color: C.muted, letterSpacing: "0.15em", textTransform: "uppercase", fontWeight: 700 }}>
                Sentiment Distribution
              </p>
              {pieData.map((entry, i) => {
                const sent = ["positive", "neutral", "negative"][i];
                const col = sentColor(sent);
                const pct = Math.round(entry.value);
                return (
                  <div key={entry.name} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
                    <Pip sentiment={sent} />
                    <span style={{ fontSize: 11, color: C.text, flex: 1, fontFamily: "monospace" }}>{entry.name}</span>
                    <div style={{ flex: 1, display: "flex", alignItems: "center", gap: 8 }}>
                      <div style={{ flex: 1, height: 3, borderRadius: 2, background: C.dim, overflow: "hidden" }}>
                        <div style={{
                          width: `${pct}%`, height: "100%", borderRadius: 2,
                          background: `linear-gradient(90deg, ${col}88, ${col})`,
                          boxShadow: glow(col, 4),
                          transition: "width 0.4s ease",
                        }} />
                      </div>
                      <span style={{ fontSize: 10, fontFamily: "monospace", color: col, width: 28, textAlign: "right" }}>
                        {pct}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Company cards */}
          <p style={{ margin: "0 0 10px", fontSize: 9, fontWeight: 700, color: C.muted, letterSpacing: "0.15em", textTransform: "uppercase" }}>
            Companies · Click to expand
          </p>
          {brandEntries.length > 0
            ? brandEntries.map(([company, data]) => (
                <CompanyCard key={company} company={company} data={data} />
              ))
            : (
              <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 6, padding: "20px", textAlign: "center", color: C.muted, fontSize: 12 }}>
                No automobile brands detected.
              </div>
            )
          }
        </>
      )}
    </div>
  );
}

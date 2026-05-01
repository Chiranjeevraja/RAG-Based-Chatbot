import { useState } from "react";

// ── SVG generators (return HTML strings for the report document) ──────────────

function svgDonut(pos, neu, neg, sz = 120) {
  const total = pos + neu + neg;
  const cx = sz / 2, cy = sz / 2;
  const R = sz * 0.42, ri = sz * 0.26;

  if (!total || total < 0.01) {
    return `<svg width="${sz}" height="${sz}" viewBox="0 0 ${sz} ${sz}">
      <circle cx="${cx}" cy="${cy}" r="${R}" fill="#e8eaf0"/>
      <circle cx="${cx}" cy="${cy}" r="${ri}" fill="#fff"/>
      <text x="${cx}" y="${cy + 4}" text-anchor="middle" fill="#8888a8" font-size="10" font-family="sans-serif">N/A</text>
    </svg>`;
  }

  const segs = [
    { v: pos, c: "#16a34a" },
    { v: neu, c: "#5865f2" },
    { v: neg, c: "#dc2626" },
  ].filter(s => s.v > 0.01);

  if (segs.length === 1) {
    return `<svg width="${sz}" height="${sz}" viewBox="0 0 ${sz} ${sz}">
      <circle cx="${cx}" cy="${cy}" r="${R}" fill="${segs[0].c}"/>
      <circle cx="${cx}" cy="${cy}" r="${ri}" fill="#fff"/>
    </svg>`;
  }

  const f = n => n.toFixed(2);
  let paths = "";
  let a = -Math.PI / 2;

  for (const seg of segs) {
    const sw = (seg.v / total) * 2 * Math.PI;
    const ea = a + sw;
    const lg = sw > Math.PI ? 1 : 0;
    paths += `<path d="M${f(cx + R * Math.cos(a))},${f(cy + R * Math.sin(a))} A${R},${R} 0 ${lg},1 ${f(cx + R * Math.cos(ea))},${f(cy + R * Math.sin(ea))} L${f(cx + ri * Math.cos(ea))},${f(cy + ri * Math.sin(ea))} A${ri},${ri} 0 ${lg},0 ${f(cx + ri * Math.cos(a))},${f(cy + ri * Math.sin(a))} Z" fill="${seg.c}"/>`;
    a = ea;
  }

  return `<svg width="${sz}" height="${sz}" viewBox="0 0 ${sz} ${sz}">${paths}</svg>`;
}

function svgHBars(items) {
  if (!items.length) return "";
  const max = Math.max(...items.map(i => i.value), 1);
  const barH = 26, gap = 10, LW = 140, BW = 260, VW = 50;
  const svgW = LW + BW + VW;
  const svgH = items.length * (barH + gap) + 10;

  const rows = items.map((item, i) => {
    const y = i * (barH + gap) + 5;
    const totalW = Math.max(4, Math.round((item.value / max) * BW));
    const label = item.label.length > 20 ? item.label.slice(0, 19) + "…" : item.label;
    const clipId = `clip-bar-${i}`;

    let barContent;
    if (item.posRatio != null) {
      const pw = Math.round(totalW * item.posRatio);
      const nw = Math.round(totalW * item.neuRatio);
      const rw = totalW - pw - nw;
      let segs = "", x = LW;
      if (pw > 0) { segs += `<rect x="${x}" y="${y}" width="${pw}" height="${barH}" fill="#16a34a" opacity="0.85"/>`; x += pw; }
      if (nw > 0) { segs += `<rect x="${x}" y="${y}" width="${nw}" height="${barH}" fill="#5865f2" opacity="0.85"/>`; x += nw; }
      if (rw > 0) { segs += `<rect x="${x}" y="${y}" width="${rw}" height="${barH}" fill="#dc2626" opacity="0.85"/>`; }
      barContent = `<clipPath id="${clipId}"><rect x="${LW}" y="${y}" width="${totalW}" height="${barH}" rx="4"/></clipPath><g clip-path="url(#${clipId})">${segs}</g>`;
    } else {
      barContent = `<rect x="${LW}" y="${y}" width="${totalW}" height="${barH}" fill="${item.color}" rx="4" opacity="0.85"/>`;
    }

    const hasTip = item.posCount != null;
    const hoverRect = hasTip
      ? `<rect x="${LW}" y="${y}" width="${totalW}" height="${barH}" fill="transparent" style="cursor:crosshair" onmousemove="showBTip(event,${item.posCount},${item.neuCount},${item.negCount},${item.value})" onmouseleave="hideBTip()"/>`
      : "";
    return `
      <text x="${LW - 10}" y="${y + barH * 0.65}" text-anchor="end" font-size="12" fill="#4a4a6a" font-family="'Segoe UI',sans-serif">${label}</text>
      ${barContent}
      ${hoverRect}
      <text x="${LW + totalW + 8}" y="${y + barH * 0.66}" font-size="12" fill="#4a4a6a" font-family="monospace" font-weight="600">${item.value}</text>
    `;
  }).join("");

  return `<svg width="${svgW}" height="${svgH}" viewBox="0 0 ${svgW} ${svgH}">${rows}</svg>`;
}

// ── Sentiment helpers ──────────────────────────────────────────────────────────

function sentColor(s) {
  return s === "positive" ? "#16a34a" : s === "negative" ? "#dc2626" : "#5865f2";
}

function sentBadge(s) {
  const c = sentColor(s || "neutral");
  return `<span style="display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:3px;border:1px solid ${c}44;background:${c}11;color:${c};font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;font-family:monospace"><span style="width:7px;height:7px;border-radius:50%;background:${c};display:inline-block;flex-shrink:0"></span>${s || "neutral"}</span>`;
}

function pctBar(pct, color, width = 200) {
  const w = Math.round(Math.max(0, Math.min(100, pct)) / 100 * width);
  return `<div style="display:flex;align-items:center;gap:8px">
    <div style="width:${width}px;height:5px;border-radius:3px;background:#e8eaf0;overflow:hidden;flex-shrink:0">
      <div style="width:${w}px;height:100%;background:${color};border-radius:3px"></div>
    </div>
    <span style="font-size:11px;font-family:monospace;color:${color};width:36px;text-align:right">${Math.round(pct)}%</span>
  </div>`;
}

// ── Per-brand insight card (Praised for / Criticized for) ────────────────────

function brandInsightsHtml(name, cData) {
  const col = sentColor(cData.overall_sentiment);

  // Collect features across all models
  const posFeatures = [];
  const negFeatures = [];
  const posFromModel = [];
  const negFromModel = [];

  for (const modelData of Object.values(cData.models || {})) {
    for (const f of modelData.features || []) {
      const item = { name: f.name, count: f.mention_count || f.count || 1 };
      if (f.sentiment === "positive") posFeatures.push(item);
      else if (f.sentiment === "negative") negFeatures.push(item);
    }
    if (modelData.top_positives?.length) posFromModel.push(...modelData.top_positives);
    if (modelData.top_negatives?.length) negFromModel.push(...modelData.top_negatives);
  }

  // Sort by mention count descending, deduplicate by name
  const dedupe = (arr) => {
    const seen = new Set();
    return arr
      .sort((a, b) => b.count - a.count)
      .filter(f => {
        const k = f.name.toLowerCase().trim();
        if (seen.has(k)) return false;
        seen.add(k);
        return true;
      });
  };

  const topPos = dedupe(posFeatures).slice(0, 6);
  const topNeg = dedupe(negFeatures).slice(0, 6);

  // Fall back to model-level summary strings when feature list is sparse
  const posDisplay = topPos.length > 0
    ? topPos
    : [...new Set(posFromModel)].slice(0, 6).map(t => ({ name: t, count: 0 }));
  const negDisplay = topNeg.length > 0
    ? topNeg
    : [...new Set(negFromModel)].slice(0, 6).map(t => ({ name: t, count: 0 }));

  const listRow = (item, bulletColor) => {
    const text = typeof item === "string" ? item : item.name;
    const count = typeof item === "string" ? 0 : (item.count || 0);
    const bullet = bulletColor === "#16a34a" ? "+" : "−";
    return `<div style="display:flex;gap:7px;margin-bottom:5px;align-items:flex-start">
       <span style="color:${bulletColor};font-weight:700;font-size:13px;flex-shrink:0;line-height:1.4">${bullet}</span>
       <span style="font-size:11px;color:#4a4a6a;text-transform:capitalize;line-height:1.5;flex:1">${text}</span>
       ${count > 0 ? `<span style="font-size:10px;color:${bulletColor};font-family:monospace;flex-shrink:0;opacity:0.75">×${count}</span>` : ""}
     </div>`;
  };

  // ── Expandable model cards ───────────────────────────────────────────────────
  const modelEntries = Object.entries(cData.models || {});
  const modelCards = modelEntries.map(([modelName, mData]) => {
    const mCol = sentColor(mData.overall_sentiment);

    const mPosFeat = (mData.features || [])
      .filter(f => f.sentiment === "positive")
      .sort((a, b) => (b.mention_count || b.count || 0) - (a.mention_count || a.count || 0));
    const mNegFeat = (mData.features || [])
      .filter(f => f.sentiment === "negative")
      .sort((a, b) => (b.mention_count || b.count || 0) - (a.mention_count || a.count || 0));

    const dedupeNames = arr => {
      const seen = new Set();
      return arr.filter(f => {
        const k = f.name.toLowerCase().trim();
        return seen.has(k) ? false : (seen.add(k), true);
      });
    };

    const mPosItems = dedupeNames(mPosFeat).slice(0, 5);
    const mNegItems = dedupeNames(mNegFeat).slice(0, 5);

    const mPosDisplay = mPosItems.length > 0
      ? mPosItems.map(f => ({ name: f.name, count: f.mention_count || f.count || 0 }))
      : (mData.top_positives || []).slice(0, 5).map(t => ({ name: t, count: 0 }));
    const mNegDisplay = mNegItems.length > 0
      ? mNegItems.map(f => ({ name: f.name, count: f.mention_count || f.count || 0 }))
      : (mData.top_negatives || []).slice(0, 5).map(t => ({ name: t, count: 0 }));

    return `
      <details style="border:1px solid #e0e2ee;border-radius:7px;overflow:hidden;background:#fff">
        <summary class="model-summary" style="padding:10px 14px;display:flex;align-items:center;gap:10px;cursor:pointer;user-select:none;list-style:none">
          <span class="chevron" style="font-size:9px;color:${mCol};display:inline-block;flex-shrink:0;transition:transform 0.18s">▶</span>
          <span style="font-size:12px;font-weight:600;color:#1a1a2e;flex:1">${modelName}</span>
          ${sentBadge(mData.overall_sentiment)}
          ${mData.mention_count ? `<span style="font-size:10px;color:#8888a8;font-family:monospace">${mData.mention_count}×</span>` : ""}
        </summary>
        <div style="display:grid;grid-template-columns:1fr 1fr;border-top:1px solid #e8eaf0;background:#f9f9fc">
          <div style="padding:12px 14px;border-right:1px solid #e8eaf0">
            <div style="font-size:8px;color:#16a34a;text-transform:uppercase;letter-spacing:0.15em;font-weight:700;margin-bottom:7px">Praised for</div>
            ${mPosDisplay.length > 0
              ? mPosDisplay.map(t => listRow(t, "#16a34a")).join("")
              : `<div style="font-size:11px;color:#8888a8;font-style:italic">None recorded</div>`}
          </div>
          <div style="padding:12px 14px">
            <div style="font-size:8px;color:#dc2626;text-transform:uppercase;letter-spacing:0.15em;font-weight:700;margin-bottom:7px">Criticized for</div>
            ${mNegDisplay.length > 0
              ? mNegDisplay.map(t => listRow(t, "#dc2626")).join("")
              : `<div style="font-size:11px;color:#8888a8;font-style:italic">None recorded</div>`}
          </div>
        </div>
      </details>`;
  }).join("");

  const modelsSection = modelEntries.length > 0 ? `
    <details style="border-top:1px solid #e8eaf0">
      <summary class="models-toggle" style="padding:10px 16px;background:#f0f1f8;display:flex;align-items:center;gap:10px;cursor:pointer;user-select:none;list-style:none">
        <span class="chevron" style="font-size:9px;color:#5865f2;display:inline-block;flex-shrink:0;transition:transform 0.18s">▶</span>
        <span style="font-size:10px;font-weight:700;color:#5865f2;text-transform:uppercase;letter-spacing:0.15em">
          Brands / Models · ${modelEntries.length}
        </span>
      </summary>
      <div style="padding:12px 16px;background:#f4f5f9;display:flex;flex-direction:column;gap:8px">
        ${modelCards}
      </div>
    </details>` : "";

  return `
    <div style="border:1px solid #d4d6e3;border-radius:9px;margin-bottom:14px;overflow:hidden;page-break-inside:avoid">
      <!-- Brand header -->
      <div style="display:flex;align-items:center;gap:12px;padding:11px 16px;background:#fff;border-bottom:1px solid #e8eaf0">
        <div style="width:3px;height:22px;background:${col};border-radius:2px;flex-shrink:0"></div>
        <span style="font-size:13px;font-weight:700;color:#1a1a2e;flex:1;text-transform:uppercase;letter-spacing:0.07em">${name}</span>
        ${sentBadge(cData.overall_sentiment)}
        ${cData.mention_count ? `<span style="font-size:10px;color:#8888a8;font-family:monospace;margin-left:4px">${cData.mention_count}×</span>` : ""}
      </div>
      <!-- Praised / Criticized columns (summary across all models) -->
      <div style="display:grid;grid-template-columns:1fr 1fr;background:#f9f9fc">
        <div style="padding:14px 16px;border-right:1px solid #e8eaf0">
          <div style="font-size:9px;color:#16a34a;text-transform:uppercase;letter-spacing:0.16em;font-weight:700;margin-bottom:9px">Praised for</div>
          ${posDisplay.length > 0
            ? posDisplay.map(t => listRow(t, "#16a34a")).join("")
            : `<div style="font-size:11px;color:#8888a8;font-style:italic">No positive feedback recorded</div>`}
        </div>
        <div style="padding:14px 16px">
          <div style="font-size:9px;color:#dc2626;text-transform:uppercase;letter-spacing:0.16em;font-weight:700;margin-bottom:9px">Criticized for</div>
          ${negDisplay.length > 0
            ? negDisplay.map(t => listRow(t, "#dc2626")).join("")
            : `<div style="font-size:11px;color:#8888a8;font-style:italic">No negative feedback recorded</div>`}
        </div>
      </div>
      <!-- Expandable models section -->
      ${modelsSection}
    </div>`;
}

// ── Merge brand_analysis across all videos ────────────────────────────────────

function combineBrands(analyzed) {
  const acc = {}; // company → { scoreSum, wTotal, mention_count, models }
  const sentScore = s => s === "positive" ? 0.8 : s === "negative" ? 0.2 : 0.5;

  for (const a of analyzed) {
    const ba = a.aggregated?.brand_analysis;
    if (!ba) continue;

    for (const [company, cData] of Object.entries(ba)) {
      const cn = company.trim();
      if (!acc[cn]) acc[cn] = { scoreSum: 0, wTotal: 0, mention_count: 0, models: {} };
      const cc = acc[cn];
      const cw = cData.mention_count || 1;
      cc.scoreSum += (cData.overall_score ?? sentScore(cData.overall_sentiment)) * cw;
      cc.wTotal   += cw;
      cc.mention_count += cData.mention_count || 0;

      for (const [modelName, mData] of Object.entries(cData.models || {})) {
        const mn = modelName.trim();
        if (!cc.models[mn]) cc.models[mn] = { scoreSum: 0, wTotal: 0, mention_count: 0, features: {}, top_positives: [], top_negatives: [] };
        const cm = cc.models[mn];
        const mw = mData.mention_count || 1;
        cm.scoreSum += (mData.overall_score ?? sentScore(mData.overall_sentiment)) * mw;
        cm.wTotal   += mw;
        cm.mention_count += mData.mention_count || 0;

        for (const f of mData.features || []) {
          const key = f.name.toLowerCase().trim();
          if (!cm.features[key]) cm.features[key] = { name: f.name, count: 0, scoreSum: 0, n: 0 };
          const cf = cm.features[key];
          cf.count    += f.mention_count || f.count || 1;
          cf.scoreSum += f.score ?? (f.sentiment === "positive" ? 0.8 : f.sentiment === "negative" ? 0.2 : 0.5);
          cf.n++;
        }
        if (mData.top_positives?.length) cm.top_positives.push(...mData.top_positives);
        if (mData.top_negatives?.length) cm.top_negatives.push(...mData.top_negatives);
      }
    }
  }

  // Finalise: compute sentiments, convert feature maps → arrays
  const result = {};
  for (const [cn, cc] of Object.entries(acc)) {
    const avgScore = cc.wTotal > 0 ? cc.scoreSum / cc.wTotal : 0.5;
    const models = {};

    for (const [mn, cm] of Object.entries(cc.models)) {
      const mAvg = cm.wTotal > 0 ? cm.scoreSum / cm.wTotal : 0.5;
      models[mn] = {
        overall_sentiment: mAvg >= 0.6 ? "positive" : mAvg <= 0.4 ? "negative" : "neutral",
        overall_score: mAvg,
        mention_count: cm.mention_count,
        top_positives: [...new Set(cm.top_positives)].slice(0, 5),
        top_negatives: [...new Set(cm.top_negatives)].slice(0, 5),
        features: Object.values(cm.features).map(f => {
          const fAvg = f.n > 0 ? f.scoreSum / f.n : 0.5;
          return {
            name: f.name,
            sentiment: fAvg >= 0.6 ? "positive" : fAvg <= 0.4 ? "negative" : "neutral",
            mention_count: f.count,
            count: f.count,
          };
        }).sort((a, b) => b.count - a.count),
      };
    }

    result[cn] = {
      overall_sentiment: avgScore >= 0.6 ? "positive" : avgScore <= 0.4 ? "negative" : "neutral",
      overall_score: avgScore,
      mention_count: cc.mention_count,
      models,
    };
  }

  // Sort by total mention_count descending
  return Object.entries(result).sort((a, b) => b[1].mention_count - a[1].mention_count);
}

// ── Brand Comparison section (interactive HTML + embedded JS) ────────────────

function generateComparisonSectionHtml(entries) {
  if (entries.length < 2) return "";

  const companiesData = Object.fromEntries(entries);

  // Build flat model lookup: "Company › Model" → { ...modelData, _company, _model }
  const modelEntries = [];
  for (const [company, cData] of entries) {
    for (const [modelName, mData] of Object.entries(cData.models || {})) {
      modelEntries.push([
        company + " › " + modelName,
        { ...mData, _company: company, _model: modelName },
      ]);
    }
  }
  modelEntries.sort((a, b) => {
    const ca = a[1]._company, cb = b[1]._company;
    if (ca < cb) return -1;
    if (ca > cb) return 1;
    return (b[1].mention_count || 0) - (a[1].mention_count || 0);
  });

  const allData = { companies: companiesData, models: Object.fromEntries(modelEntries) };
  const dataJson = JSON.stringify(allData).replace(/<\//g, "<\\/");

  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  // Company dropdown options
  const coOptions = entries
    .map(([name]) => `<option value="${esc(name)}">${esc(name)}</option>`)
    .join("");

  // Brand/Model dropdown options grouped by company (uses <optgroup>)
  let moOptions = "";
  let curGroup = null;
  for (const [key, mData] of modelEntries) {
    if (mData._company !== curGroup) {
      if (curGroup !== null) moOptions += "</optgroup>";
      moOptions += `<optgroup label="${esc(mData._company)}">`;
      curGroup = mData._company;
    }
    moOptions += `<option value="${esc(key)}">${esc(mData._model)}</option>`;
  }
  if (curGroup !== null) moOptions += "</optgroup>";

  const selSt = "padding:8px 12px;border:1px solid #d4d6e3;border-radius:7px;font-size:13px;color:#1a1a2e;background:#fff;min-width:210px;cursor:pointer;outline:none";
  const actSt = "padding:7px 18px;font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;border-radius:6px;cursor:pointer;border:1px solid #5865f2;background:#5865f2;color:#fff";
  const inaSt = "padding:7px 18px;font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;border-radius:6px;cursor:pointer;border:1px solid #d4d6e3;background:transparent;color:#8888a8";

  const actionBtns = `<button data-cmp="run" style="background:#5865f2;color:#fff;border:none;border-radius:7px;padding:10px 22px;font-size:12px;font-weight:700;cursor:pointer;letter-spacing:0.08em;text-transform:uppercase;align-self:flex-end">Compare</button>
      <button data-cmp="clear" style="background:transparent;color:#8888a8;border:1px solid #d4d6e3;border-radius:7px;padding:10px 16px;font-size:12px;font-weight:600;cursor:pointer;align-self:flex-end">Clear</button>`;

  return `
  <div class="no-print" style="margin-bottom:24px;background:#fff;border:1px solid #d4d6e3;border-radius:10px;padding:24px 28px;box-shadow:0 2px 8px rgba(0,0,0,0.04)">
    <div style="font-size:12px;font-weight:700;color:#5865f2;letter-spacing:0.22em;text-transform:uppercase;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid #e8eaf0">Company &amp; Brand Comparison</div>

    <!-- Mode toggle -->
    <div style="display:flex;gap:6px;margin-bottom:18px">
      <button id="cmp-mode-co" data-cmp-mode="company" style="${actSt}">Company</button>
      <button id="cmp-mode-mo" data-cmp-mode="model" style="${inaSt}">Brand / Model</button>
    </div>

    <!-- Company selects (visible by default) -->
    <div id="cmp-co-row" style="display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap">
      <div>
        <label style="font-size:10px;color:#8888a8;text-transform:uppercase;letter-spacing:0.15em;font-weight:700;display:block;margin-bottom:6px">Company A</label>
        <select id="cmp-co-a" style="${selSt}"><option value="">Select company…</option>${coOptions}</select>
      </div>
      <div style="font-size:18px;font-weight:800;color:#5865f2;padding-bottom:9px">vs</div>
      <div>
        <label style="font-size:10px;color:#8888a8;text-transform:uppercase;letter-spacing:0.15em;font-weight:700;display:block;margin-bottom:6px">Company B</label>
        <select id="cmp-co-b" style="${selSt}"><option value="">Select company…</option>${coOptions}</select>
      </div>
      ${actionBtns}
    </div>

    <!-- Brand / Model selects (hidden until mode toggled) -->
    <div id="cmp-mo-row" style="display:none;gap:16px;align-items:flex-end;flex-wrap:wrap">
      <div>
        <label style="font-size:10px;color:#8888a8;text-transform:uppercase;letter-spacing:0.15em;font-weight:700;display:block;margin-bottom:6px">Brand / Model A</label>
        <select id="cmp-mo-a" style="${selSt}"><option value="">Select brand / model…</option>${moOptions}</select>
      </div>
      <div style="font-size:18px;font-weight:800;color:#5865f2;padding-bottom:9px">vs</div>
      <div>
        <label style="font-size:10px;color:#8888a8;text-transform:uppercase;letter-spacing:0.15em;font-weight:700;display:block;margin-bottom:6px">Brand / Model B</label>
        <select id="cmp-mo-b" style="${selSt}"><option value="">Select brand / model…</option>${moOptions}</select>
      </div>
      ${actionBtns}
    </div>
  </div>

  <div id="cmp-result" style="margin-bottom:44px"></div>

  <div class="no-print" style="margin-bottom:24px;background:#fff;border:1px solid #d4d6e3;border-radius:10px;padding:24px 28px;box-shadow:0 2px 8px rgba(0,0,0,0.04)">
    <div style="font-size:12px;font-weight:700;color:#5865f2;letter-spacing:0.22em;text-transform:uppercase;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid #e8eaf0">Company &amp; Brand Deep Dive</div>

    <div style="display:flex;gap:6px;margin-bottom:18px">
      <button id="dive-mode-co" data-dive-mode="company" style="${actSt}">Company</button>
      <button id="dive-mode-mo" data-dive-mode="model" style="${inaSt}">Brand / Model</button>
    </div>

    <div id="dive-co-row" style="display:flex;gap:16px;align-items:flex-end;flex-wrap:wrap">
      <div>
        <label style="font-size:10px;color:#8888a8;text-transform:uppercase;letter-spacing:0.15em;font-weight:700;display:block;margin-bottom:6px">Company</label>
        <select id="dive-co-sel" style="${selSt}"><option value="">Select company…</option>${coOptions}</select>
      </div>
      <button data-dive="run" style="background:#5865f2;color:#fff;border:none;border-radius:7px;padding:10px 22px;font-size:12px;font-weight:700;cursor:pointer;letter-spacing:0.08em;text-transform:uppercase;align-self:flex-end">Analyse</button>
      <button data-dive="clear" style="background:#fff;color:#8888a8;border:1px solid #d4d6e3;border-radius:7px;padding:10px 18px;font-size:12px;font-weight:700;cursor:pointer;letter-spacing:0.08em;text-transform:uppercase;align-self:flex-end">Clear</button>
    </div>

    <div id="dive-mo-row" style="display:none;gap:16px;align-items:flex-end;flex-wrap:wrap">
      <div>
        <label style="font-size:10px;color:#8888a8;text-transform:uppercase;letter-spacing:0.15em;font-weight:700;display:block;margin-bottom:6px">Brand / Model</label>
        <select id="dive-mo-sel" style="${selSt}"><option value="">Select brand / model…</option>${moOptions}</select>
      </div>
      <button data-dive="run" style="background:#5865f2;color:#fff;border:none;border-radius:7px;padding:10px 22px;font-size:12px;font-weight:700;cursor:pointer;letter-spacing:0.08em;text-transform:uppercase;align-self:flex-end">Analyse</button>
      <button data-dive="clear" style="background:#fff;color:#8888a8;border:1px solid #d4d6e3;border-radius:7px;padding:10px 18px;font-size:12px;font-weight:700;cursor:pointer;letter-spacing:0.08em;text-transform:uppercase;align-self:flex-end">Clear</button>
    </div>
  </div>

  <div id="dive-result" style="margin-bottom:44px"></div>

  <script>
  (function() {
    var DATA;
    try { DATA = ${dataJson}; } catch(e) { DATA = null; }
    var cmpMode = 'company';
    var diveMode = 'company';
    var ACT = '${actSt}';
    var INA = '${inaSt}';
    var cardSeq = 0;
    var recStore = {};

    function mdToHtml(s) {
      return s
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>')
        .replace(/\\n\\n+/g, '\\n\\n')
        .split('\\n\\n').map(function(p) { return '<p style="margin:0 0 10px">' + p.replace(/\\n/g, '<br>') + '</p>'; }).join('');
    }

    function sc(s) { return s === 'positive' ? '#16a34a' : s === 'negative' ? '#dc2626' : '#5865f2'; }

    function badge(s) {
      var c = sc(s || 'neutral');
      return '<span style="display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:3px;border:1px solid ' + c + '44;background:' + c + '11;color:' + c + ';font-size:10px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;font-family:monospace"><span style="width:7px;height:7px;border-radius:50%;background:' + c + ';display:inline-block;flex-shrink:0"></span>' + (s || 'neutral') + '</span>';
    }

    function extractFeats(data, isModel) {
      var pos = [], neg = [], sp = {}, sn = {};
      if (isModel) {
        var feats = data.features || [];
        for (var i = 0; i < feats.length; i++) {
          var f = feats[i], k = (f.name || '').toLowerCase().trim();
          if (f.sentiment === 'positive' && !sp[k]) { sp[k] = 1; pos.push({name: f.name, count: f.mention_count || f.count || 1}); }
          else if (f.sentiment === 'negative' && !sn[k]) { sn[k] = 1; neg.push({name: f.name, count: f.mention_count || f.count || 1}); }
        }
        if (!pos.length && data.top_positives) pos = (data.top_positives || []).slice(0,8).map(function(t){return {name:t,count:0};});
        if (!neg.length && data.top_negatives) neg = (data.top_negatives || []).slice(0,8).map(function(t){return {name:t,count:0};});
      } else {
        var models = data.models || {};
        for (var mn in models) {
          var mFeats = models[mn].features || [];
          for (var fi = 0; fi < mFeats.length; fi++) {
            var ff = mFeats[fi], fk = (ff.name || '').toLowerCase().trim();
            if (ff.sentiment === 'positive' && !sp[fk]) { sp[fk] = 1; pos.push({name: ff.name, count: ff.mention_count || ff.count || 1}); }
            else if (ff.sentiment === 'negative' && !sn[fk]) { sn[fk] = 1; neg.push({name: ff.name, count: ff.mention_count || ff.count || 1}); }
          }
        }
      }
      pos.sort(function(a,b){return b.count-a.count;}); neg.sort(function(a,b){return b.count-a.count;});
      return {pos: pos.slice(0,8), neg: neg.slice(0,8)};
    }

    function fList(arr, col) {
      if (!arr || !arr.length) return '<div style="font-size:11px;color:#8888a8;font-style:italic">None recorded</div>';
      var bul = col === '#16a34a' ? '+' : '&#8722;';
      return arr.map(function(f) {
        return '<div style="display:flex;gap:7px;margin-bottom:6px;align-items:flex-start"><span style="color:' + col + ';font-weight:700;font-size:13px;flex-shrink:0;line-height:1.4">' + bul + '</span><span style="font-size:12px;color:#4a4a6a;text-transform:capitalize;line-height:1.5;flex:1">' + f.name + '</span>' + (f.count > 0 ? '<span style="font-size:10px;color:' + col + ';font-family:monospace;flex-shrink:0;opacity:0.75">\xd7' + f.count + '</span>' : '') + '</div>';
      }).join('');
    }

    function middleCell(d, isModel, isLeft) {
      var bdr = isLeft ? 'border-right:1px solid #d4d6e3;' : '';
      var lbl, inner;
      if (isModel) {
        lbl = 'Parent Company';
        var co = d._company || 'Unknown';
        var coData = DATA && DATA.companies ? DATA.companies[co] : null;
        var coSent = coData ? coData.overall_sentiment : 'neutral';
        var c = sc(coSent);
        inner = '<div style="display:flex;align-items:center;gap:8px">' +
          '<div style="width:3px;height:16px;background:' + c + ';border-radius:2px"></div>' +
          '<span style="font-size:13px;font-weight:700;color:#1a1a2e;text-transform:uppercase;letter-spacing:0.05em">' + co + '</span>' +
          badge(coSent) + '</div>' +
          (coData && coData.mention_count ? '<div style="font-size:11px;color:#8888a8;font-family:monospace;margin-top:4px;padding-left:11px">' + coData.mention_count + ' company-level mentions</div>' : '');
      } else {
        lbl = 'Models / Variants';
        var keys = Object.keys(d.models || {});
        if (!keys.length) {
          inner = '<span style="font-size:11px;color:#8888a8;font-style:italic">No models recorded</span>';
        } else {
          inner = keys.slice(0, 8).map(function(mn) {
            var m = d.models[mn], mc = sc(m.overall_sentiment);
            return '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px"><span style="width:6px;height:6px;border-radius:50%;background:' + mc + ';flex-shrink:0"></span><span style="font-size:12px;color:#1a1a2e;font-weight:600">' + mn + '</span>' + (m.mention_count ? '<span style="font-size:10px;color:#8888a8;font-family:monospace">\xd7' + m.mention_count + '</span>' : '') + '</div>';
          }).join('');
          if (keys.length > 8) inner += '<div style="font-size:10px;color:#8888a8;margin-top:4px">+' + (keys.length - 8) + ' more</div>';
        }
      }
      return '<div style="padding:14px 22px;' + bdr + '">' +
        '<div style="font-size:9px;color:#5865f2;text-transform:uppercase;letter-spacing:0.16em;font-weight:700;margin-bottom:8px">' + lbl + '</div>' +
        inner + '</div>';
    }

    function buildCard(na, nb, a, b, isModel) {
      var rid = 'rc' + (++cardSeq);
      var aF = extractFeats(a, isModel), bF = extractFeats(b, isModel);
      var aC = sc(a.overall_sentiment), bC = sc(b.overall_sentiment);
      var aLabel = isModel ? (a._model || na) : na;
      var bLabel = isModel ? (b._model || nb) : nb;
      var aScore = a.overall_score != null ? Math.round(a.overall_score * 100) : null;
      var bScore = b.overall_score != null ? Math.round(b.overall_score * 100) : null;
      var modeTag = isModel ? 'Brand / Model Comparison' : 'Company Comparison';

      recStore[rid] = {
        name_a: aLabel, name_b: bLabel,
        mode: isModel ? 'model' : 'company',
        sentiment_a: a.overall_sentiment || 'neutral',
        sentiment_b: b.overall_sentiment || 'neutral',
        positives_a: aF.pos.map(function(f){return f.name;}),
        negatives_a: aF.neg.map(function(f){return f.name;}),
        positives_b: bF.pos.map(function(f){return f.name;}),
        negatives_b: bF.neg.map(function(f){return f.name;}),
        company_a: (isModel && a._company) ? a._company : null,
        company_b: (isModel && b._company) ? b._company : null
      };

      return '<div style="border:1px solid #d4d6e3;border-radius:10px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.04)">' +

        '<div style="padding:14px 22px;background:#f0f1f8;border-bottom:1px solid #d4d6e3;display:flex;align-items:center;gap:12px;flex-wrap:wrap">' +
          '<span style="font-size:12px;font-weight:700;color:#5865f2;letter-spacing:0.2em;text-transform:uppercase">' + aLabel + ' vs ' + bLabel + '</span>' +
          '<span style="font-size:10px;color:#8888a8;font-family:monospace;margin-left:auto;padding:2px 9px;border:1px solid #e0e2ee;border-radius:4px;background:#fff">' + modeTag + '</span>' +
        '</div>' +

        '<div style="display:grid;grid-template-columns:1fr 1fr">' +
          '<div style="padding:18px 22px;background:' + aC + '0d;border-right:1px solid #d4d6e3;border-bottom:1px solid #d4d6e3">' +
            '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:5px">' +
              '<div style="width:3px;height:22px;background:' + aC + ';border-radius:2px;flex-shrink:0"></div>' +
              '<span style="font-size:15px;font-weight:800;color:#1a1a2e;text-transform:uppercase;letter-spacing:0.07em">' + aLabel + '</span>' +
              badge(a.overall_sentiment) +
            '</div>' +
            '<div style="font-size:11px;color:#8888a8;font-family:monospace;padding-left:11px">' +
              (a.mention_count ? a.mention_count + ' mentions' : '') +
              (aScore !== null ? ' \xb7 score: ' + aScore + '/100' : '') +
              (isModel && a._company ? ' \xb7 ' + a._company : '') +
            '</div>' +
          '</div>' +
          '<div style="padding:18px 22px;background:' + bC + '0d;border-bottom:1px solid #d4d6e3">' +
            '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:5px">' +
              '<div style="width:3px;height:22px;background:' + bC + ';border-radius:2px;flex-shrink:0"></div>' +
              '<span style="font-size:15px;font-weight:800;color:#1a1a2e;text-transform:uppercase;letter-spacing:0.07em">' + bLabel + '</span>' +
              badge(b.overall_sentiment) +
            '</div>' +
            '<div style="font-size:11px;color:#8888a8;font-family:monospace;padding-left:11px">' +
              (b.mention_count ? b.mention_count + ' mentions' : '') +
              (bScore !== null ? ' \xb7 score: ' + bScore + '/100' : '') +
              (isModel && b._company ? ' \xb7 ' + b._company : '') +
            '</div>' +
          '</div>' +
        '</div>' +

        '<div style="display:grid;grid-template-columns:1fr 1fr;background:#f9f9fc;border-bottom:1px solid #d4d6e3">' +
          middleCell(a, isModel, true) +
          middleCell(b, isModel, false) +
        '</div>' +

        '<div>' +
          '<div style="padding:9px 22px;background:#dcfce7;border-bottom:1px solid #bbf7d0">' +
            '<span style="font-size:9px;color:#16a34a;text-transform:uppercase;letter-spacing:0.18em;font-weight:700">Praised For</span>' +
          '</div>' +
          '<div style="display:grid;grid-template-columns:1fr 1fr;background:#f0fdf4">' +
            '<div style="padding:16px 22px;border-right:1px solid #d4d6e3">' + fList(aF.pos, '#16a34a') + '</div>' +
            '<div style="padding:16px 22px">' + fList(bF.pos, '#16a34a') + '</div>' +
          '</div>' +
        '</div>' +

        '<div style="border-top:1px solid #d4d6e3">' +
          '<div style="padding:9px 22px;background:#fee2e2;border-bottom:1px solid #fecaca">' +
            '<span style="font-size:9px;color:#dc2626;text-transform:uppercase;letter-spacing:0.18em;font-weight:700">Criticized For</span>' +
          '</div>' +
          '<div style="display:grid;grid-template-columns:1fr 1fr;background:#fff5f5">' +
            '<div style="padding:16px 22px;border-right:1px solid #d4d6e3">' + fList(aF.neg, '#dc2626') + '</div>' +
            '<div style="padding:16px 22px">' + fList(bF.neg, '#dc2626') + '</div>' +
          '</div>' +
        '</div>' +

        '<div style="border-top:2px solid #e0e2ee;background:#fafbff">' +
          '<div style="padding:12px 22px;display:flex;align-items:center;gap:14px;flex-wrap:wrap">' +
            '<span style="font-size:9px;font-weight:700;color:#5865f2;text-transform:uppercase;letter-spacing:0.18em">Recommendations</span>' +
            '<button id="rec-btn-' + rid + '" data-rec="' + rid + '" style="background:#5865f2;color:#fff;border:none;border-radius:7px;padding:7px 18px;font-size:11px;font-weight:700;cursor:pointer;letter-spacing:0.07em;text-transform:uppercase">Get Recommendations</button>' +
          '</div>' +
          '<div id="rec-out-' + rid + '" style="display:none;padding:0 22px 18px"></div>' +
        '</div>' +

      '</div>';
    }

    function getRecommendation(rid) {
      var payload = recStore[rid];
      var btn = document.getElementById('rec-btn-' + rid);
      var out = document.getElementById('rec-out-' + rid);
      if (!payload || !btn || !out) return;

      btn.disabled = true;
      btn.textContent = 'Generating…';
      out.style.display = 'block';
      out.innerHTML = '<div style="padding:10px 0 4px;color:#8888a8;font-size:12px;font-style:italic">Analysing comparison data…</div>';

      fetch('http://localhost:8000/api/recommend', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      }).then(function(resp) {
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        var reader = resp.body.getReader();
        var dec = new TextDecoder();
        var text = '';
        out.innerHTML = '<div id="rec-txt-' + rid + '" style="padding:10px 0;font-size:13px;line-height:1.8;color:#1a1a2e;font-family:sans-serif"></div>';
        var el = document.getElementById('rec-txt-' + rid);
        function pump() {
          return reader.read().then(function(r) {
            if (r.done) { btn.textContent = 'Regenerate'; btn.disabled = false; return; }
            text += dec.decode(r.value, {stream: true});
            el.innerHTML = mdToHtml(text);
            return pump();
          });
        }
        return pump();
      }).catch(function(err) {
        out.innerHTML = '<div style="padding:10px 0;color:#dc2626;font-size:12px">Failed: ' + err.message + '. Ensure the backend is running on port 8000.</div>';
        btn.textContent = 'Retry';
        btn.disabled = false;
      });
    }

    function runComparison() {
      var isModel = (cmpMode === 'model');
      var na = document.getElementById(isModel ? 'cmp-mo-a' : 'cmp-co-a').value;
      var nb = document.getElementById(isModel ? 'cmp-mo-b' : 'cmp-co-b').value;
      var el = document.getElementById('cmp-result');
      var noun = isModel ? 'brands / models' : 'companies';
      if (!na || !nb) { el.innerHTML = '<div style="padding:16px 20px;background:#fff;border:1px solid #d4d6e3;border-radius:10px;color:#dc2626;font-size:13px">Please select two ' + noun + ' to compare.</div>'; return; }
      if (na === nb) { el.innerHTML = '<div style="padding:16px 20px;background:#fff;border:1px solid #d4d6e3;border-radius:10px;color:#dc2626;font-size:13px">Please select two different ' + noun + '.</div>'; return; }
      if (!DATA) { el.innerHTML = '<div style="padding:16px;color:#dc2626;font-size:13px">Data unavailable.</div>'; return; }
      var a = isModel ? DATA.models[na] : DATA.companies[na];
      var b = isModel ? DATA.models[nb] : DATA.companies[nb];
      if (!a || !b) { el.innerHTML = '<div style="padding:16px;color:#dc2626;font-size:13px">Data not found for selected items.</div>'; return; }
      el.innerHTML = buildCard(na, nb, a, b, isModel);
      var rid = 'rc' + cardSeq;
      if (recStore[rid] && document.getElementById('rec-out-' + rid)) {
        getRecommendation(rid);
      }
    }

    function clearComparison() {
      var el = document.getElementById('cmp-result');
      if (el) el.innerHTML = '';
      ['cmp-co-a','cmp-co-b','cmp-mo-a','cmp-mo-b'].forEach(function(id) {
        var s = document.getElementById(id);
        if (s) s.value = '';
      });
    }

    function clearDive() {
      var el = document.getElementById('dive-result');
      if (el) el.innerHTML = '';
      ['dive-co-sel','dive-mo-sel'].forEach(function(id) {
        var s = document.getElementById(id);
        if (s) s.value = '';
      });
    }

    function setDiveMode(m) {
      diveMode = m;
      var co = document.getElementById('dive-mode-co');
      var mo = document.getElementById('dive-mode-mo');
      var coRow = document.getElementById('dive-co-row');
      var moRow = document.getElementById('dive-mo-row');
      if (co) co.setAttribute('style', m === 'company' ? ACT : INA);
      if (mo) mo.setAttribute('style', m === 'model' ? ACT : INA);
      if (coRow) coRow.style.display = m === 'company' ? 'flex' : 'none';
      if (moRow) moRow.style.display = m === 'model' ? 'flex' : 'none';
      clearDive();
    }

    function runDiveAnalysis() {
      var isModel = (diveMode === 'model');
      var sel = document.getElementById(isModel ? 'dive-mo-sel' : 'dive-co-sel');
      var key = sel ? sel.value : '';
      var el = document.getElementById('dive-result');
      if (!key) { el.innerHTML = '<div style="padding:16px 20px;background:#fff;border:1px solid #d4d6e3;border-radius:10px;color:#dc2626;font-size:13px">Please select a ' + (isModel ? 'brand / model' : 'company') + '.</div>'; return; }
      if (!DATA) { el.innerHTML = '<div style="padding:16px;color:#dc2626;font-size:13px">Data unavailable.</div>'; return; }
      var d = isModel ? DATA.models[key] : DATA.companies[key];
      if (!d) { el.innerHTML = '<div style="padding:16px;color:#dc2626;font-size:13px">Data not found.</div>'; return; }
      var aF = extractFeats(d, isModel);
      var label = isModel ? (d._model || key) : key;

      // Pass all other brands so the LLM can identify real competitors
      var allBrands = [];
      var sourceMap = isModel ? DATA.models : DATA.companies;
      if (sourceMap) {
        for (var ck in sourceMap) {
          if (ck !== key) {
            var bd = sourceMap[ck];
            var bf = extractFeats(bd, isModel);
            allBrands.push({
              name: isModel ? (bd._model || ck) : ck,
              sentiment: bd.overall_sentiment || 'neutral',
              mention_count: bd.mention_count || 0,
              positives: bf.pos.slice(0, 5).map(function(f){return f.name;}),
              negatives: bf.neg.slice(0, 5).map(function(f){return f.name;})
            });
          }
        }
      }

      var payload = {
        name: label,
        mode: isModel ? 'model' : 'company',
        sentiment: d.overall_sentiment || 'neutral',
        positives: aF.pos.map(function(f){return f.name;}),
        negatives: aF.neg.map(function(f){return f.name;}),
        company: (isModel && d._company) ? d._company : null,
        all_brands: allBrands
      };
      el.innerHTML = '<div style="background:#fff;border:1px solid #d4d6e3;border-radius:10px;overflow:hidden">' +
        '<div style="padding:14px 22px;background:#f0f1f8;border-bottom:1px solid #d4d6e3;display:flex;align-items:center;gap:10px">' +
          '<span style="font-size:13px;font-weight:700;color:#1a1a2e;text-transform:uppercase;letter-spacing:0.07em">' + label + '</span>' +
          badge(d.overall_sentiment) +
        '</div>' +
        '<div id="dive-out" style="padding:18px 22px;font-size:13px;line-height:1.8;color:#8888a8;font-style:italic">Generating analysis…</div>' +
      '</div>';
      fetch('http://localhost:8000/api/brand-insight', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      }).then(function(resp) {
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        var reader = resp.body.getReader();
        var dec = new TextDecoder();
        var text = '';
        var out = document.getElementById('dive-out');
        out.style.fontStyle = 'normal';
        out.style.color = '#1a1a2e';
        out.innerHTML = '';
        function pump() {
          return reader.read().then(function(r) {
            if (r.done) return;
            text += dec.decode(r.value, {stream: true});
            out.innerHTML = mdToHtml(text);
            return pump();
          });
        }
        return pump();
      }).catch(function(err) {
        var out = document.getElementById('dive-out');
        if (out) out.innerHTML = '<span style="color:#dc2626">Failed: ' + err.message + '. Ensure the backend is running on port 8000.</span>';
      });
    }

    function setCmpMode(m) {
      cmpMode = m;
      var co = document.getElementById('cmp-mode-co');
      var mo = document.getElementById('cmp-mode-mo');
      var coRow = document.getElementById('cmp-co-row');
      var moRow = document.getElementById('cmp-mo-row');
      if (co) co.setAttribute('style', m === 'company' ? ACT : INA);
      if (mo) mo.setAttribute('style', m === 'model' ? ACT : INA);
      if (coRow) coRow.style.display = m === 'company' ? 'flex' : 'none';
      if (moRow) moRow.style.display = m === 'model' ? 'flex' : 'none';
      clearComparison();
    }

    document.addEventListener('click', function(e) {
      var t = e.target;
      var btn = t.closest ? t.closest('[data-cmp],[data-cmp-mode],[data-rec],[data-dive],[data-dive-mode]') : null;
      if (!btn && t.getAttribute) {
        if (t.getAttribute('data-cmp') || t.getAttribute('data-cmp-mode') || t.getAttribute('data-rec') || t.getAttribute('data-dive') || t.getAttribute('data-dive-mode')) btn = t;
      }
      if (!btn) return;
      var act  = btn.getAttribute('data-cmp');
      var mode = btn.getAttribute('data-cmp-mode');
      var rid  = btn.getAttribute('data-rec');
      var dive = btn.getAttribute('data-dive');
      var dm   = btn.getAttribute('data-dive-mode');
      if (act === 'run') runComparison();
      else if (act === 'clear') clearComparison();
      else if (mode) setCmpMode(mode);
      else if (rid) getRecommendation(rid);
      else if (dive === 'run') runDiveAnalysis();
      else if (dive === 'clear') clearDive();
      else if (dm) setDiveMode(dm);
    });
  })();
  </script>`;
}

// ── Report HTML generator ──────────────────────────────────────────────────────

function generateReport(analyses, allVideos) {
  const date = new Date().toLocaleString("en-US", {
    year: "numeric", month: "long", day: "numeric",
    hour: "2-digit", minute: "2-digit",
  });

  const analyzed = analyses.filter(a => a?.aggregated);

  // Weighted aggregate sentiment across all videos
  let sumPos = 0, sumNeu = 0, sumNeg = 0, wTotal = 0;
  for (const a of analyzed) {
    const agg = a.aggregated;
    const w = agg.total_chunks_analyzed || 1;
    sumPos += (agg.sentiment_distribution?.positive ?? 0) * w;
    sumNeu += (agg.sentiment_distribution?.neutral  ?? 0) * w;
    sumNeg += (agg.sentiment_distribution?.negative ?? 0) * w;
    wTotal += w;
  }
  const W = wTotal || 1;
  const aggPos = sumPos / W;
  const aggNeu = sumNeu / W;
  const aggNeg = sumNeg / W;
  const overallSent = aggPos >= 60 ? "positive" : aggNeg >= 60 ? "negative" : "neutral";
  const totalChunks = analyzed.reduce((s, a) => s + (a.aggregated.total_chunks_analyzed || 0), 0);

  // Cross-video brand aggregation
  const brandMap = {};
  for (const a of analyzed) {
    const agg = a.aggregated;
    if (agg.brand_analysis) {
      for (const [company, cData] of Object.entries(agg.brand_analysis)) {
        const nm = company.trim();
        if (!brandMap[nm]) brandMap[nm] = { count: 0, scoreSum: 0, n: 0 };
        brandMap[nm].count += cData.mention_count || 1;
        brandMap[nm].scoreSum += cData.overall_score ?? 0.5;
        brandMap[nm].n++;
      }
    } else if (agg.brands?.length) {
      for (const b of agg.brands) {
        const nm = b.name.trim();
        if (!brandMap[nm]) brandMap[nm] = { count: 0, scoreSum: 0.5, n: 1 };
        brandMap[nm].count += b.count || 1;
      }
    }
  }

  const uniqueBrands = Object.keys(brandMap).length;

  // ── Cross-video brand aggregation with sentiment breakdown ───────────────────
  const combinedBrandEntries = combineBrands(analyzed);

  const topBrands = combinedBrandEntries
    .map(([name, cData]) => {
      let posCount = 0, neuCount = 0, negCount = 0;
      for (const mData of Object.values(cData.models || {})) {
        for (const f of mData.features || []) {
          const c = f.mention_count || f.count || 1;
          if (f.sentiment === "positive") posCount += c;
          else if (f.sentiment === "negative") negCount += c;
          else neuCount += c;
        }
      }
      const total = posCount + neuCount + negCount || 1;
      return {
        label: name,
        value: cData.mention_count || 0,
        posRatio: posCount / total,
        neuRatio: neuCount / total,
        negRatio: negCount / total,
        posCount,
        neuCount,
        negCount,
      };
    })
    .sort((a, b) => b.value - a.value)
    .slice(0, 14);

  // ── Section: Executive Summary ──────────────────────────────────────────────
  const execSummary = `
    <section style="margin-bottom:44px">
      <h2 style="margin:0 0 20px;font-size:12px;font-weight:700;color:#5865f2;letter-spacing:0.22em;text-transform:uppercase;border-bottom:1px solid #e8eaf0;padding-bottom:10px">Executive Summary</h2>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:28px">
        ${[
          ["Videos Indexed",  allVideos.length,  "#1a1a2e"],
          ["Videos Analyzed", analyzed.length,   "#5865f2"],
          ["Unique Brands",   uniqueBrands,       "#5865f2"],
          ["Data Chunks",     totalChunks,        "#5865f2"],
        ].map(([label, val, col]) => `
          <div style="background:#fff;border:1px solid #d4d6e3;border-radius:10px;padding:20px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,0.04)">
            <div style="font-size:9px;color:#8888a8;text-transform:uppercase;letter-spacing:0.18em;font-weight:700;margin-bottom:10px">${label}</div>
            <div style="font-size:34px;font-weight:800;color:${col};font-family:monospace;line-height:1">${val}</div>
          </div>
        `).join("")}
      </div>
      <div style="background:#fff;border:1px solid #d4d6e3;border-radius:10px;padding:28px;display:flex;align-items:center;gap:36px;box-shadow:0 2px 8px rgba(0,0,0,0.04)">
        <div style="flex-shrink:0;display:flex;flex-direction:column;align-items:center;gap:10px">
          ${svgDonut(aggPos, aggNeu, aggNeg, 140)}
        </div>
        <div style="flex:1">
          <div style="font-size:9px;color:#8888a8;text-transform:uppercase;letter-spacing:0.18em;font-weight:700;margin-bottom:18px">Combined Sentiment Distribution</div>
          <div style="margin-bottom:10px">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
              <span style="width:9px;height:9px;border-radius:50%;background:#16a34a;flex-shrink:0"></span>
              <span style="font-size:12px;color:#4a4a6a;font-family:monospace;width:64px">Positive</span>
              ${pctBar(aggPos, "#16a34a", 260)}
            </div>
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
              <span style="width:9px;height:9px;border-radius:50%;background:#5865f2;flex-shrink:0"></span>
              <span style="font-size:12px;color:#4a4a6a;font-family:monospace;width:64px">Neutral</span>
              ${pctBar(aggNeu, "#5865f2", 260)}
            </div>
            <div style="display:flex;align-items:center;gap:10px">
              <span style="width:9px;height:9px;border-radius:50%;background:#dc2626;flex-shrink:0"></span>
              <span style="font-size:12px;color:#4a4a6a;font-family:monospace;width:64px">Negative</span>
              ${pctBar(aggNeg, "#dc2626", 260)}
            </div>
          </div>
        </div>
      </div>
    </section>`;

  // ── Section: Company Leaderboard ─────────────────────────────────────────────
  const brandSection = topBrands.length > 0 ? `
    <div id="bar-tip" class="no-print" style="position:fixed;display:none;background:#1a1a2e;color:#fff;border-radius:9px;padding:12px 16px;font-size:12px;font-family:'Segoe UI',sans-serif;pointer-events:none;z-index:9999;box-shadow:0 6px 24px rgba(0,0,0,0.3);min-width:170px"></div>
    <script>
    function showBTip(e, pos, neu, neg, total) {
      var t = document.getElementById('bar-tip');
      t.innerHTML =
        '<div style="font-size:10px;font-weight:700;color:#8888cc;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:10px">Sentiment Breakdown</div>' +
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">' +
          '<span style="width:10px;height:10px;border-radius:50%;background:#16a34a;flex-shrink:0"></span>' +
          '<span style="color:#d1fae5;flex:1">Positive</span>' +
          '<span style="font-weight:700;color:#16a34a;font-family:monospace">' + pos + '</span>' +
        '</div>' +
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">' +
          '<span style="width:10px;height:10px;border-radius:50%;background:#5865f2;flex-shrink:0"></span>' +
          '<span style="color:#e0e7ff;flex:1">Neutral</span>' +
          '<span style="font-weight:700;color:#818cf8;font-family:monospace">' + neu + '</span>' +
        '</div>' +
        '<div style="display:flex;align-items:center;gap:8px">' +
          '<span style="width:10px;height:10px;border-radius:50%;background:#dc2626;flex-shrink:0"></span>' +
          '<span style="color:#fecaca;flex:1">Negative</span>' +
          '<span style="font-weight:700;color:#ef4444;font-family:monospace">' + neg + '</span>' +
        '</div>';
      var tw = 186, th = 110;
      var lx = e.clientX + 16;
      if (lx + tw > window.innerWidth - 10) lx = e.clientX - tw - 10;
      var ty = e.clientY - 55;
      if (ty < 10) ty = 10;
      if (ty + th > window.innerHeight - 10) ty = window.innerHeight - th - 10;
      t.style.left = lx + 'px';
      t.style.top = ty + 'px';
      t.style.display = 'block';
    }
    function hideBTip() { document.getElementById('bar-tip').style.display = 'none'; }
    </script>
    <section style="margin-bottom:44px">
      <h2 style="margin:0 0 20px;font-size:12px;font-weight:700;color:#5865f2;letter-spacing:0.22em;text-transform:uppercase;border-bottom:1px solid #e8eaf0;padding-bottom:10px">Company Leaderboard · Top ${topBrands.length} by Total Mentions</h2>
      <div style="background:#fff;border:1px solid #d4d6e3;border-radius:10px;padding:28px;box-shadow:0 2px 8px rgba(0,0,0,0.04);overflow-x:auto">
        ${svgHBars(topBrands)}
        <div style="display:flex;gap:24px;margin-top:18px;padding-top:16px;border-top:1px solid #e8eaf0;flex-wrap:wrap">
          ${[["Positive","#16a34a"],["Neutral","#5865f2"],["Negative","#dc2626"]].map(([s,c]) =>
            `<span style="display:flex;align-items:center;gap:7px;font-size:11px;color:#4a4a6a"><span style="width:13px;height:13px;border-radius:3px;background:${c};display:inline-block"></span>${s}</span>`
          ).join("")}
        </div>
      </div>
    </section>` : "";

  // ── Full HTML document ────────────────────────────────────────────────────────
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Automotive Intelligence Report · ${date}</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #f4f5f9; font-family: 'Segoe UI', Inter, Arial, sans-serif; color: #4a4a6a; line-height: 1.6; }
  details > summary { list-style: none; }
  details > summary::-webkit-details-marker { display: none; }
  details > summary::marker { display: none; }
  details[open] > summary .chevron { transform: rotate(90deg); }
  @media print {
    body { background: #fff; }
    .no-print { display: none !important; }
    section, .page-card { page-break-inside: avoid; }
    .header-bar { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    details { display: block; }
    details > div { display: block !important; }
  }
</style>
</head>
<body>
<div style="max-width:960px;margin:0 auto;padding:40px 24px 60px">

  <!-- Report header -->
  <div class="header-bar" style="background:linear-gradient(135deg,#1a1a2e 0%,#2d2d5e 100%);border-radius:12px;padding:40px;margin-bottom:44px;color:#fff;position:relative;overflow:hidden">
    <h1 style="font-size:30px;font-weight:800;margin-bottom:20px;letter-spacing:0.04em">Automotive Intelligence Report</h1>
    <button class="no-print" onclick="window.print()" style="background:#5865f2;color:#fff;border:none;border-radius:7px;padding:11px 26px;font-size:13px;font-weight:700;cursor:pointer;letter-spacing:0.06em">Save as PDF</button>
    ${analyzed.length === 0 ? `<div style="margin-top:20px;padding:14px 18px;background:#dc262633;border-radius:7px;font-size:13px;color:#ffaaaa">No analyses available. Run analysis on one or more videos first, then regenerate the report.</div>` : ""}
  </div>

  ${analyzed.length > 0 ? execSummary : ""}
  ${analyzed.length > 0 ? brandSection : ""}

  ${combinedBrandEntries.length >= 2 ? generateComparisonSectionHtml(combinedBrandEntries) : ""}

  ${combinedBrandEntries.length > 0 ? `
  <section>
    <h2 style="margin:0 0 20px;font-size:12px;font-weight:700;color:#5865f2;letter-spacing:0.22em;text-transform:uppercase;border-bottom:1px solid #e8eaf0;padding-bottom:10px">Brand Analysis · ${combinedBrandEntries.length} Compan${combinedBrandEntries.length !== 1 ? "ies" : "y"} · Aggregated Across All Sources</h2>
    ${combinedBrandEntries.map(([name, d]) => brandInsightsHtml(name, d)).join("")}
  </section>
  ` : ""}


</div>
</body>
</html>`;
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function DownloadReport() {
  const [generating, setGenerating] = useState(false);

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      const videosRes = await fetch("/api/videos");
      if (!videosRes.ok) throw new Error("Failed to fetch videos");
      const { videos = [] } = await videosRes.json();

      const analyses = (
        await Promise.all(
          videos.map(v =>
            fetch(`/api/analysis/${v.video_id}`)
              .then(r => (r.ok ? r.json() : null))
              .catch(() => null)
          )
        )
      ).filter(Boolean);

      const win = window.open("", "_blank");
      if (!win) {
        alert("Pop-up blocked. Please allow pop-ups for this site and try again.");
        return;
      }
      win.document.write(generateReport(analyses, videos));
      win.document.close();
    } catch (e) {
      console.error("Report generation failed:", e);
      alert("Failed to generate report. Make sure the backend is running.");
    } finally {
      setGenerating(false);
    }
  };

  return (
    <button
      onClick={handleGenerate}
      disabled={generating}
      title="Generate a PDF-ready report summarising all analyses"
      style={{
        display: "flex",
        alignItems: "center",
        gap: 7,
        background: generating ? "transparent" : "var(--accent, #5865f2)",
        color: generating ? "var(--text-muted, #8888a8)" : "#fff",
        border: `1px solid ${generating ? "var(--border, #d4d6e3)" : "var(--accent, #5865f2)"}`,
        borderRadius: 7,
        padding: "8px 16px",
        fontSize: 12,
        fontWeight: 700,
        cursor: generating ? "not-allowed" : "pointer",
        letterSpacing: "0.08em",
        textTransform: "uppercase",
        whiteSpace: "nowrap",
        flexShrink: 0,
        fontFamily: "var(--font, 'Inter', sans-serif)",
        transition: "opacity 0.2s",
        opacity: generating ? 0.6 : 1,
      }}
    >
      {generating ? "⟳ Generating…" : "↓ Download Report"}
    </button>
  );
}

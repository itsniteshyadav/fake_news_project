// Re-implements the trained scikit-learn pipeline (clean -> TF-IDF bigrams -> logistic regression) in plain JS.
const M = require("./model.json");
const STOP = new Set(M.stop);

const clean = (t) => String(t)
  .replace(/^.{0,60}?\(Reuters\)\s*-?\s*/, "").replace(/\bReuters\b/gi, "")
  .replace(/https?:\/\/\S+|www\.\S+/g, " ").replace(/\s+/g, " ").trim();

function score(text) {
  const toks = (clean(text).toLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) || []).filter((w) => !STOP.has(w));
  const counts = new Map();
  const add = (k) => counts.set(k, (counts.get(k) || 0) + 1);
  toks.forEach((w, i) => { add(w); if (i + 1 < toks.length) add(w + " " + toks[i + 1]); });

  const feats = []; let norm = 0;
  for (const [t, c] of counts) {
    if (!Object.hasOwn(M.terms, t)) continue;
    const [idf, coef] = M.terms[t], w = (1 + Math.log(c)) * idf;   // sublinear tf x idf
    norm += w * w; feats.push([t, w, coef]);
  }
  norm = Math.sqrt(norm) || 1;                                      // L2 normalisation
  let z = M.intercept; const contrib = [];
  for (const [t, w, coef] of feats) { const v = (w / norm) * coef; z += v; contrib.push([t, v]); }
  contrib.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  return { pFake: 1 / (1 + Math.exp(-z)), known: feats.length,
           top: contrib.slice(0, 12).map(([term, impact]) => ({ term, impact: +impact.toFixed(4) })),
           fake: contrib.filter((c) => c[1] > 0).slice(0, 15).map(([term, impact]) => ({ term, impact: +impact.toFixed(4) })),
           real: contrib.filter((c) => c[1] < 0).slice(0, 10).map(([term, impact]) => ({ term, impact: +impact.toFixed(4) })) };
}

// Scores each sentence separately so we can say how much of the article looks fake.
function sentences(text) {
  const list = text.split(/(?<=[.!?])\s+/).filter(Boolean).slice(0, 40)
    .map((t) => ({ t, p: t.split(/\s+/).length >= 6 ? +score(t).pFake.toFixed(3) : null }));
  const scored = list.filter((x) => x.p !== null), flagged = scored.filter((x) => x.p >= 0.5).length;
  return { list, scored: scored.length, flagged, pct: scored.length ? Math.round((100 * flagged) / scored.length) : null };
}
module.exports = { score, sentences, metrics: M.metrics };

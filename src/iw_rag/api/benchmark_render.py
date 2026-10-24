# ruff: noqa: E501
"""Shared client-side renderer for benchmark report detail + A/B compare.

The Benchmark Studio page (``benchmark_page.py``, served at
``/benchmark/ui``) and the Operator Console Results view
(``results_page.py``, served at ``/console/results``) both render the
**same** persisted ``benchmark_reports`` row.  They used to carry two
hand-copied ``loadReport()`` / ``compare()`` renderers that drifted —
and, worse, both read latency fields that the backend never writes
(``per_stage_ms.p50``, ``e2e_p50_ms`` …), so the whole latency panel
showed blanks.

This module is the single source of truth for that rendering.  It
exports three injectable fragments the two pages splice into their
inline ``<style>`` / ``<script>`` at placeholder comments:

* :data:`TOOLTIP_CSS` — the ``.iw-tip`` hover-glossary styling (also used
  by the Scoreboard page).
* :data:`REPORT_CSS`  — headline stat cards, the metric table, the value
  bars, and the score colour classes.  Uses **only** ``var(--…)`` tokens
  (never a raw hex), so the per-page "the brand teal appears exactly
  once" assertions stay green.
* :data:`REPORT_JS`   — the metric glossary + the ``IW_renderReport`` /
  ``IW_renderCompare`` functions returning HTML strings.  Injected
  *inside* each page's IIFE; it defines its own ``IW_*``-namespaced
  helpers so it never collides with the page's ``esc`` / ``fmt``.

Crucially, :data:`REPORT_JS` reads the **real** persisted shapes
(``iw_rag.eval.benchmark.assembly`` + ``perf``):

* ``latency.model_load_ms`` (ms, its own bucket — Invariant 4),
* ``latency.per_stage_ms.{retrieve_generate_ms,…}`` (per-query mean ms),
* ``latency.{ttft,tpot,e2e,queue_time}_p99_s`` (seconds, from the vLLM
  ``/metrics`` scraper — blank when the endpoint was not scraped),
* ``resource.gpu.{vram_*_mb,util_*_pct,available,device_name}``,
* ``resource.vllm_server.{prefix_cache_hit_rate,kv_cache_peak_pct,…}``.

Every number is labelled with its unit and carries a hover tooltip
explaining what it measures and how to read it.
"""

from __future__ import annotations

import json

from iw_rag.learn.metrics import plain_map

# ---------------------------------------------------------------------------
# CSS — tooltip glossary (shared with the Scoreboard page)
# ---------------------------------------------------------------------------

TOOLTIP_CSS = """
  /* Hover/focus glossary badge — a small "i" that reveals an explanation
     bubble.  No new colour literals (var() only) so the per-page
     single-brand-colour assertions hold. */
  .iw-tip { position: relative; display: inline-flex; align-items: center; justify-content: center; width: 15px; height: 15px; margin-left: 6px; border-radius: 50%; border: 1px solid var(--line); color: var(--muted); font-size: 10px; font-style: normal; font-weight: 700; line-height: 1; cursor: help; flex: none; vertical-align: middle; }
  .iw-tip:hover, .iw-tip:focus { color: var(--text); border-color: var(--teal-deep); outline: none; }
  .iw-tip .iw-bub { position: absolute; bottom: 160%; left: 50%; transform: translateX(-50%); width: 310px; max-width: 78vw; background: var(--ink-3); border: 1px solid var(--line); border-radius: 10px; padding: 10px 12px; font-size: 12px; font-weight: 400; line-height: 1.5; color: var(--text); text-align: left; letter-spacing: normal; text-transform: none; opacity: 0; visibility: hidden; transition: opacity .15s; z-index: 60; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
  .iw-tip:hover .iw-bub, .iw-tip:focus .iw-bub { opacity: 1; visibility: visible; }
  /* Headers and cells aligned to the right: open the bubble leftwards so it stays on screen. */
  th.num .iw-tip .iw-bub, td.num .iw-tip .iw-bub { left: auto; right: -8px; transform: none; }
  .iw-tip .iw-bub .t { display: block; color: var(--teal); font-weight: 700; margin-bottom: 3px; }
  .iw-tip .iw-bub .u { display: block; color: var(--muted); margin-bottom: 5px; font-size: 11px; }
  .iw-tip .iw-bub .p { display: block; margin-bottom: 6px; }
  .iw-tip .iw-bub .e, .iw-tip .iw-bub .g { display: block; margin-bottom: 6px; color: var(--muted); }
  .iw-tip .iw-bub .e b, .iw-tip .iw-bub .g b { color: var(--text); font-weight: 600; }
  .iw-tip .iw-bub .d { display: block; margin-top: 6px; padding-top: 6px; border-top: 1px solid var(--line); color: var(--muted); font-size: 11px; }
"""

# ---------------------------------------------------------------------------
# CSS — report detail (stat cards, metric table, value bars)
# ---------------------------------------------------------------------------

REPORT_CSS = """
  .iw-statgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 4px 0 20px; }
  .iw-stat { background: var(--ink-2); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
  .iw-stat .h { display: flex; align-items: center; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; color: var(--muted); }
  .iw-stat .b { font-size: 25px; font-weight: 700; font-family: "JetBrains Mono", ui-monospace, monospace; color: var(--teal); margin-top: 5px; line-height: 1.1; }
  .iw-stat .b.iw-na { color: var(--muted); font-size: 18px; }
  .iw-stat .u { font-size: 11px; color: var(--muted); margin-top: 3px; }
  .iw-sec { margin: 20px 0 8px; }
  .iw-sec h3 { font-family: "Space Grotesk", system-ui, sans-serif; font-size: 14px; margin: 0 0 2px; color: var(--text); text-transform: none; letter-spacing: 0; display: flex; align-items: center; }
  .iw-sec p.iw-secsub { margin: 0 0 10px; font-size: 12px; color: var(--muted); }
  table.iw-metrics { width: 100%; border-collapse: collapse; }
  table.iw-metrics th, table.iw-metrics td { text-align: left; padding: 9px 8px; border-bottom: 1px solid var(--line); font-size: 13px; vertical-align: middle; }
  table.iw-metrics th { color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }
  table.iw-metrics td.num { text-align: right; font-variant-numeric: tabular-nums; font-family: "JetBrains Mono", ui-monospace, monospace; white-space: nowrap; }
  table.iw-metrics td.lab { display: flex; align-items: center; }
  table.iw-metrics .scale { color: var(--muted); font-size: 12px; }
  .iw-bar { height: 5px; border-radius: 3px; background: var(--ink-3); overflow: hidden; margin-top: 5px; max-width: 170px; }
  .iw-bar > i { display: block; height: 100%; background: linear-gradient(90deg, var(--teal-deep), var(--teal)); }
  .iw-v-good { color: var(--teal); }
  .iw-v-mid { color: var(--warn); }
  .iw-v-low { color: var(--danger); }
  .iw-na { color: var(--muted); }
  .iw-note { font-size: 12px; color: var(--muted); margin: 8px 0 0; line-height: 1.5; }
  .iw-note.iw-warn { color: var(--warn); }
  .iw-cover { display: inline-flex; align-items: center; gap: 6px; font-size: 12px; padding: 3px 10px; border-radius: 999px; border: 1px solid var(--line); margin: 2px 0 8px; }
  .iw-cover.ok { color: var(--teal); border-color: var(--teal-deep); }
  .iw-cover.partial { color: var(--warn); border-color: var(--warn); }
  .iw-cover.none { color: var(--danger); border-color: var(--danger); }
  .iw-intro { font-size: 14px; line-height: 1.6; color: var(--text); background: var(--ink-2); border: 1px solid var(--line); border-left: 3px solid var(--teal); border-radius: 10px; padding: 14px 16px; margin: 4px 0 18px; }
  .iw-wf { margin-top: 4px; }
  .iw-wf-row { display: grid; grid-template-columns: 190px 1fr 132px; align-items: center; gap: 10px; padding: 5px 0; }
  .iw-wf-label { font-size: 13px; color: var(--text); }
  .iw-wf-track { height: 14px; background: var(--ink-3); border-radius: 7px; overflow: hidden; }
  .iw-wf-bar { height: 100%; background: linear-gradient(90deg, var(--teal-deep), var(--teal)); border-radius: 7px; }
  .iw-wf-val { font-size: 12px; color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; font-family: "JetBrains Mono", ui-monospace, monospace; }
  @media (max-width: 620px) { .iw-wf-row { grid-template-columns: 1fr; } .iw-wf-val { text-align: left; } }
  .iw-concl { background: var(--ink-2); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; margin: 4px 0 8px; font-size: 14px; line-height: 1.6; color: var(--text); white-space: pre-wrap; }
  .iw-concl-actions { display: flex; gap: 10px; align-items: center; margin: 8px 0 18px; }
"""

# ---------------------------------------------------------------------------
# JS — glossary + renderers.  Raw string (regex backslashes stay literal);
# injected INSIDE each page's IIFE.  All identifiers are IW_-namespaced.
# ---------------------------------------------------------------------------

_REPORT_JS_TEMPLATE = r"""
  // -----------------------------------------------------------------------
  // Metric glossary — what each number means + how to read it.  Surfaced as
  // hover tooltips so an operator never sees a bare "0.234" again.
  // -----------------------------------------------------------------------
  var IW_GLOSSARY = {
    recall_at_5:        { label: "Recall@5",  unit: "0–1 · higher is better", desc: "Of the passages that SHOULD answer the question, the fraction that appeared in the top 5 retrieved results. Measures whether retrieval surfaces the right evidence at all." },
    recall_at_10:       { label: "Recall@10", unit: "0–1 · higher is better", desc: "Same as Recall@5, but counting the top 10 retrieved passages." },
    ndcg_at_5:          { label: "nDCG@5",    unit: "0–1 · higher is better", desc: "Normalised Discounted Cumulative Gain over the top 5. Rewards ranking the most relevant passages near the top, not merely retrieving them. 1.0 = perfect ordering." },
    ndcg_at_10:         { label: "nDCG@10",   unit: "0–1 · higher is better", desc: "Ranking quality over the top 10 retrieved passages." },
    mrr:                { label: "MRR",       unit: "0–1 · higher is better", desc: "Mean Reciprocal Rank: 1 / (position of the first relevant passage), averaged over questions. 1.0 = the first hit was always relevant; 0.5 = it was typically 2nd." },
    faithfulness:       { label: "Faithfulness",      unit: "0–1 · higher is better", desc: "Of the claims the model made in its answer, the fraction actually supported by the retrieved context. Low = it invented things beyond its sources (hallucination). Scored by the judge model." },
    context_precision:  { label: "Context precision", unit: "0–1 · higher is better", desc: "Of the passages handed to the generator, the fraction that were genuinely relevant. Low = the context was padded with noise." },
    faithfulness_kappa: { label: "Judge κ", unit: "Cohen's κ · −1…1 · higher = more agreement", desc: "Agreement between the LLM judge and human labels on faithfulness. Only populated when a separate kappa harness has run; blank here means this run did not measure it.", nobar: true },
    context_recall:     { label: "Context recall",      unit: "0–1 · higher is better", desc: "Of the facts in the gold reference answer, the fraction covered by the retrieved context. Requires a dataset that ships reference answers." },
    factual_correctness:{ label: "Factual correctness", unit: "0–1 · higher is better", desc: "How factually aligned the generated answer is with the gold reference answer. Requires reference answers." },
    noise_sensitivity:  { label: "Noise sensitivity",   unit: "0–1 · LOWER is better", desc: "How often the answer makes claims drawn from irrelevant retrieved passages. Lower = more robust to noisy context. Requires reference answers.", lower: true },
    model_load_ms:      { label: "Model load",          unit: "cold-swap time", desc: "Wall-clock time to cold-swap and load the generator onto the GPU. Tracked in its own bucket and NEVER folded into per-query or end-to-end latency." },
    retrieve_generate_ms:{ label: "Retrieve + generate", unit: "per query, mean", desc: "Average wall-clock time for one query's retrieve + generate stage, measured by the app's own request timeline." },
    ttft_p99_s:         { label: "TTFT p99",  unit: "seconds · lower is better", desc: "Time-To-First-Token, 99th percentile: how long until the first output token appears. Reported by the vLLM server; blank if /metrics was not scraped." },
    tpot_p99_s:         { label: "TPOT p99",  unit: "seconds/token · lower is better", desc: "Time-Per-Output-Token (inter-token latency), 99th percentile — the streaming speed once generation starts. Reported by vLLM." },
    e2e_p99_s:          { label: "End-to-end p99", unit: "seconds · lower is better", desc: "Full request latency at the 99th percentile, as seen by the vLLM server. The headline 'how slow is the worst case' number." },
    queue_time_p99_s:   { label: "Queue p99", unit: "seconds · lower is better", desc: "Time a request waited in the vLLM queue before processing began, 99th percentile." },
    vram_peak_mb:       { label: "VRAM peak",      unit: "GPU memory", desc: "Highest GPU memory in use at any point during the measured run." },
    vram_baseline_mb:   { label: "VRAM baseline",  unit: "GPU memory", desc: "The idle GPU-memory floor before this run's generator became resident." },
    vram_incremental_mb:{ label: "VRAM footprint", unit: "GPU memory", desc: "Peak minus baseline — the resident model's own memory footprint for this run." },
    util_peak_pct:      { label: "GPU util peak",  unit: "%", desc: "Highest GPU compute utilisation sampled during the run." },
    util_mean_pct:      { label: "GPU util mean",  unit: "%", desc: "Average GPU compute utilisation across the measured run." },
    prefix_cache_hit_rate: { label: "Prefix-cache hit rate", unit: "0–1 · higher is better", desc: "Fraction of prompt tokens served from vLLM's prefix cache instead of recomputed. Higher = cheaper repeated prefixes." },
    kv_cache_peak_pct:  { label: "KV-cache peak", unit: "%", desc: "Peak fraction of vLLM's KV-cache pool in use. Near 100% means the run was memory-bound." },
    generation_tokens_delta: { label: "Generated tokens", unit: "tokens (this run)", desc: "Total output tokens vLLM generated across the run (counter delta). Used to derive throughput on the Scoreboard." }
  };

  // Plain-language layer (single source: iw_rag/learn/metrics.py): what each number
  // means in everyday words, a concrete example and which direction is better.
  var IW_PLAIN = __IW_PLAIN__;

  function IW_g(key) {
    if (IW_GLOSSARY[key]) return IW_GLOSSARY[key];
    var pl = IW_PLAIN[key];
    return { label: pl ? pl.name : key, unit: "", desc: "" };
  }

  function IW_esc(s) { var d = document.createElement("div"); d.textContent = s == null ? "" : s; return d.innerHTML; }

  function IW_num(v, d) {
    if (v == null || isNaN(v)) return "—";
    if (d == null) d = 3;
    return Number(v).toFixed(d);
  }

  // Tooltip badge for a key: label + unit, then the plain-language meaning, an example and
  // which direction is better; the original technical description stays as a last line.
  function IW_tip(key) {
    var g = IW_g(key), pl = IW_PLAIN[key];
    var body = "";
    if (pl) {
      body += '<span class="p">' + IW_esc(pl.plain) + '</span>' +
        '<span class="e"><b>Example:</b> ' + IW_esc(pl.example) + '</span>' +
        '<span class="g"><b>Good or bad?</b> ' + IW_esc(pl.good) + (pl.watch ? ' ' + IW_esc(pl.watch) : '') + '</span>';
      if (g.desc) body += '<span class="d">' + IW_esc(g.desc) + '</span>';
    } else {
      body = IW_esc(g.desc);
    }
    return '<span class="iw-tip" tabindex="0" aria-label="' + IW_esc(g.label) + '">i' +
      '<span class="iw-bub"><span class="t">' + IW_esc(g.label) + '</span>' +
      (g.unit ? '<span class="u">' + IW_esc(g.unit) + '</span>' : '') + body + '</span></span>';
  }

  // Generic tooltip (no glossary entry) — used for section headers.
  function IW_tipRaw(title, body) {
    return '<span class="iw-tip" tabindex="0" aria-label="' + IW_esc(title) + '">i' +
      '<span class="iw-bub"><span class="t">' + IW_esc(title) + '</span>' + IW_esc(body) + '</span></span>';
  }

  // Promoted scalar column first, else mean-reduce the per-query vector in
  // the metrics blob (handles faithfulness_mean vs the blob key
  // "faithfulness", etc.).
  function IW_pick(rep, keys) {
    for (var k = 0; k < keys.length; k++) {
      var name = keys[k];
      if (rep[name] != null) return rep[name];
      var v = (rep.metrics || {})[name];
      if (v != null) {
        if (Object.prototype.toString.call(v) === "[object Array]") {
          var s = 0, c = 0;
          for (var i = 0; i < v.length; i++) { if (typeof v[i] === "number" && !isNaN(v[i])) { s += v[i]; c++; } }
          return c ? s / c : null;
        }
        return v;
      }
    }
    return null;
  }

  function IW_scoreClass(v, lower) {
    var good = lower ? (v <= 0.30) : (v >= 0.75);
    var mid = lower ? (v <= 0.60) : (v >= 0.50);
    return good ? "iw-v-good" : (mid ? "iw-v-mid" : "iw-v-low");
  }
  function IW_bar(v) {
    if (v == null || isNaN(v)) return "";
    var pct = Math.max(0, Math.min(1, Number(v))) * 100;
    return '<div class="iw-bar"><i style="width:' + pct.toFixed(0) + '%"></i></div>';
  }

  // Unit-aware formatters for the speed / resource panels.
  function IW_fmtSeconds(s) {
    if (s == null || isNaN(s)) return null;
    s = Number(s);
    if (s < 1) return (s * 1000).toFixed(0) + " ms";
    return s.toFixed(2) + " s";
  }
  function IW_fmtMs(ms) {
    if (ms == null || isNaN(ms)) return null;
    ms = Number(ms);
    if (ms >= 1000) return (ms / 1000).toFixed(ms >= 10000 ? 1 : 2) + " s";
    return ms.toFixed(0) + " ms";
  }
  function IW_fmtMB(mb) {
    if (mb == null || isNaN(mb)) return null;
    mb = Number(mb);
    if (mb >= 1024) return (mb / 1024).toFixed(1) + " GB";
    return mb.toFixed(0) + " MB";
  }
  function IW_fmtPct(p) {
    if (p == null || isNaN(p)) return null;
    return Number(p).toFixed(0) + "%";
  }
  function IW_fmtRate01(v) {
    if (v == null || isNaN(v)) return null;
    return (Number(v) * 100).toFixed(0) + "%";
  }

  // ---- headline stat cards -------------------------------------------------
  function IW_statCard(key, valueStr) {
    var g = IW_g(key);
    var na = valueStr == null;
    return '<div class="iw-stat"><div class="h">' + IW_esc(g.label) + IW_tip(key) + '</div>' +
      '<div class="b' + (na ? ' iw-na' : '') + '">' + (na ? "not measured" : IW_esc(valueStr)) + '</div>' +
      '<div class="u">' + IW_esc(g.unit) + '</div></div>';
  }

  function IW_renderHeadline(rep) {
    var faith = IW_pick(rep, ["faithfulness_mean", "faithfulness"]);
    var ndcg = IW_pick(rep, ["ndcg_at_5"]);
    var lat = rep.latency || {};
    var ps = lat.per_stage_ms || {};
    var cards = "";
    var any = false;
    if (faith != null) { cards += IW_statCard("faithfulness", IW_num(faith, 3)); any = true; }
    if (ndcg != null) { cards += IW_statCard("ndcg_at_5", IW_num(ndcg, 3)); any = true; }
    // Prefer the vLLM end-to-end p99; fall back to the app's per-query mean.
    if (lat.e2e_p99_s != null) { cards += IW_statCard("e2e_p99_s", IW_fmtSeconds(lat.e2e_p99_s)); any = true; }
    else if (ps.retrieve_generate_ms != null) { cards += IW_statCard("retrieve_generate_ms", IW_fmtMs(ps.retrieve_generate_ms)); any = true; }
    var gpu = (rep.resource || {}).gpu || {};
    if (gpu.vram_incremental_mb != null) { cards += IW_statCard("vram_incremental_mb", IW_fmtMB(gpu.vram_incremental_mb)); any = true; }
    if (!any) return "";
    return '<div class="iw-statgrid">' + cards + '</div>';
  }

  // ---- how to read the numbers ----------------------------------------------
  function IW_legend() {
    return '<p class="iw-note">How to read the colours on 0–1 scores: ' +
      '<span class="iw-v-good">green</span> is 0.75 or more, <span class="iw-v-mid">amber</span> is 0.5 to 0.75, ' +
      '<span class="iw-v-low">red</span> is below 0.5. It is a rough guide, not a verdict, and for "lower is better" metrics the colours flip. ' +
      'Hover the <b>i</b> beside any metric for what it means in plain words, with an example.</p>';
  }

  // ---- quality metric tiers ------------------------------------------------
  function IW_metricRow(rep, def) {
    var g = IW_g(def.g);
    var v = IW_pick(rep, def.keys);
    var lower = !!g.lower;
    var valHtml, scaleHtml;
    if (v == null) {
      valHtml = '<span class="iw-na">—</span>';
      scaleHtml = '<span class="scale">' + IW_esc(def.na || "not measured") + '</span>';
    } else {
      valHtml = '<span class="' + IW_scoreClass(Number(v), lower) + '">' + IW_num(v, 3) + '</span>';
      scaleHtml = g.nobar ? '<span class="scale">' + IW_esc(g.unit) + '</span>' : IW_bar(v);
    }
    return '<tr><td class="lab">' + IW_esc(g.label) + IW_tip(def.g) + '</td>' +
      '<td class="num">' + valHtml + '</td><td>' + scaleHtml + '</td></tr>';
  }

  // Judge-coverage banner — explains WHY the judged tier may be blank.
  // ``faithfulness`` is the only judged metric the benchmark image scores
  // live (via the FaithfulnessJudge); n_judged counts the queries that got a
  // verdict.  0/N is the loud case the operator just hit: the judge endpoint
  // was unreachable or every verdict failed to parse.
  function IW_judgeCoverage(rep) {
    var nq = rep.n_queries, nj = rep.n_judged;
    if (nq == null) return "";
    if (nj == null || nj === 0) {
      return '<span class="iw-cover none">⚠ Judge scored 0 of ' + IW_esc(nq) + ' queries</span>' +
        '<p class="iw-note iw-warn">The judge produced no usable verdict for any query, so the metrics below are blank. ' +
        'Most often the judge endpoint was unreachable during the run, or every verdict failed to parse. ' +
        'The run still records retrieval + speed; check the api logs for "judge scored 0" to see the exact failure.</p>';
    }
    if (nj < nq) {
      return '<span class="iw-cover partial">Judge scored ' + IW_esc(nj) + ' of ' + IW_esc(nq) + ' queries</span>' +
        '<p class="iw-note">The remaining queries failed and are excluded from the means below.</p>';
    }
    return '<span class="iw-cover ok">Judge scored all ' + IW_esc(nq) + ' queries</span>';
  }

  function IW_metricTier(rep, tier) {
    var rowsHtml = "";
    for (var i = 0; i < tier.rows.length; i++) rowsHtml += IW_metricRow(rep, tier.rows[i]);
    var head = '<div class="iw-sec"><h3>' + IW_esc(tier.title) + IW_tipRaw(tier.title, tier.help) + '</h3>' +
      '<p class="iw-secsub">' + IW_esc(tier.sub) + '</p></div>';
    var note = tier.judged ? IW_judgeCoverage(rep) : (tier.note ? '<p class="iw-note">' + IW_esc(tier.note) + '</p>' : "");
    return head + note + '<table class="iw-metrics"><thead><tr><th>Metric</th><th class="num">Value</th><th>Scale</th></tr></thead><tbody>' + rowsHtml + '</tbody></table>';
  }

  // ``na`` strings spell out WHY a metric is blank — they are distinct on
  // purpose: faithfulness blanks because the judge failed (see the coverage
  // banner); the Ragas-sourced metrics blank because the Ragas package is not
  // enabled in the benchmark image; κ comes from a separate harness.
  var IW_NA_RAGAS = "needs the Ragas metrics package (not enabled in this benchmark image)";
  var IW_NA_RAGAS_REF = "needs gold reference answers + the Ragas package (not enabled here)";
  var IW_TIERS = [
    { title: "Retrieval quality", sub: "Did the search fetch the right passages? Scored against the dataset's answer key; no AI judge is involved.",
      help: "These compare the retrieved passages against the dataset's relevance labels. They do not look at the generated answer at all.",
      rows: [
        { g: "recall_at_5", keys: ["recall_at_5"] },
        { g: "recall_at_10", keys: ["recall_at_10"] },
        { g: "ndcg_at_5", keys: ["ndcg_at_5"] },
        { g: "ndcg_at_10", keys: ["ndcg_at_10"] },
        { g: "mrr", keys: ["mrr"] }
      ] },
    { title: "Answer quality — reference-free", sub: "Is the answer honest? A judge model checks it against the passages that were retrieved; no reference answer is needed.", judged: true,
      help: "The judge reads the question, the retrieved context, and the generated answer, then scores how well-grounded the answer is.",
      rows: [
        { g: "faithfulness", keys: ["faithfulness_mean", "faithfulness"], na: "the judge returned no verdict (see the banner above)" },
        { g: "context_precision", keys: ["context_precision"], na: IW_NA_RAGAS },
        { g: "faithfulness_kappa", keys: ["faithfulness_kappa"], na: "measured by a separate κ harness, not this run" }
      ] },
    { title: "Answer quality — needs reference answers", sub: "Does the answer match a known-correct reference? Needs reference answers and the Ragas package.",
      note: "These compare the answer against a known-correct reference and are produced by the Ragas package, which is not enabled in this benchmark image — so they are blank for this run (never zero).",
      help: "These require a known-correct answer to compare against AND the Ragas metrics package; both must be present for a value to appear.",
      rows: [
        { g: "context_recall", keys: ["context_recall"], na: IW_NA_RAGAS_REF },
        { g: "factual_correctness", keys: ["factual_correctness"], na: IW_NA_RAGAS_REF },
        { g: "noise_sensitivity", keys: ["noise_sensitivity"], na: IW_NA_RAGAS_REF }
      ] }
  ];

  // ---- speed / latency -----------------------------------------------------
  function IW_latCell(key, valueStr) {
    var g = IW_g(key);
    var na = valueStr == null;
    return '<div><div class="v' + (na ? ' iw-na' : '') + '">' + (na ? "—" : IW_esc(valueStr)) +
      '</div><div class="l">' + IW_esc(g.label) + IW_tip(key) + '</div></div>';
  }

  function IW_renderSpeed(rep) {
    var lat = rep.latency || {};
    var ps = lat.per_stage_ms || {};
    var cells = IW_latCell("model_load_ms", IW_fmtMs(lat.model_load_ms));
    // Per-stage means (the app's own timeline). retrieve_generate_ms is the
    // canonical one; render any other stage keys generically too.
    var stageKeys = [];
    for (var sk in ps) { if (Object.prototype.hasOwnProperty.call(ps, sk)) stageKeys.push(sk); }
    if (stageKeys.length === 0) cells += IW_latCell("retrieve_generate_ms", null);
    for (var i = 0; i < stageKeys.length; i++) {
      var key = stageKeys[i];
      if (IW_GLOSSARY[key]) cells += IW_latCell(key, IW_fmtMs(ps[key]));
      else cells += '<div><div class="v">' + IW_esc(IW_fmtMs(ps[key]) || "—") + '</div><div class="l">' + IW_esc(key) + '</div></div>';
    }
    cells += IW_latCell("ttft_p99_s", IW_fmtSeconds(lat.ttft_p99_s));
    cells += IW_latCell("tpot_p99_s", IW_fmtSeconds(lat.tpot_p99_s));
    cells += IW_latCell("e2e_p99_s", IW_fmtSeconds(lat.e2e_p99_s));
    cells += IW_latCell("queue_time_p99_s", IW_fmtSeconds(lat.queue_time_p99_s));
    var sawP99 = lat.ttft_p99_s != null || lat.e2e_p99_s != null || lat.tpot_p99_s != null || lat.queue_time_p99_s != null;
    var note = sawP99
      ? "Load time is a separate bucket — never added into per-query or end-to-end latency."
      : "The p99 figures come from the vLLM /metrics endpoint; they are blank because it was not scraped for this run. Load time is always its own bucket.";
    return '<div class="iw-sec"><h3>Speed &amp; latency</h3>' +
      '<p class="iw-secsub">How fast the run answered. Per-query mean is wall-clock; the p99 figures are the worst-case tail from the vLLM server.</p></div>' +
      '<div class="latency-grid">' + cells + '</div>' +
      '<p class="iw-note">' + IW_esc(note) + '</p>';
  }

  // ---- GPU / VRAM + vLLM server -------------------------------------------
  function IW_renderResource(rep) {
    var res = rep.resource || {};
    var gpu = res.gpu || res.vram || {};
    var html = "";
    var hasGpu = gpu.vram_peak_mb != null || gpu.vram_incremental_mb != null || gpu.util_peak_pct != null;
    if (gpu.available === false) {
      html += '<div class="iw-sec"><h3>GPU &amp; memory</h3></div><p class="iw-note">No GPU was sampled for this run (the GPU/VRAM sampler was unavailable).</p>';
    } else if (hasGpu) {
      var cells = IW_latCell("vram_peak_mb", IW_fmtMB(gpu.vram_peak_mb)) +
        IW_latCell("vram_baseline_mb", IW_fmtMB(gpu.vram_baseline_mb)) +
        IW_latCell("vram_incremental_mb", IW_fmtMB(gpu.vram_incremental_mb)) +
        IW_latCell("util_peak_pct", IW_fmtPct(gpu.util_peak_pct)) +
        IW_latCell("util_mean_pct", IW_fmtPct(gpu.util_mean_pct));
      var dev = gpu.device_name ? '<p class="iw-note">Device: ' + IW_esc(gpu.device_name) + '</p>' : "";
      html += '<div class="iw-sec"><h3>GPU &amp; memory</h3><p class="iw-secsub">How much of the single GPU this configuration consumed.</p></div>' +
        '<div class="latency-grid">' + cells + '</div>' + dev;
    }
    var vllm = res.vllm_server || {};
    var hasVllm = vllm.prefix_cache_hit_rate != null || vllm.kv_cache_peak_pct != null || vllm.generation_tokens_delta != null;
    if (hasVllm) {
      var vcells = IW_latCell("prefix_cache_hit_rate", IW_fmtRate01(vllm.prefix_cache_hit_rate)) +
        IW_latCell("kv_cache_peak_pct", IW_fmtPct(vllm.kv_cache_peak_pct)) +
        IW_latCell("generation_tokens_delta", vllm.generation_tokens_delta != null ? String(vllm.generation_tokens_delta) : null);
      html += '<div class="iw-sec"><h3>vLLM server</h3><p class="iw-secsub">Server-side cache and token counters scraped from vLLM for this run.</p></div>' +
        '<div class="latency-grid">' + vcells + '</div>';
    }
    return html;
  }

  // ---- public: full report body -------------------------------------------
  function IW_reportMeta(rep) {
    var parts = [];
    if (rep.report_label) parts.push("label=" + rep.report_label);
    parts.push("queries=" + (rep.n_queries != null ? rep.n_queries : "?"));
    if (rep.n_judged != null) parts.push("judged=" + rep.n_judged);
    if (rep.created_at) parts.push(rep.created_at);
    if (rep.created_by) parts.push("by " + rep.created_by);
    return IW_esc(parts.join("  ·  "));
  }

  // ---- intro paragraph (templated, deterministic) -------------------------
  function IW_renderIntro(rep) {
    var c = rep.config_snapshot || {};
    var p = c.params || {};
    var gen = c.generator || "the configured generator";
    var emb = c.embedder || "an embedder";
    var rer = c.reranker || "a reranker";
    var grd = c.grader || "the configured grader";
    var nq = rep.n_queries != null ? rep.n_queries : "?";
    var ds = rep.dataset_name || c.dataset_id || "the golden";
    var tk = p.top_k != null ? p.top_k : "–";
    var tp = p.temperature != null ? p.temperature : "–";
    var mt = p.max_tokens != null ? p.max_tokens : "–";
    var txt =
      "This benchmark evaluated the " + gen + " generator — with " + emb + " embeddings and " + rer +
      " reranking — over the “" + ds + "” set of " + nq + " questions on a single RTX 5090 (32 GB). " +
      "Retrieval quality (recall, nDCG, MRR) was scored deterministically against the dataset's relevance labels. " +
      "Answer quality was judged locally, never by a hosted API: faithfulness by Lynx-8B, and context precision/recall, " +
      "factual correctness and noise sensitivity by the Ragas suite using " + grd + " as the grader. " +
      "Run parameters: top-k=" + tk + ", temperature=" + tp + ", max-tokens=" + mt + ".";
    return '<div class="iw-intro">' + IW_esc(txt) + '</div>';
  }

  // ---- step-timings waterfall ---------------------------------------------
  var IW_PHASES = [
    { key: "model_load_ms", label: "Model load / swap" },
    { key: "generate_ms", label: "Retrieval + generation" },
    { key: "faithfulness_ms", label: "Faithfulness judge (Lynx)" },
    { key: "grading_ms", label: "Grading judge (Selene)" },
    { key: "ragas_ms", label: "Ragas metrics" }
  ];
  function IW_renderTimings(rep) {
    var pm = (rep.latency || {}).phase_ms;
    if (!pm) return "";  // runs from before the timing instrumentation
    var phases = [];
    for (var i = 0; i < IW_PHASES.length; i++) {
      var k = IW_PHASES[i].key;
      if (pm[k] != null) phases.push({ label: IW_PHASES[i].label, ms: Number(pm[k]) });
    }
    // Fallback: no engine sub-split, but a total scoring time is present.
    if (phases.length === 0 && pm.score_ms != null) {
      phases = [
        { label: "Model load / swap", ms: Number(pm.model_load_ms || 0) },
        { label: "Retrieval + generation", ms: Number(pm.generate_ms || 0) },
        { label: "Scoring (judging)", ms: Number(pm.score_ms || 0) }
      ];
    }
    if (phases.length === 0) return "";
    var total = 0;
    for (var t = 0; t < phases.length; t++) total += phases[t].ms;
    if (total <= 0) total = 1;
    var rows = "";
    for (var j = 0; j < phases.length; j++) {
      var ph = phases[j], pct = ph.ms / total * 100;
      rows += '<div class="iw-wf-row"><div class="iw-wf-label">' + IW_esc(ph.label) + '</div>' +
        '<div class="iw-wf-track"><div class="iw-wf-bar" style="width:' + pct.toFixed(1) + '%"></div></div>' +
        '<div class="iw-wf-val">' + IW_esc(IW_fmtMs(ph.ms)) + ' · ' + pct.toFixed(0) + '%</div></div>';
    }
    return '<div class="iw-sec"><h3>Step timings' +
      IW_tipRaw("Step timings", "Wall-clock time spent in each phase of the run. The judging phases (faithfulness + Ragas) usually dominate — Ragas decomposes every answer into claims and checks each one.") +
      '</h3><p class="iw-secsub">How long each step took — total ≈ ' + IW_esc(IW_fmtMs(total)) + '.</p></div>' +
      '<div class="iw-wf">' + rows + '</div>';
  }

  // ---- LLM conclusion (populated by the run-time generator; stored in the
  // metrics blob under "conclusion"; the Results page owns the Regenerate +
  // Generate-Report buttons) ------------------------------------------------
  function IW_renderConclusion(rep) {
    var m = rep.metrics || {};
    var c = m.conclusion;
    var text = c ? (typeof c === "string" ? c : c.text) : "";
    var basis = (c && typeof c === "object" && c.basis) ? c.basis : "";
    var head = '<div class="iw-sec"><h3>Conclusion' +
      IW_tipRaw("Conclusion", "A local-LLM summary of how this run did, grounded in the measured numbers and compared against prior runs on the same dataset. Regenerate refreshes it against the latest comparable runs.") +
      '</h3></div>';
    var bodyHtml = text
      ? '<div class="iw-concl">' + IW_esc(text) + '</div>'
      : '<p class="iw-note">No conclusion stored for this run — Regenerate writes one from the measured numbers + comparable runs.</p>';
    var note = basis === "regenerated"
      ? '<span class="muted">refreshed against the latest runs</span>'
      : (text ? '<span class="muted">written at run time</span>' : '');
    var actions = '<div class="iw-concl-actions">' +
      '<button class="btn-ghost" data-regen-conclusion="' + IW_esc(rep.id || "") + '">&#8635; Regenerate</button>' +
      note + '</div>';
    return head + bodyHtml + actions;
  }

  function IW_renderReport(rep) {
    var html = '<div class="iw-concl-actions" style="justify-content:flex-end;margin:0 0 8px">' +
      '<button class="btn-primary" data-pdf-report="' + IW_esc(rep.id || "") + '">&#128196; Generate Report (PDF)</button></div>';
    html += IW_renderIntro(rep);
    html += IW_renderHeadline(rep);
    html += IW_legend();
    html += IW_renderConclusion(rep);
    for (var i = 0; i < IW_TIERS.length; i++) html += IW_metricTier(rep, IW_TIERS[i]);
    html += IW_renderTimings(rep);
    html += IW_renderSpeed(rep);
    html += IW_renderResource(rep);
    if (rep.config_snapshot) {
      html += '<details style="margin-top:14px"><summary>Model configuration snapshot</summary>' +
        '<pre style="white-space:pre-wrap;font-size:12px;color:var(--muted)">' + IW_esc(JSON.stringify(rep.config_snapshot, null, 2)) + '</pre></details>';
    }
    return html;
  }

  // One plain-language sentence for an A/B result: what changed on the two metrics that
  // decide a winner, and whether the gap can be trusted.
  function IW_plainVerdict(d) {
    var rows = d.rows || [], pick = {};
    for (var i = 0; i < rows.length; i++) pick[rows[i].metric] = rows[i];
    function part(name, row) {
      if (!row || row.delta == null || isNaN(row.delta)) return name + ": not measured in both runs";
      var dl = Number(row.delta);
      if (Math.abs(dl) < 0.005) return name + ": B is about the same as A";
      var s = name + ": B is " + Math.abs(dl).toFixed(3) + (dl > 0 ? " higher" : " lower") + " than A";
      return s + (row.significant ? " (a real difference)" : " (not significant, so it may be luck)");
    }
    var out = "In short. " + part("Search quality (nDCG@5)", pick.ndcg_at_5) + ". " + part("Honesty (faithfulness)", pick.faithfulness) + ". " +
      (d.winner ? "Run " + d.winner + " wins: it is better on both." : "There is no clear winner: neither run is better on both.");
    return '<div class="iw-intro">' + IW_esc(out) + '</div>';
  }

  // ---- public: A/B compare body -------------------------------------------
  function IW_renderCompare(d) {
    var html = "";
    if (d.underpowered || /underpower|small.{0,8}sample|n[^\d]{0,3}37|n[^\d]{0,3}150|detectable/i.test(d.caveat || "")) {
      html += '<div class="caveat-box"><strong>Underpowered.</strong>' + IW_tip("underpowered") + ' Read this with care — ' +
        IW_esc(d.caveat || "with only ~37 queries the test reliably detects only ~0.08–0.12 retrieval / ~0.06–0.10 faithfulness gaps. Grow the golden set toward ~150 queries for firmer conclusions.") + '</div>';
    }
    var wClass = d.winner === "A" ? "A" : (d.winner === "B" ? "B" : "tie");
    var wText = d.winner
      ? ("Winner: " + d.winner + " — better on BOTH nDCG@5 and faithfulness")
      : "No clear winner — neither config wins on both nDCG@5 and faithfulness";
    html += '<p style="margin-top:6px"><span class="compare-winner ' + wClass + '">' + IW_esc(wText) + '</span>' + IW_tipRaw("How the winner is decided", "A configuration is only crowned when it beats the other on both ranking quality (nDCG@5) AND grounding (faithfulness). Ties and split decisions report no winner on purpose.") + '</p>';
    html += IW_plainVerdict(d);
    if (d.reasoning) html += '<div class="compare-reasoning">' + IW_esc(d.reasoning) + '</div>';
    // Canonical AC5 column set (metric, val A, val B, delta, p_value,
    // p_corrected, CI 95%, significant) — kept verbatim so the stats
    // columns stay machine-anchored; the hover tooltips add the plain
    // language explanation on top.
    html += '<table class="iw-metrics"><thead><tr><th>metric</th><th class="num">val A</th><th class="num">val B</th><th class="num">delta' + IW_tip("delta") + '</th><th class="num">p_value' + IW_tip("p_value") + '</th><th class="num">p_corrected' + IW_tip("p_corrected") + '</th><th class="num">CI 95%' + IW_tip("ci_95") + '</th><th>significant' + IW_tip("significant") + '</th></tr></thead><tbody>';
    var rows = d.rows || [];
    for (var i = 0; i < rows.length; i++) {
      var row = rows[i];
      var chip = row.significant ? '<span class="chip ok">✓ significant</span>' : '<span class="chip muted">not significant</span>';
      var ci = row.ci || (row.ci_95 && row.ci_95.length === 2 ? row.ci_95 : null);
      var ciStr = ci ? ("[" + IW_num(ci[0], 3) + ", " + IW_num(ci[1], 3) + "]") : "—";
      var gk = (IW_GLOSSARY[row.metric] || IW_PLAIN[row.metric]) ? IW_tip(row.metric) : "";
      html += '<tr><td class="lab">' + IW_esc(row.metric) + gk + '</td>' +
        '<td class="num">' + IW_num(row.val_a, 3) + '</td>' +
        '<td class="num">' + IW_num(row.val_b, 3) + '</td>' +
        '<td class="num">' + IW_num(row.delta, 3) + '</td>' +
        '<td class="num">' + IW_num(row.p_value, 4) + '</td>' +
        '<td class="num">' + IW_num(row.p_corrected, 4) + '</td>' +
        '<td class="num">' + ciStr + '</td><td>' + chip + '</td></tr>';
    }
    html += '</tbody></table>';
    html += '<p class="iw-note">Delta is B minus A: for "lower is better" metrics a negative delta is the improvement. ' +
      '"Significant" needs the corrected p-value under 0.05 and a 95% range that stays on one side of zero. ' +
      'With few questions, read a gap that is not significant as "this data cannot tell", not as "no difference". Hover the <b>i</b> beside a column for an example.</p>';
    return html;
  }
"""


def _plain_json() -> str:
    """The plain-language explanations as JSON that is safe inside a ``<script>`` block."""
    return json.dumps(plain_map(), ensure_ascii=False).replace("</", "<\\/")


REPORT_JS = _REPORT_JS_TEMPLATE.replace("__IW_PLAIN__", _plain_json())


def plain_terms_json() -> str:
    """Plain-language explanations for pages that carry their own script (the Scoreboard)."""
    return _plain_json()


__all__ = ["REPORT_CSS", "REPORT_JS", "TOOLTIP_CSS", "plain_terms_json"]

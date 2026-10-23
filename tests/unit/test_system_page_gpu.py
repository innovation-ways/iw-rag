# ruff: noqa: E501
"""System-view GPU-aware UI tests (F-00098).

The page is a self-contained vanilla-JS document, so two layers are checked:
static markup/endpoint wiring on :func:`system_page_html`, and — when ``node`` is
available — a behavioural run of the real page script against a stub DOM and a
scripted ``fetch`` (profiles, capacity warnings, yield banner, history chart).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from iw_rag.api.system_page import system_page_html

HTML = system_page_html()


# ------------------------------- static checks -------------------------------


def test_new_endpoints_are_wired() -> None:
    for path in ("/system/profiles", "/system/profile", "/system/capacity", "/system/history"):
        assert f'API + "{path}' in HTML, path


def test_new_cards_present_and_gated_by_default() -> None:
    for card in ("profileCard", "yieldCard", "historyCard"):
        assert re.search(rf'<div class="card hidden" id="{card}"', HTML), card
    assert 'id="capWarn"' in HTML and 'id="vramSplit"' in HTML


def test_history_window_controls() -> None:
    for secs in ("300", "900", "3600"):
        assert f'data-hist-window="{secs}"' in HTML
    assert 'aria-label="History window"' in HTML


def test_no_external_assets_or_chart_library() -> None:
    assert "cdn." not in HTML and "https://" not in HTML.replace("http://www.w3.org", "")
    assert "<script src" not in HTML


def test_force_requires_confirmation() -> None:
    assert HTML.count("window.confirm(") >= 2  # profile override + component force-start


def test_changing_gpu_cap_needs_restart_is_stated() -> None:
    assert HTML.count("needs a container restart") == 1


# ------------------------------- behavioural (node) -------------------------------

HARNESS = r"""
const fs = require('fs');
const scenario = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const src = fs.readFileSync(process.argv[2], 'utf8');
const esc = (s) => String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
function mk(id) {
  const o = { id, _html: '', _t: '', children: [], attrs: {}, style: {}, disabled: false, value: '', title: '', className: '',
    classList: { _s: new Set(), add(...c){c.forEach(x=>this._s.add(x))}, remove(...c){c.forEach(x=>this._s.delete(x))},
      toggle(c,f){ f?this._s.add(c):this._s.delete(c) }, contains(c){ return this._s.has(c) } },
    setAttribute(k,v){ this.attrs[k]=v }, getAttribute(k){ return this.attrs[k] ?? null },
    addEventListener(){}, appendChild(c){ this.children.push(c); return c }, querySelector(){ return null }, closest(){ return null } };
  Object.defineProperty(o, 'textContent', { get(){ return this._t }, set(v){ this._t = v; this._html = esc(v ?? '') } });
  Object.defineProperty(o, 'innerHTML', { get(){ return this._html }, set(v){ this._html = v; if (v === '') this.children = [] } });
  return o;
}
const els = {};
global.document = {
  getElementById: (id) => (els[id] ||= mk(id)),
  createElement: () => mk('x'),
  querySelectorAll: () => [],
};
const store = { iw_token: 't', iw_principal: 'op', iw_groups: '["eval-operator"]' };
global.sessionStorage = { getItem: (k) => store[k] ?? null, setItem(k,v){ store[k]=v }, removeItem(k){ delete store[k] } };
global.location = { pathname: '/console/system' };
global.window = { confirm: () => false };
global.setInterval = () => 1; global.clearInterval = () => {}; global.setTimeout = () => 1;
global.fetch = async (url, opts) => {
  const path = url.replace('/api', '');
  const key = Object.keys(scenario.routes).find((k) => path === k || path.startsWith(k + '?'));
  const r = scenario.routes[key] || { status: 404, body: {} };
  return { status: r.status ?? 200, ok: (r.status ?? 200) < 400, json: async () => JSON.parse(JSON.stringify(r.body)) };
};
(0, eval)(src);
(async () => {
  for (let i = 0; i < 40; i++) await new Promise((r) => setImmediate(r));
  const g = (id) => els[id];
  console.log(JSON.stringify({
    chart: g('historyChart')?._html ?? null,
    historyHidden: g('historyCard')?.classList.contains('hidden'),
    profileHidden: g('profileCard')?.classList.contains('hidden'),
    profiles: (g('profileCards')?.children ?? []).map((c) => ({ html: c._html, disabled: c.disabled, name: c.attrs['data-profile'], cls: c.className })),
    yieldHidden: g('yieldCard')?.classList.contains('hidden'),
    banner: g('yieldBanner')?._t, bannerHidden: g('yieldBanner')?.classList.contains('hidden'),
    events: (g('eventList')?.children ?? []).map((c) => c._html).join('') || g('eventList')?._html, info: g('yieldInfo')?._t, split: g('vramSplit')?._t,
    capWarn: g('capWarn')?._t, capWarnHidden: g('capWarn')?.classList.contains('hidden'), btnOnDisabled: g('btnOn')?.disabled,
  }));
})();
"""

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _run(tmp_path: Path, routes: dict[str, Any]) -> dict[str, Any]:
    js = re.search(r"<script>(.*)</script>", HTML, re.S)
    assert js is not None
    (tmp_path / "page.js").write_text(js.group(1))
    (tmp_path / "harness.js").write_text(HARNESS)
    (tmp_path / "scenario.json").write_text(json.dumps({"routes": routes}))
    out = subprocess.run(
        [
            "node",
            str(tmp_path / "harness.js"),
            str(tmp_path / "page.js"),
            str(tmp_path / "scenario.json"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])  # type: ignore[no-any-return]


def _status(**over: Any) -> dict[str, Any]:
    import time

    now = time.time()
    base = {
        "enabled": True,
        "available": True,
        "state": "partial",
        "mode": "serving",
        "mode_switch_enabled": True,
        "components": [
            {
                "key": "generator",
                "label": "Generator",
                "state": "off",
                "containers": [],
                "idle_unload": True,
            },
            {
                "key": "embedder",
                "label": "Embedder",
                "state": "on",
                "containers": [],
                "idle_unload": False,
            },
        ],
        "vram": {"available": True, "used_mb": 8000, "free_mb": 24000, "total_mb": 32000},
        "estimated_held_mb": 4096,
        "estimated_other_gb": 3.7,
        "profile": "normal",
        "yield": {
            "enabled": True,
            "threshold_gb": 2.0,
            "sustain_s": 20.0,
            "margin_gb": 3.0,
            "recent_events": [
                {
                    "ts": now - 60,
                    "kind": "yield",
                    "free_mb": 1500,
                    "components": ["generator", "judges"],
                    "detail": "x",
                },
                {
                    "ts": now - 30,
                    "kind": "warm_refused",
                    "free_mb": 1400,
                    "components": ["generator"],
                    "detail": "<b>y</b>",
                },
            ],
        },
    }
    base.update(over)
    return base


def _routes(**over: Any) -> dict[str, Any]:
    import time

    now = time.time()
    samples = [
        {
            "ts": now - 600 + i * 30,
            "used_mb": 8000 + i * 100,
            "free_mb": 24000 - i * 100,
            "total_mb": 32000,
        }
        for i in range(20)
    ]
    routes = {
        "/system/status": {"body": _status()},
        "/system/vram": {
            "body": {"available": True, "used_mb": 8000, "free_mb": 24000, "total_mb": 32000}
        },
        "/system/profiles": {
            "body": {
                "enabled": True,
                "active": "normal",
                "profiles": [
                    {
                        "name": "normal",
                        "label": "Normal",
                        "description": "all",
                        "budget_gb": 29.0,
                        "active": True,
                    },
                    {
                        "name": "yield",
                        "label": "Yield GPU",
                        "description": "<script>x</script>",
                        "budget_gb": 4.0,
                        "active": False,
                    },
                ],
            }
        },
        "/system/history": {
            "body": {
                "available": True,
                "samples": samples,
                "interval_s": 5,
                "events": [{"ts": now - 300, "kind": "yield", "components": ["generator"]}],
            }
        },
        "/system/capacity": {
            "body": {
                "ok": False,
                "needed_mb": 18432,
                "margin_mb": 3072,
                "free_mb": 3072,
                "shortfall_mb": 18432,
            }
        },
    }
    routes.update(over)
    return routes


@needs_node
def test_profiles_render_with_budgets_active_state_and_escaping(tmp_path: Path) -> None:
    out = _run(tmp_path, _routes())
    assert out["profileHidden"] is False
    by = {p["name"]: p for p in out["profiles"]}
    assert "Budget ~29 GB" in by["normal"]["html"] and by["normal"]["disabled"] is True
    assert by["yield"]["disabled"] is False and "Budget ~4 GB" in by["yield"]["html"]
    assert "<script>" not in by["yield"]["html"] and "&lt;script&gt;" in by["yield"]["html"]


@needs_node
def test_yield_banner_events_and_split(tmp_path: Path) -> None:
    out = _run(tmp_path, _routes())
    assert out["yieldHidden"] is False and out["bannerHidden"] is False
    assert "released generator, judges" in out["banner"] and "1.5 GB" in out["banner"]
    assert "Released memory" in out["events"] and "Re-warm refused" in out["events"]
    assert "&lt;b&gt;y&lt;/b&gt;" in out["events"] and "<b>y</b>" not in out["events"]
    assert "below 2 GB for 20 s" in out["info"]
    assert "iw-rag ≈ 4.0 GB" in out["split"] and "other processes ≈ 3.7 GB" in out["split"]
    assert "estimated" in out["split"]


@needs_node
def test_history_chart_draws_series_axis_and_event_marker(tmp_path: Path) -> None:
    out = _run(tmp_path, _routes())
    chart = out["chart"]
    assert out["historyHidden"] is False
    assert chart.startswith('<svg class="hist"') and 'role="img"' in chart
    assert (
        'class="used-line"' in chart
        and 'class="free-line"' in chart
        and 'class="used-area"' in chart
    )
    assert chart.count('class="ev-line"') == 1 and "released generator" in chart
    assert "31.3 GB" in chart  # axis ceiling = 32000 MiB
    assert "latest:" in chart


@needs_node
def test_history_empty_state(tmp_path: Path) -> None:
    out = _run(
        tmp_path,
        _routes(**{"/system/history": {"body": {"available": True, "samples": [], "events": []}}}),
    )
    assert "No data yet" in out["chart"] and "<svg" not in out["chart"]


@needs_node
def test_history_unavailable_state(tmp_path: Path) -> None:
    out = _run(
        tmp_path,
        _routes(**{"/system/history": {"body": {"available": False, "samples": [], "events": []}}}),
    )
    assert out["chart"].count("History unavailable") == 1


@needs_node
def test_master_capacity_warning_disables_power_on(tmp_path: Path) -> None:
    out = _run(tmp_path, _routes())
    assert out["capWarnHidden"] is False
    assert "Power ON would not fit" in out["capWarn"] and "short by 18.0 GB" in out["capWarn"]
    assert out["btnOnDisabled"] is True


@needs_node
def test_capacity_unavailable_note_does_not_block(tmp_path: Path) -> None:
    out = _run(
        tmp_path,
        _routes(**{"/system/capacity": {"body": {"ok": True, "skipped": "nvml_unavailable"}}}),
    )
    assert "Capacity check unavailable" in out["capWarn"] and out["btnOnDisabled"] is False


@needs_node
def test_disabled_power_control_hides_gpu_cards(tmp_path: Path) -> None:
    out = _run(
        tmp_path,
        _routes(
            **{
                "/system/status": {
                    "body": {"enabled": False, "available": False, "components": []}
                },
                "/system/profiles": {"body": {"enabled": False, "active": None, "profiles": []}},
            }
        ),
    )
    assert (
        out["yieldHidden"] is True and out["profileHidden"] is True and out["historyHidden"] is True
    )


@needs_node
def test_page_script_passes_node_syntax_check(tmp_path: Path) -> None:
    js = re.search(r"<script>(.*)</script>", HTML, re.S)
    assert js is not None
    f = tmp_path / "p.js"
    f.write_text(js.group(1))
    assert subprocess.run(["node", "--check", str(f)], timeout=30).returncode == 0

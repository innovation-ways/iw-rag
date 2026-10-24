# ruff: noqa: E501
"""Styles shared by every Learn lab (the lab box, its controls and headings)."""

from __future__ import annotations

BASE_CSS = """
  .lab { border: 1px solid var(--teal-deep); border-radius: var(--radius); padding: 18px; margin: 20px 0; background: rgba(45,212,191,0.04); }
  .lab h2.lab-title { margin: 0 0 4px; font-size: 16px; color: var(--teal); }
  .lab .lab-sub { margin: 0 0 14px; color: var(--muted); font-size: 14px; max-width: 76ch; line-height: 1.55; }
  .lab-ctl { display: flex; flex-wrap: wrap; gap: 14px 28px; align-items: flex-end; margin-bottom: 14px; }
  .lab-ctl .grp > .lbl { display: block; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 6px; }
  .seg-btn { display: inline-flex; gap: 0; border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }
  .seg-btn button { border: none; border-radius: 0; background: var(--ink-2); padding: 7px 12px; font-size: 13px; }
  .seg-btn button[aria-pressed="true"] { background: var(--teal-deep); color: #06231f; font-weight: 600; }
  .preset-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
  .lab h3 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin: 16px 0 8px; }
  .lab-err { color: var(--danger); font-size: 14px; }
"""

__all__ = ["BASE_CSS"]

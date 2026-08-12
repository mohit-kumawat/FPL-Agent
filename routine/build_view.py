#!/usr/bin/env python3
"""Build routine/view.html from memory/briefs/*.md.

One self-contained file, no dependencies, no server. Open it and every brief is
there, newest first, with the run history alongside. Regenerated after each
scheduled run; also fine to run by hand:

    python3 routine/build_view.py && open routine/view.html

The markdown here is deliberately a subset -- what the brief templates in
routine/gw1_prompt.md and routine/weekly_prompt.md produce, hardened for
model-written drift. It is not a general-purpose renderer.
"""
from __future__ import annotations

import html
import json
import re
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
BRIEFS = PROJECT / "memory" / "briefs"
LEDGER = PROJECT / "routine" / "logs" / "runs.jsonl"
OUT = PROJECT / "routine" / "view.html"

_CODE_SPAN = re.compile(r"`([^`]+)`")
# One level of balanced parens, so stats and wiki URLs survive. A bare [^)]+
# stops at the first ")" and yields a *wrong* href that still looks like a
# working citation -- worse than dropping the link.
_LINK = re.compile(r"\[([^\]]+)\]\(((?:[^()\s]|\([^()\s]*\))+)\)")
_SAFE_URL = re.compile(r"https?://", re.I)
_STASHED = re.compile(r"\x00(\d+)\x00")

_INLINE = (
    (re.compile(r"\*\*([^*]+)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"(?<![*\w])\*([^*\n]+)\*(?!\*)"), r"<em>\1</em>"),
)


def _anchor(m: re.Match) -> str:
    """Link only http(s); anything else keeps its label and loses the href.

    Briefs quote scraped web content, so a `javascript:` or `data:` URL would
    otherwise become a live script trigger in a page opened over file://.
    """
    text, url = m.group(1), m.group(2)
    if not _SAFE_URL.match(url):
        return text
    return f'<a href="{url}" rel="noopener noreferrer" target="_blank">{text}</a>'


def inline(text: str) -> str:
    out = html.escape(text)
    # Stash code spans before any other substitution: markup inside backticks
    # must stay literal, or quoting a suspicious URL in a brief makes it live.
    spans: list[str] = []

    def stash(m: re.Match) -> str:
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    out = _CODE_SPAN.sub(stash, out)
    for pattern, repl in _INLINE:
        out = pattern.sub(repl, out)
    out = _LINK.sub(_anchor, out)
    return _STASHED.sub(lambda m: f"<code>{spans[int(m.group(1))]}</code>", out)


_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def md_to_html(src: str) -> str:
    """Convert the brief subset: headings, lists, tables, quotes, code, paras.

    Briefs are model-written from a soft template, so this has to survive
    constructs the template never shows. Lists stay flat by design -- nesting
    <ul> directly inside <ul> is invalid HTML, and an indent class carries the
    visual instead.
    """
    lines = src.splitlines()
    out: list[str] = []
    para: list[str] = []
    in_list = False
    i = 0

    def flush_para() -> None:
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    while i < len(lines):
        line = lines[i].rstrip()

        if line.strip().startswith("```"):
            flush_para(); close_list()
            buf: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            out.append("<pre><code>" + html.escape("\n".join(buf)) + "</code></pre>")
            i += 1
            continue

        if not line.strip():
            flush_para(); close_list(); i += 1
            continue

        # Tables: the shape a model reaches for unprompted (price changes, a
        # transfer comparison). With no branch they fall through to prose and
        # render as one run-on paragraph of literal pipes.
        if _TABLE_ROW.match(line) and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            flush_para(); close_list()
            head = "".join(f"<th>{inline(c)}</th>" for c in _cells(line))
            i += 2
            body: list[str] = []
            while i < len(lines) and _TABLE_ROW.match(lines[i]):
                cells = "".join(f"<td>{inline(c)}</td>" for c in _cells(lines[i]))
                body.append(f"<tr>{cells}</tr>")
                i += 1
            out.append(
                f"<div class=scroll><table><thead><tr>{head}</tr></thead>"
                f"<tbody>{''.join(body)}</tbody></table></div>"
            )
            continue

        if re.fullmatch(r"\s*([-*_])\1{2,}\s*", line):
            flush_para(); close_list()
            out.append("<hr>"); i += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            flush_para(); close_list()
            level = len(heading.group(1))
            out.append(f"<h{level}>{inline(heading.group(2))}</h{level}>")
            i += 1
            continue

        if line.lstrip().startswith("> "):
            flush_para(); close_list()
            out.append(f"<blockquote>{inline(line.lstrip()[2:])}</blockquote>")
            i += 1
            continue

        # A "**Key:** value" line is its own block. The brief template stacks
        # Status and Deadline on adjacent lines with no blank between them, and
        # joining them into one paragraph runs the two most-scanned facts in the
        # document together.
        if re.match(r"^\*\*[^*]+:\*\*", line.strip()):
            flush_para(); close_list()
            out.append(f"<p>{inline(line.strip())}</p>")
            i += 1
            continue

        bullet = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", line)
        if bullet:
            flush_para()
            if not in_list:
                out.append("<ul>"); in_list = True
            indent, _, body = bullet.groups()
            classes = ["sub"] if len(indent) >= 2 else []
            box = re.match(r"^\[([ xX])\]\s+(.*)$", body)
            if box:
                if box.group(1).lower() == "x":
                    classes.append("done")
                    mark = "checked"
                else:
                    mark = ""
                inner = f'<input type="checkbox" disabled {mark}> {inline(box.group(2))}'
            else:
                inner = inline(body)
            cls = f' class="{" ".join(classes)}"' if classes else ""
            out.append(f"<li{cls}>{inner}</li>")
            i += 1
            continue

        para.append(line.strip())
        i += 1

    flush_para(); close_list()
    return "\n".join(out)


_STATUSES = ("quiet", "action needed", "blocked")


def status_of(src: str) -> str:
    """Read the status from the leading token, exactly -- never by substring.

    Searching for substrings turns "quiet, no action needed" amber and the
    template's own "quiet | action needed | blocked" menu red, which is the
    signal the README promises means the routine broke.
    """
    m = re.search(r"^\*\*Status:\*\*\s*(.+)$", src, re.MULTILINE)
    if not m:
        return "unknown"
    value = m.group(1).strip().lower()
    leading = re.split(r"\s*[|,;.]|\s+[—–-]\s+", value)[0]
    for candidate in (value, leading):
        norm = candidate.strip().strip("*").replace("-", " ").strip()
        if norm in _STATUSES:
            return norm.replace(" ", "-")
    return "unknown"


def verdict_of(src: str) -> str:
    m = re.search(r"^#\s+(.*)$", src, re.MULTILINE)
    if not m:
        return ""
    title = m.group(1)
    return title.split("—", 1)[1].strip() if "—" in title else title.strip()


_FAILED_STATUSES = (
    "failed", "no-brief", "no-credentials", "no-prompt", "bad-credentials",
)


def _slug(name: str) -> str:
    """A DOM-safe anchor from a brief filename.

    The filename comes from a directory the agent writes to, so it is untrusted:
    interpolated raw it can close the attribute and inject a handler, and even a
    benign quote or space breaks the nav/article pairing so the day renders
    blank. Escaping alone would fix the injection but still leave ids that the
    JS and the URL fragment can't round-trip.
    """
    return re.sub(r"[^A-Za-z0-9_-]", "-", name)


def load_ledger() -> dict[str, dict]:
    """Per date: the last row, plus whether any run that date succeeded.

    `ok_seen` matters because the two sources answer different questions. The
    brief's own status describes the day's football; the ledger describes
    whether a run worked. Once any run has produced a real brief, a later crash
    must not repaint it -- otherwise a failed 13:00 check marks a valid morning
    recommendation "blocked" hours before the deadline.
    """
    runs: dict[str, dict] = {}
    if not LEDGER.exists():
        return runs
    for line in LEDGER.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Briefs are now one-per-window (<date>-<window>.md) and the ledger
        # carries the stem in `brief`; older rows only have `date`. Key on
        # whichever exists so both generations of ledger render.
        key = row.get("brief") or row.get("date")
        if not key:
            continue
        previous = runs.get(key)
        row["ok_seen"] = row.get("status") == "ok" or bool(
            previous and previous.get("ok_seen")
        )
        runs[key] = row                   # last write for a stem wins
    return runs


CSS = """
*,*::before,*::after{box-sizing:border-box}
:root{
  --bg:#fbfaf9; --panel:#fff; --ink:#1c1a17; --muted:#6b6560; --line:#e6e2dd;
  --accent:#0b6b53; --quiet:#8a9199; --action:#b4690e; --blocked:#c0392b;
  --code:#f3f1ee;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#16181a; --panel:#1d2022; --ink:#e8e6e3; --muted:#9aa0a6; --line:#2c3033;
  --accent:#4ec9a5; --quiet:#7d858c; --action:#e0a458; --blocked:#e06c5f;
  --code:#24282b;
}}
:root[data-theme=dark]{
  --bg:#16181a; --panel:#1d2022; --ink:#e8e6e3; --muted:#9aa0a6; --line:#2c3033;
  --accent:#4ec9a5; --quiet:#7d858c; --action:#e0a458; --blocked:#e06c5f;
  --code:#24282b;
}
:root[data-theme=light]{
  --bg:#fbfaf9; --panel:#fff; --ink:#1c1a17; --muted:#6b6560; --line:#e6e2dd;
  --accent:#0b6b53; --quiet:#8a9199; --action:#b4690e; --blocked:#c0392b;
  --code:#f3f1ee;
}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:32px 20px 64px}
header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:6px}
h1{font-size:19px;margin:0;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13px}
.toggle{margin-left:auto;background:none;border:1px solid var(--line);color:var(--muted);
  border-radius:6px;padding:4px 10px;font-size:12px;cursor:pointer;font-family:inherit}
.toggle:hover{color:var(--ink);border-color:var(--muted)}
.layout{display:grid;grid-template-columns:212px 1fr;gap:26px;margin-top:24px;align-items:start}
@media(max-width:760px){.layout{grid-template-columns:1fr}nav{position:static!important;max-height:none!important}}
nav{position:sticky;top:20px;max-height:calc(100vh - 48px);overflow-y:auto;
  border-right:1px solid var(--line);padding-right:14px}
@media(max-width:760px){nav{border-right:0;border-bottom:1px solid var(--line);padding:0 0 12px}}
.day{display:flex;align-items:center;gap:9px;width:100%;background:none;border:0;
  color:var(--muted);font:inherit;font-size:13px;text-align:left;padding:7px 9px;
  border-radius:6px;cursor:pointer}
.day:hover{background:var(--code);color:var(--ink)}
.day[aria-current=true]{background:var(--code);color:var(--ink);font-weight:600}
.dot{width:7px;height:7px;border-radius:50%;flex:0 0 auto;background:var(--quiet)}
.dot.action-needed{background:var(--action)} .dot.blocked{background:var(--blocked)}
.dot.quiet{background:var(--accent)}
article{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:26px 30px}
article[hidden]{display:none}
.meta{display:flex;gap:8px;align-items:center;flex-wrap:wrap;color:var(--muted);
  font-size:12px;margin-bottom:18px;padding-bottom:14px;border-bottom:1px solid var(--line)}
.pill{border:1px solid var(--line);border-radius:999px;padding:2px 9px;font-size:11px;
  text-transform:uppercase;letter-spacing:.05em}
.pill.quiet{color:var(--accent);border-color:var(--accent)}
.pill.action-needed{color:var(--action);border-color:var(--action)}
.pill.blocked{color:var(--blocked);border-color:var(--blocked)}
.warn{color:var(--action)}
article h1{font-size:20px;margin:0 0 4px}
article h2{font-size:13px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
  margin:26px 0 8px;font-weight:600}
article h3{font-size:15px;margin:20px 0 6px}
article p{margin:0 0 11px}
article ul,article ol{margin:0 0 11px;padding-left:20px}
article li{margin:3px 0}
article li.sub{margin-left:18px;list-style-type:circle}
article li.done{color:var(--muted);text-decoration:line-through}
.scroll{overflow-x:auto;margin:0 0 12px}
table{border-collapse:collapse;font-size:13.5px;min-width:100%}
th,td{border-bottom:1px solid var(--line);padding:6px 12px 6px 0;text-align:left;
  white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:11.5px;text-transform:uppercase;
  letter-spacing:.05em}
article a{color:var(--accent)}
code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:.88em;
  font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--code);padding:13px 15px;border-radius:8px;overflow-x:auto;margin:0 0 12px}
pre code{background:none;padding:0;font-size:12.5px;line-height:1.55}
blockquote{margin:0 0 12px;padding:8px 14px;border-left:3px solid var(--action);
  background:var(--code);border-radius:0 6px 6px 0;color:var(--muted)}
hr{border:0;border-top:1px solid var(--line);margin:22px 0}
.empty{color:var(--muted);text-align:center;padding:48px 20px}
"""

JS = """
const days=[...document.querySelectorAll('.day')];
const shown=[...document.querySelectorAll('article')];
function pick(id){
  shown.forEach(a=>a.hidden=a.id!==id);
  days.forEach(b=>b.setAttribute('aria-current',String(b.dataset.target===id)));
  history.replaceState(null,'','#'+id);
}
days.forEach(b=>b.addEventListener('click',()=>pick(b.dataset.target)));
const first=location.hash.slice(1);
if(shown.length)pick(shown.some(a=>a.id===first)?first:shown[0].id);
const root=document.documentElement, btn=document.querySelector('.toggle');
btn&&btn.addEventListener('click',()=>{
  const dark=root.getAttribute('data-theme')==='dark'
    ||(!root.hasAttribute('data-theme')&&matchMedia('(prefers-color-scheme:dark)').matches);
  root.setAttribute('data-theme',dark?'light':'dark');
});
"""


def build() -> str:
    files = sorted(BRIEFS.glob("*.md"), reverse=True) if BRIEFS.exists() else []
    runs = load_ledger()
    generated = datetime.now().strftime("%d %b %Y, %H:%M")

    if not files:
        body = (
            '<div class="empty"><p>No briefs yet.</p>'
            "<p>The scheduled run writes one to <code>memory/briefs/</code> each day.</p>"
            "<p>Run it now with <code>FPL_ROUTINE_FORCE=1 ./routine/run.sh</code>.</p></div>"
        )
        nav = ""
    else:
        nav_items, articles = [], []
        for path in files:
            src = path.read_text()
            date = path.stem
            anchor = f"d-{_slug(date)}"
            state = status_of(src)
            run = runs.get(date, {})
            secs = run.get("seconds")
            timing = f"{secs}s" if isinstance(secs, (int, float)) else ""
            failed = run.get("status") in _FAILED_STATUSES
            if failed and not run.get("ok_seen"):
                state = "blocked"
            # Stems are `<date>` (legacy) or `<date>-<window>`.
            try:
                label = datetime.strptime(date[:10], "%Y-%m-%d").strftime("%a %d %b")
                if len(date) > 11:
                    label += f" · {date[11:]}"
            except ValueError:
                label = date
            nav_items.append(
                f'<button class="day" data-target="{anchor}" aria-current="false">'
                f'<span class="dot {state}"></span>{html.escape(label)}</button>'
            )
            bits = [f'<span class="pill {state}">{state.replace("-", " ")}</span>',
                    f"<span>{html.escape(date)}</span>"]
            if timing:
                bits.append(f"<span>ran in {timing}</span>")
            verdict = verdict_of(src)
            if verdict:
                bits.append(f"<span>{html.escape(verdict)}</span>")
            if failed and run.get("ok_seen"):
                # The brief itself is real; a later check that day died. Say so
                # without discrediting what the successful run produced.
                bits.append('<span class="warn">a later check failed</span>')
            articles.append(
                f'<article id="{anchor}" hidden>'
                f'<div class="meta">{"".join(bits)}</div>{md_to_html(src)}</article>'
            )
        nav = f"<nav>{''.join(nav_items)}</nav>"
        body = "".join(articles)

    count = f"{len(files)} brief{'s' if len(files) != 1 else ''}"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FPL Agent — daily briefs</title><style>{CSS}</style></head>
<body><div class="wrap">
<header><h1>FPL Agent</h1>
<span class="sub">{count} · rebuilt {html.escape(generated)}</span>
<button class="toggle" type="button">theme</button></header>
<div class="layout">{nav}<div>{body}</div></div>
</div><script>{JS}</script></body></html>"""


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT.relative_to(PROJECT)}")

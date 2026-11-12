"""Convert a generated Markdown page to a standalone HTML page for meetings (local, not committed).

Python-Markdown with the tables and fenced-code extensions, plus a small preprocessor that turns
```mermaid fences into <pre class="mermaid"> blocks for mermaid.js. Inline CSS with light and dark themes
(prefers-color-scheme). The only external URL is the pinned mermaid script: tables read offline, the
diagram needs the network.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor

MERMAID_URL = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"
_FENCE = re.compile(r"^```mermaid\n(.*?)^```\s*$", re.MULTILINE | re.DOTALL)

CSS = """
:root { --bg: #ffffff; --fg: #1f2328; --muted: #59636e; --border: #d1d9e0; --stripe: #f6f8fa; --code: #eff1f3; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1; --border: #3d444d; --stripe: #151b23; --code: #262c36; }
}
body { background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
       max-width: 1400px; margin: 0 auto; padding: 24px 16px; }
h1, h2 { border-bottom: 1px solid var(--border); padding-bottom: .3em; }
table { border-collapse: collapse; margin: 12px 0; display: block; overflow-x: auto; font-size: 13px; }
th, td { border: 1px solid var(--border); padding: 4px 8px; }
tr:nth-child(even) td { background: var(--stripe); }
code { background: var(--code); padding: 1px 4px; border-radius: 4px; font-size: 90%; }
pre.mermaid { background: transparent; text-align: center; }
a { color: #4493f8; }
"""


class _MermaidPre(Preprocessor):
    def run(self, lines: list[str]) -> list[str]:
        text = "\n".join(lines)

        def repl(mo: re.Match) -> str:
            block = f'<pre class="mermaid">\n{html.escape(mo.group(1))}</pre>'
            return self.md.htmlStash.store(block)

        return _FENCE.sub(repl, text).split("\n")


class MermaidExtension(Extension):
    def extendMarkdown(self, md):
        # after normalize_whitespace (30), which strips the stash markers, and before fenced_code (25)
        md.preprocessors.register(_MermaidPre(md), "mermaid_pre", 29)


def convert(md_text: str, title: str) -> str:
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", MermaidExtension()])
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{html.escape(title)}</title>\n<style>{CSS}</style>\n</head>\n<body>\n{body}\n"
        f"<script src=\"{MERMAID_URL}\"></script>\n"
        "<script>mermaid.initialize({startOnLoad: true, theme: "
        "matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'default'});</script>\n"
        "</body>\n</html>\n"
    )


def write(md_path: Path, html_path: Path, title: str) -> None:
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(convert(md_path.read_text(), title))

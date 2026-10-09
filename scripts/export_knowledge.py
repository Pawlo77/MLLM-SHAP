"""Export vault notes to plain Markdown for the Sphinx site (``docs/handbook/``).

Usage:
    python scripts/export_knowledge.py [--out DIR]

Copies ``knowledge/{handbook,architecture,decisions,concepts}/*.md`` and rewrites
wikilinks, embeds, callouts, Dataview and Mermaid blocks into Markdown that Sphinx
renders. The Roadmap and decision-log sections that name a plan phase are not exported.
The output folders are rebuilt on every run (``docs/Makefile`` runs it before builds).

@domain tooling
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_vault import Vault

ROOT = Path(__file__).resolve().parent.parent
FOLDERS = ("handbook", "architecture", "decisions", "concepts")
REPOSITORY = "https://github.com/Pawlo77/nlp-shap/blob/main"
LINK = re.compile(r"(!?)\[\[([^\]|#\\]*)(#[^\]|\\]*)?(?:\\?\|([^\]]*))?\]\]")
CALLOUT = re.compile(r"^(\s*>\s*)\[!(\w+)\][+-]?\s*(.*)$")
FENCE = re.compile(r"^\s*```(\w*)")
UNEXPORTED = {("handbook", "Roadmap")}
"""(folder, note) pairs kept out of the site: internal plan phases."""
PLAN_PHASE = re.compile(r"\bPhases? \d")
"""A plan phase number; decision-log sections that mention one are not exported."""


def slug(name: str) -> str:
    """Return a lowercase, hyphenated file stem for ``name``."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def anchor(heading: str) -> str:
    """Return the MyST heading anchor of ``heading`` (``myst_heading_anchors``)."""
    return re.sub(r"[^\w\- ]", "", heading.lower().replace(" ", "-"))


def site_text(folder: str, note: Path) -> str:
    """Return the note text for the site (decision logs lose plan-phase sections)."""
    text = note.read_text(encoding="utf-8")
    if folder != "decisions":
        return text
    head, *sections = re.split(r"(?m)^(?=## )", text)
    return head + "".join(part for part in sections if not PLAN_PHASE.search(part))


def body_of(text: str) -> str:
    """Return the note text without frontmatter and without its first H1."""
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)
    return re.sub(r"\A\s*# [^\n]*\n", "", text, count=1).strip("\n") + "\n"


class Exporter:
    """Rewrite vault notes to portable Markdown."""

    def __init__(self, root: Path, out: Path) -> None:
        """Load the vault at ``root`` and write exported notes under ``out``."""
        self.root = root
        self.out = out
        self.vault = Vault.load(root)
        self.exported: dict[Path, Path] = {}
        self.texts: dict[Path, str] = {}
        for folder in FOLDERS:
            for note in sorted((root / "knowledge" / folder).glob("*.md")):
                if (folder, note.stem) not in UNEXPORTED:
                    self.exported[note] = out / folder / f"{slug(note.stem)}.md"
                    self.texts[note] = site_text(folder, note)

    def link(self, match: re.Match[str], source_out: Path, depth: int) -> str:
        """Return the Markdown replacement for one wikilink or embed."""
        embed, target, heading, alias = match.groups()
        target = target.strip()
        if not target:
            return (alias or (heading or "").lstrip("#")).strip()
        resolved = self.vault.resolve(target)
        text = (alias or Path(target).stem or target).strip()
        if embed and resolved is not None and resolved.suffix == ".md":
            if depth > 2:
                return text
            inner = body_of(resolved.read_text(encoding="utf-8"))
            return self.convert(inner, source_out, depth + 1)
        if resolved is None:
            return text
        section = (heading or "").lstrip("#").strip()
        kept = re.findall(r"(?m)^##? (.+?)\s*$", self.texts.get(resolved, ""))
        if resolved in self.exported and (not section or section in kept):
            relative = Path(
                *[".."] * (len(source_out.relative_to(self.out).parts) - 1),
                self.exported[resolved].relative_to(self.out),
            )
            fragment = f"#{anchor(section)}" if section else ""
            return f"[{text}]({relative.as_posix()}{fragment})"
        path = resolved.relative_to(self.root).as_posix().replace(" ", "%20")
        return f"[{text}]({REPOSITORY}/{path})"

    def convert(self, text: str, source_out: Path, depth: int = 0) -> str:
        """Return ``text`` with Obsidian syntax rewritten."""
        lines: list[str] = []
        fence: str | None = None
        for line in text.splitlines():
            opening = FENCE.match(line)
            if fence is not None:
                if line.strip().startswith("```"):
                    if fence != "dataview":
                        lines.append(line)
                    fence = None
                elif fence != "dataview":
                    lines.append(line)
                continue
            if opening:
                fence = opening.group(1) or "text"
                if fence == "dataview":
                    lines.append(
                        "*Live Dataview table: open the vault in Obsidian to see it.*"
                    )
                elif fence == "mermaid":
                    lines.append(line.replace("mermaid", "text", 1))
                else:
                    lines.append(line)
                continue
            callout = CALLOUT.match(line)
            if callout:
                prefix, kind, title = callout.groups()
                label = f"**{kind.capitalize()}:**"
                line = f"{prefix}{label} {title}".rstrip()
            line = LINK.sub(lambda m: self.link(m, source_out, depth), line)
            lines.append(line)
        return "\n".join(lines) + "\n"

    def run(self) -> int:
        """Write every exported note; return the count."""
        for folder in FOLDERS:
            shutil.rmtree(self.out / folder, ignore_errors=True)
        for note, target in self.exported.items():
            text = self.texts[note]
            title_match = re.search(r"(?m)^# (.+)$", text)
            title = title_match.group(1) if title_match else note.stem
            content = f"# {title}\n\n" + self.convert(body_of(text), target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return len(self.exported)


def main(argv: list[str] | None = None) -> int:
    """Export the notes and print a summary line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "handbook")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    count = Exporter(args.root.resolve(), args.out.resolve()).run()
    print(f"export_knowledge: {count} notes written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# rag-managed: true
"""Chunking strategies: semantic, fixed-size, and hybrid."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ---------------------------------------------------------------------------
# Lightweight token estimator (avoids tiktoken dependency).
# Uses the ~4 chars/token heuristic.  Good enough for budget enforcement;
# the assembly stage does a precise count with the actual tokenizer if
# tiktoken is installed.
# ---------------------------------------------------------------------------
CHARS_PER_TOKEN = 4

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_enc.encode(text, disallowed_special=()))
except ImportError:
    def count_tokens(text: str) -> int:
        return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class Chunk:
    id: str
    content: str
    source_file: str
    start_line: int
    end_line: int
    token_count: int
    content_hash: str
    chunk_type: str = "code"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_file": self.source_file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "token_count": self.token_count,
            "content_hash": self.content_hash,
            "chunk_type": self.chunk_type,
            **self.metadata,
        }


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _make_chunk(
    content: str,
    source_file: str,
    start_line: int,
    end_line: int,
    chunk_type: str = "code",
    metadata: dict | None = None,
) -> Chunk:
    tokens = count_tokens(content)
    cid = f"{source_file}:{start_line}-{end_line}:{_hash(content)}"
    return Chunk(
        id=cid,
        content=content,
        source_file=source_file,
        start_line=start_line,
        end_line=end_line,
        token_count=tokens,
        content_hash=_hash(content),
        chunk_type=chunk_type,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Semantic chunking — splits on structural boundaries
# ---------------------------------------------------------------------------

# Markdown: split on headings
_MD_HEADING = re.compile(r"^(#{1,4})\s+", re.MULTILINE)

# Python: split on top-level def / class
_PY_TOPLEVEL = re.compile(r"^(class |def |async def )", re.MULTILINE)

# JS/TS: split on function, class, export, const/let arrow fns
_JS_TOPLEVEL = re.compile(
    r"^(export\s+)?(function |class |const |let |var )",
    re.MULTILINE,
)


def _split_by_regex(
    text: str, pattern: re.Pattern, source_file: str,
    chunk_type: str, max_tokens: int,
) -> list[Chunk]:
    """Split text at regex match positions, merging small sections."""
    lines = text.split("\n")
    positions = [0]
    for m in pattern.finditer(text):
        line_no = text[:m.start()].count("\n")
        if line_no > 0 and line_no not in positions:
            positions.append(line_no)
    positions.append(len(lines))

    chunks: list[Chunk] = []
    buf_lines: list[str] = []
    buf_start = 0

    for i in range(len(positions) - 1):
        section = lines[positions[i]:positions[i + 1]]
        section_text = "\n".join(section)
        section_tokens = count_tokens(section_text)

        # If adding this section keeps us under budget, accumulate
        buf_text = "\n".join(buf_lines)
        if buf_lines and count_tokens(buf_text) + section_tokens > max_tokens:
            # Flush buffer
            if buf_text.strip():
                chunks.append(_make_chunk(
                    buf_text, source_file,
                    buf_start + 1, buf_start + len(buf_lines),
                    chunk_type,
                ))
            buf_lines = section
            buf_start = positions[i]
        else:
            if not buf_lines:
                buf_start = positions[i]
            buf_lines.extend(section)

    # Flush remaining
    buf_text = "\n".join(buf_lines)
    if buf_text.strip():
        chunks.append(_make_chunk(
            buf_text, source_file,
            buf_start + 1, buf_start + len(buf_lines),
            chunk_type,
        ))

    # Split any oversized chunks with fixed-size fallback
    result: list[Chunk] = []
    for c in chunks:
        if c.token_count > max_tokens * 1.5:
            result.extend(_fixed_size_chunk(
                c.content, source_file, c.start_line, max_tokens, 0,
            ))
        else:
            result.append(c)
    return result


def _fixed_size_chunk(
    text: str, source_file: str, base_line: int,
    max_tokens: int, overlap_tokens: int,
) -> list[Chunk]:
    """Fixed-size chunking with optional overlap."""
    lines = text.split("\n")
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_start = 0

    for i, line in enumerate(lines):
        buf.append(line)
        buf_text = "\n".join(buf)
        if count_tokens(buf_text) >= max_tokens:
            chunks.append(_make_chunk(
                buf_text, source_file,
                base_line + buf_start, base_line + i,
                "fixed",
            ))
            # Calculate overlap
            if overlap_tokens > 0:
                overlap_lines: list[str] = []
                for ln in reversed(buf):
                    overlap_lines.insert(0, ln)
                    if count_tokens("\n".join(overlap_lines)) >= overlap_tokens:
                        break
                buf = overlap_lines
                buf_start = i - len(overlap_lines) + 1
            else:
                buf = []
                buf_start = i + 1

    if buf and "\n".join(buf).strip():
        chunks.append(_make_chunk(
            "\n".join(buf), source_file,
            base_line + buf_start, base_line + len(lines) - 1,
            "fixed",
        ))
    return chunks


# ---------------------------------------------------------------------------
# File-type detection and dispatch
# ---------------------------------------------------------------------------

_EXT_MAP: dict[str, tuple[re.Pattern | None, str]] = {
    ".py": (_PY_TOPLEVEL, "python"),
    ".js": (_JS_TOPLEVEL, "javascript"),
    ".ts": (_JS_TOPLEVEL, "typescript"),
    ".tsx": (_JS_TOPLEVEL, "typescript"),
    ".jsx": (_JS_TOPLEVEL, "javascript"),
    ".md": (_MD_HEADING, "markdown"),
    ".mdx": (_MD_HEADING, "markdown"),
    ".rst": (None, "text"),
    ".txt": (None, "text"),
    ".yaml": (None, "config"),
    ".yml": (None, "config"),
    ".json": (None, "config"),
    ".toml": (None, "config"),
    ".cfg": (None, "config"),
    ".ini": (None, "config"),
    ".sh": (None, "shell"),
    ".bash": (None, "shell"),
    ".go": (re.compile(r"^(func |type )", re.MULTILINE), "go"),
    ".rs": (re.compile(r"^(pub )?(fn |struct |enum |impl |mod )", re.MULTILINE), "rust"),
    ".java": (re.compile(r"^(\s*)(public |private |protected )?(static )?(class |interface |enum |void |[\w<>]+\s+\w+\s*\()", re.MULTILINE), "java"),
    ".rb": (re.compile(r"^(class |module |def )", re.MULTILINE), "ruby"),
    ".php": (re.compile(r"^(class |function |public |private |protected )", re.MULTILINE), "php"),
    ".c": (re.compile(r"^[\w\*]+\s+\w+\s*\(", re.MULTILINE), "c"),
    ".h": (re.compile(r"^[\w\*]+\s+\w+\s*\(", re.MULTILINE), "c"),
    ".cpp": (re.compile(r"^[\w\*:]+\s+[\w:]+\s*\(", re.MULTILINE), "cpp"),
    ".cs": (re.compile(r"^(\s*)(public |private |protected |internal )?(static )?(class |interface |void |async |[\w<>]+\s+\w+\s*\()", re.MULTILINE), "csharp"),
    ".swift": (re.compile(r"^(func |class |struct |enum |protocol )", re.MULTILINE), "swift"),
}

# Max file size to index (skip binaries / generated files)
MAX_FILE_BYTES = 512 * 1024  # 512KB


def chunk_file(
    filepath: Path,
    strategy: str = "semantic",
    max_tokens: int = 400,
    overlap_tokens: int = 0,
) -> list[Chunk]:
    """Chunk a single file according to the chosen strategy."""
    if not filepath.is_file():
        return []
    if filepath.stat().st_size > MAX_FILE_BYTES:
        return []

    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []

    if not text.strip():
        return []

    source = str(filepath)
    ext = filepath.suffix.lower()

    # Whole-file indexing for small files
    if count_tokens(text) <= max_tokens:
        ctype = _EXT_MAP.get(ext, (None, "text"))[1]
        return [_make_chunk(text, source, 1, text.count("\n") + 1, ctype)]

    if strategy == "semantic":
        pattern, ctype = _EXT_MAP.get(ext, (None, "text"))
        if pattern is not None:
            return _split_by_regex(text, pattern, source, ctype, max_tokens)
        # Fallback: paragraph splitting
        return _paragraph_chunk(text, source, max_tokens)

    elif strategy == "fixed":
        return _fixed_size_chunk(text, source, 1, max_tokens, overlap_tokens)

    elif strategy == "hybrid":
        # Try semantic first, fall back to fixed for non-structured files
        pattern, ctype = _EXT_MAP.get(ext, (None, "text"))
        if pattern is not None:
            return _split_by_regex(text, pattern, source, ctype, max_tokens)
        return _fixed_size_chunk(text, source, 1, max_tokens, overlap_tokens)

    return _fixed_size_chunk(text, source, 1, max_tokens, overlap_tokens)


def _paragraph_chunk(
    text: str, source_file: str, max_tokens: int,
) -> list[Chunk]:
    """Split on double-newlines (paragraphs), merge small ones."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[Chunk] = []
    buf: list[str] = []
    line_offset = 0

    for para in paragraphs:
        para_lines = para.count("\n") + 1
        if buf and count_tokens("\n\n".join(buf + [para])) > max_tokens:
            merged = "\n\n".join(buf)
            chunks.append(_make_chunk(
                merged, source_file,
                line_offset + 1,
                line_offset + merged.count("\n") + 1,
                "text",
            ))
            line_offset += merged.count("\n") + 2
            buf = [para]
        else:
            buf.append(para)

    if buf:
        merged = "\n\n".join(buf)
        chunks.append(_make_chunk(
            merged, source_file,
            line_offset + 1,
            line_offset + merged.count("\n") + 1,
            "text",
        ))
    return chunks

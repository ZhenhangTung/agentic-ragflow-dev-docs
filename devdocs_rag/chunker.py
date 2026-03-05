"""
DevDocs RAG Framework - Custom chunking strategy for developer API documentation.

This chunker is specifically designed for API reference docs that follow patterns like:
- HTTP endpoint definitions (### Create dataset, POST /api/v1/datasets)
- Python SDK method signatures (DataSet.upload_documents(...))
- Request/Response examples with code blocks
- Parameter tables and descriptions

Strategy:
1. Parse the document into a hierarchy: Section > API Endpoint > Sub-section
2. Each API endpoint becomes a self-contained chunk with its context
3. Code blocks are kept intact with their surrounding description
4. Parameter lists stay with their parent API method
5. Add metadata: doc_name, section_path, api_method, http_verb, endpoint_url
"""
import re
import os
from dataclasses import dataclass, field

from devdocs_rag.config import get_settings


@dataclass
class DocChunk:
    """A chunk of developer documentation with rich metadata."""
    content: str
    doc_name: str
    section_path: str  # e.g. "DATASET MANAGEMENT > Create dataset > Request"
    chunk_type: str  # "api_endpoint", "parameter_list", "code_example", "concept", "overview"
    api_method: str = ""  # e.g. "POST", "GET", "DELETE", "PUT"
    endpoint_url: str = ""  # e.g. "/api/v1/datasets"
    sdk_signature: str = ""  # e.g. "RAGFlow.create_dataset(...)"
    language: str = ""  # "python", "bash", "json"
    chunk_index: int = 0

    def to_indexable_text(self) -> str:
        """Build an enriched text for both embedding and full-text search."""
        parts = [self.content]
        if self.api_method and self.endpoint_url:
            parts.insert(0, f"[{self.api_method} {self.endpoint_url}]")
        if self.sdk_signature:
            parts.insert(0, f"[SDK: {self.sdk_signature}]")
        if self.section_path:
            parts.insert(0, f"[Section: {self.section_path}]")
        return "\n".join(parts)

    def to_metadata(self) -> dict:
        return {
            "doc_name": self.doc_name,
            "section_path": self.section_path,
            "chunk_type": self.chunk_type,
            "api_method": self.api_method,
            "endpoint_url": self.endpoint_url,
            "sdk_signature": self.sdk_signature,
            "language": self.language,
            "chunk_index": self.chunk_index,
        }


# ── Regex patterns for API doc structure ──────────────────────────────────
RE_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)
RE_H2 = re.compile(r"^##\s+(.+)$", re.MULTILINE)
RE_H3 = re.compile(r"^###\s+(.+)$", re.MULTILINE)
RE_H4 = re.compile(r"^####\s+(.+)$", re.MULTILINE)
RE_HTTP_METHOD = re.compile(
    r"\*\*(GET|POST|PUT|DELETE|PATCH)\*\*\s+`([^`]+)`", re.MULTILINE
)
RE_CODE_BLOCK = re.compile(
    r"```(\w*)\s*\n(.*?)```", re.DOTALL
)
RE_FRONTMATTER = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def _build_sdk_signature_pattern(class_names: list[str]) -> re.Pattern:
    """Build regex for SDK method signatures from configurable class names."""
    if not class_names:
        return re.compile(r"(?!)")  # never-match pattern when no classes configured
    names = "|".join(re.escape(n) for n in class_names)
    return re.compile(
        rf"^```python\s*\n((?:{names})\.\w+\([^)]*\).*?)$",
        re.MULTILINE,
    )


def _build_sdk_inline_pattern(class_names: list[str]) -> re.Pattern:
    """Build regex for inline SDK method references."""
    if not class_names:
        return re.compile(r"(?!)")  # never-match pattern
    names = "|".join(re.escape(n) for n in class_names)
    return re.compile(
        rf"`((?:{names})\.\w+\([^)]*\))`",
    )


def _build_sdk_content_pattern(class_names: list[str]) -> re.Pattern:
    """Build regex to detect SDK class references in content."""
    if not class_names:
        return re.compile(r"(?!)")  # never-match pattern
    names = "|".join(re.escape(n) for n in class_names)
    return re.compile(rf"{names}")


def _clean_frontmatter(text: str) -> str:
    """Remove YAML frontmatter from the beginning."""
    return RE_FRONTMATTER.sub("", text, count=1).strip()


def _split_by_headings(text: str, level: int = 2) -> list[tuple[str, str]]:
    """
    Split markdown text by headings of a given level.
    Returns [(heading_text, section_content), ...].
    """
    if level == 2:
        pattern = re.compile(r"^(##\s+.+)$", re.MULTILINE)
    elif level == 3:
        pattern = re.compile(r"^(###\s+.+)$", re.MULTILINE)
    elif level == 4:
        pattern = re.compile(r"^(####\s+.+)$", re.MULTILINE)
    else:
        raise ValueError(f"Unsupported heading level: {level}")

    parts = pattern.split(text)

    sections = []
    # First part before any heading
    if parts[0].strip():
        sections.append(("_preamble", parts[0].strip()))

    # Pair up heading + content
    for i in range(1, len(parts), 2):
        heading = parts[i].lstrip("#").strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append((heading, content))

    return sections


def _extract_http_endpoint(text: str) -> tuple[str, str]:
    """Extract HTTP method and URL from section text."""
    m = RE_HTTP_METHOD.search(text)
    if m:
        return m.group(1), m.group(2)
    return "", ""


def _extract_sdk_signature(text: str, class_names: list[str] | None = None) -> str:
    """Extract Python SDK method signature using configurable class names."""
    if class_names is None:
        class_names = get_settings().sdk_class_names
    names = "|".join(re.escape(n) for n in class_names) if class_names else "(?!)"

    # Look for patterns like: DataSet.create(...) or RAGFlow.create_dataset(...)
    m = re.search(
        rf"^```python\s*\n((?:{names})\.\w+\(.*?\))",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if m:
        return m.group(1).split("\n")[0].strip()

    # Also try inline code patterns
    m = re.search(
        rf"`((?:{names})\.\w+\([^)]*\))`",
        text,
    )
    if m:
        return m.group(1)
    return ""


def _smart_split_large_section(text: str, max_tokens: int = 1024) -> list[str]:
    """
    Split a large section into smaller chunks while preserving:
    - Code blocks (never split mid-code-block)
    - Parameter lists (keep related params together)
    - Logical paragraph boundaries
    """
    # Rough token estimate: 1 token ≈ 4 chars for English, ≈ 2 chars for mixed
    max_chars = max_tokens * 3  # conservative estimate

    if len(text) <= max_chars:
        return [text]

    chunks = []
    current_chunk = []
    current_len = 0

    # Split into blocks: paragraphs and code blocks
    blocks = []
    pos = 0
    for m in RE_CODE_BLOCK.finditer(text):
        # Text before code block
        pre_text = text[pos : m.start()].strip()
        if pre_text:
            # Split pre_text by paragraphs
            for para in pre_text.split("\n\n"):
                if para.strip():
                    blocks.append(("text", para.strip()))
        # The code block itself
        blocks.append(("code", m.group(0)))
        pos = m.end()

    # Remaining text after last code block
    remaining = text[pos:].strip()
    if remaining:
        for para in remaining.split("\n\n"):
            if para.strip():
                blocks.append(("text", para.strip()))

    for block_type, block_content in blocks:
        block_len = len(block_content)

        # If a single block exceeds max, force it as its own chunk
        if block_len > max_chars:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_len = 0
            chunks.append(block_content)
            continue

        # If adding this block exceeds limit, start new chunk
        if current_len + block_len > max_chars and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_len = 0

        current_chunk.append(block_content)
        current_len += block_len

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def _classify_chunk(text: str, heading: str, sdk_content_pattern: re.Pattern | None = None) -> str:
    """Classify a chunk type based on heading and content."""
    heading_lower = heading.lower()

    if any(kw in heading_lower for kw in ["parameter", "request param", "body param"]):
        return "parameter_list"
    if any(kw in heading_lower for kw in ["example", "request example", "usage"]):
        return "code_example"
    if any(kw in heading_lower for kw in ["response", "return", "success", "failure"]):
        return "response"
    if any(kw in heading_lower for kw in ["error", "error code"]):
        return "error_reference"
    if RE_HTTP_METHOD.search(text):
        return "api_endpoint"
    if "```python" in text and sdk_content_pattern and sdk_content_pattern.search(text):
        return "sdk_method"
    if heading_lower in ("_preamble", "glossary"):
        return "overview"
    return "concept"


def chunk_document(filepath: str, max_chunk_tokens: int = 1024, sdk_class_names: list[str] | None = None) -> list[DocChunk]:
    """
    Parse and chunk a developer documentation file into semantically
    meaningful chunks optimized for RAG retrieval.
    
    The strategy:
    1. Split by H2 sections (major API categories like DATASET MANAGEMENT)
    2. Within each H2, split by H3 (individual API endpoints)
    3. For each endpoint, group: description + request + params + response
    4. If any section is too large, smart-split preserving code blocks
    5. Enrich each chunk with metadata for better retrieval
    """
    if sdk_class_names is None:
        sdk_class_names = get_settings().sdk_class_names

    sdk_content_pattern = _build_sdk_content_pattern(sdk_class_names)

    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    doc_name = os.path.basename(filepath)
    text = _clean_frontmatter(raw_text)

    # Extract document title
    h1_match = RE_H1.search(text)
    doc_title = h1_match.group(1) if h1_match else doc_name

    chunks: list[DocChunk] = []
    chunk_idx = 0

    # Split by H2 (major sections)
    h2_sections = _split_by_headings(text, level=2)

    for h2_name, h2_content in h2_sections:
        if not h2_content.strip():
            continue

        # Split each H2 by H3 (individual endpoints/methods)
        h3_sections = _split_by_headings(h2_content, level=3)

        for h3_name, h3_content in h3_sections:
            if not h3_content.strip():
                continue

            # Build section path
            if h3_name == "_preamble":
                section_path = f"{doc_title} > {h2_name}" if h2_name != "_preamble" else doc_title
            else:
                section_path = f"{doc_title} > {h2_name} > {h3_name}" if h2_name != "_preamble" else f"{doc_title} > {h3_name}"

            # Extract API metadata
            full_section = f"### {h3_name}\n\n{h3_content}" if h3_name != "_preamble" else h3_content
            http_method, endpoint_url = _extract_http_endpoint(full_section)
            sdk_sig = _extract_sdk_signature(full_section, sdk_class_names)

            # Try to keep an entire API endpoint as one chunk
            # Split by H4 if the section is too large
            h4_sections = _split_by_headings(h3_content, level=4)

            if len(h4_sections) <= 1:
                # Small section: treat as single chunk
                sub_chunks = _smart_split_large_section(full_section, max_chunk_tokens)
                for sub in sub_chunks:
                    chunk_type = _classify_chunk(sub, h3_name, sdk_content_pattern)
                    chunks.append(DocChunk(
                        content=sub,
                        doc_name=doc_name,
                        section_path=section_path,
                        chunk_type=chunk_type,
                        api_method=http_method,
                        endpoint_url=endpoint_url,
                        sdk_signature=sdk_sig,
                        chunk_index=chunk_idx,
                    ))
                    chunk_idx += 1
            else:
                # Large section with H4 subsections
                # Group related H4s: (Request + Params) and (Response) and (Examples)
                grouped = _group_h4_subsections(h4_sections, h3_name, max_chunk_tokens)

                for group_heading, group_content in grouped:
                    sub_section_path = f"{section_path} > {group_heading}" if group_heading != "_preamble" else section_path
                    chunk_type = _classify_chunk(group_content, group_heading, sdk_content_pattern)

                    sub_chunks = _smart_split_large_section(group_content, max_chunk_tokens)
                    for sub in sub_chunks:
                        chunks.append(DocChunk(
                            content=sub,
                            doc_name=doc_name,
                            section_path=sub_section_path,
                            chunk_type=chunk_type,
                            api_method=http_method,
                            endpoint_url=endpoint_url,
                            sdk_signature=sdk_sig,
                            chunk_index=chunk_idx,
                        ))
                        chunk_idx += 1

    return chunks


def _group_h4_subsections(
    h4_sections: list[tuple[str, str]],
    parent_h3: str,
    max_tokens: int,
) -> list[tuple[str, str]]:
    """
    Intelligently group H4 subsections to create self-contained chunks.
    
    Groups:
    - "Request" + "Request example" + "Request parameters" → one chunk
    - "Response" (kept together)
    - "Parameters" + "Returns" → one chunk
    - "Examples" (kept as own chunk)
    """
    groups: list[tuple[str, str]] = []
    max_chars = max_tokens * 3

    # Define grouping rules
    request_parts = []
    response_parts = []
    param_parts = []
    example_parts = []
    other_parts = []

    for h4_name, h4_content in h4_sections:
        h4_lower = h4_name.lower()
        combined = f"#### {h4_name}\n\n{h4_content}" if h4_name != "_preamble" else h4_content

        if h4_name == "_preamble":
            other_parts.append(("_preamble", combined))
        elif "request" in h4_lower and "param" not in h4_lower:
            request_parts.append((h4_name, combined))
        elif "param" in h4_lower:
            param_parts.append((h4_name, combined))
        elif "response" in h4_lower or "return" in h4_lower:
            response_parts.append((h4_name, combined))
        elif "example" in h4_lower:
            example_parts.append((h4_name, combined))
        else:
            other_parts.append((h4_name, combined))

    # Merge request + params (they provide context for each other)
    if request_parts or param_parts:
        merged_content = "\n\n".join(
            c for _, c in (request_parts + param_parts)
        )
        heading = "Request & Parameters"
        if len(merged_content) <= max_chars:
            groups.append((heading, merged_content))
        else:
            # Too big: split them
            for name, content in request_parts + param_parts:
                groups.append((name, content))

    # Response
    if response_parts:
        merged = "\n\n".join(c for _, c in response_parts)
        groups.append(("Response", merged))

    # Examples
    if example_parts:
        merged = "\n\n".join(c for _, c in example_parts)
        groups.append(("Examples", merged))

    # Other
    for name, content in other_parts:
        groups.append((name, content))

    return groups


def chunk_all_docs(docs_dir: str, max_chunk_tokens: int = 1024, sdk_class_names: list[str] | None = None) -> list[DocChunk]:
    """Chunk all documents in the docs directory."""
    all_chunks = []
    for filename in os.listdir(docs_dir):
        if filename.endswith((".md", ".mdx")):
            filepath = os.path.join(docs_dir, filename)
            doc_chunks = chunk_document(filepath, max_chunk_tokens, sdk_class_names=sdk_class_names)
            all_chunks.extend(doc_chunks)
    return all_chunks

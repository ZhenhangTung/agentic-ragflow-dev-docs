"""
Metadata enrichment for document chunks.

Generates two kinds of metadata:
  1. Built-in (rule-based): file category, API topics, key entities extracted via regex.
  2. LLM-generated: per-file summary and keywords using a fast small model.

The file-level metadata is stored in the database and used at retrieval time
to pre-filter relevant documents before the expensive hybrid search.
"""
import json
import logging
import re
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from src.config import get_settings
from src.chunker import DocChunk

logger = logging.getLogger(__name__)

# ── Built-in metadata (rule-based) ───────────────────────────────────────

# Mapping from doc filename patterns → file category
_FILE_CATEGORIES = {
    "http_api": "HTTP API Reference",
    "python_api": "Python SDK Reference",
    "glossary": "Glossary / Concepts",
}

_TOPIC_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("dataset_management", re.compile(r"dataset|knowledge.?base", re.I)),
    ("document_management", re.compile(r"document|upload|download|parse|chunk", re.I)),
    ("file_management", re.compile(r"file.?management|upload.?file|download.?file", re.I)),
    ("chat_assistant", re.compile(r"chat|assistant|session|conversation", re.I)),
    ("agent", re.compile(r"\bagent\b", re.I)),
    ("retrieval", re.compile(r"retriev|search|recall", re.I)),
    ("configuration", re.compile(r"config|setting|param", re.I)),
    ("authentication", re.compile(r"auth|api.?key|token", re.I)),
]

_ENTITY_RE = re.compile(
    r"\b(RAGFlow|DataSet|Dataset|Document|Chunk|Chat|Session|Agent|Memory|Ragflow)\b"
)

_ENDPOINT_RE = re.compile(
    r"(GET|POST|PUT|DELETE|PATCH)\s+(/api/v\d+/\S+)", re.I
)

_SDK_METHOD_RE = re.compile(
    r"(?:RAGFlow|DataSet|Dataset|Document|Chunk|Chat|Session|Agent|Memory)\.(\w+)"
)


def _detect_file_category(doc_name: str) -> str:
    name_lower = doc_name.lower()
    for pattern, category in _FILE_CATEGORIES.items():
        if pattern in name_lower:
            return category
    return "General"


def _detect_topics(text: str) -> list[str]:
    topics = []
    for topic, pattern in _TOPIC_PATTERNS:
        if pattern.search(text):
            topics.append(topic)
    return topics


def _extract_entities(text: str) -> list[str]:
    return sorted(set(_ENTITY_RE.findall(text)))


def _extract_endpoints(text: str) -> list[str]:
    return [f"{m.group(1)} {m.group(2)}" for m in _ENDPOINT_RE.finditer(text)]


def _extract_sdk_methods(text: str) -> list[str]:
    return sorted(set(_SDK_METHOD_RE.findall(text)))


def extract_builtin_metadata(doc_name: str, full_text: str) -> dict:
    """Extract rule-based metadata from a full document."""
    return {
        "file_category": _detect_file_category(doc_name),
        "topics": _detect_topics(full_text),
        "entities": _extract_entities(full_text),
        "endpoints": _extract_endpoints(full_text)[:50],  # cap for storage
        "sdk_methods": _extract_sdk_methods(full_text),
    }


# ── LLM-generated metadata ──────────────────────────────────────────────

_FILE_METADATA_PROMPT = """\
You are a documentation analyst. Given the first ~3000 characters of a developer \
documentation file, produce a concise JSON object describing its contents.

File name: {doc_name}

Content (truncated):
{content_preview}

Respond with ONLY a JSON object (no fencing):
{{
  "summary": "2-3 sentence summary of what this file covers",
  "keywords": ["keyword1", "keyword2", ...],
  "covered_apis": ["brief description of each major API section covered"],
  "target_queries": ["example questions this file can answer"]
}}"""


@dataclass
class FileMetadata:
    """Aggregated metadata for a single documentation file."""
    doc_name: str
    # Built-in
    file_category: str = ""
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    endpoints: list[str] = field(default_factory=list)
    sdk_methods: list[str] = field(default_factory=list)
    # LLM-generated
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    covered_apis: list[str] = field(default_factory=list)
    target_queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "doc_name": self.doc_name,
            "file_category": self.file_category,
            "topics": self.topics,
            "entities": self.entities,
            "endpoints": self.endpoints,
            "sdk_methods": self.sdk_methods,
            "summary": self.summary,
            "keywords": self.keywords,
            "covered_apis": self.covered_apis,
            "target_queries": self.target_queries,
        }

    def to_match_text(self) -> str:
        """Build a text blob for the pre-filter LLM to compare against queries."""
        parts = [
            f"File: {self.doc_name}",
            f"Category: {self.file_category}",
            f"Summary: {self.summary}",
            f"Keywords: {', '.join(self.keywords)}",
            f"Topics: {', '.join(self.topics)}",
        ]
        if self.endpoints:
            parts.append(f"Endpoints: {', '.join(self.endpoints[:20])}")
        if self.sdk_methods:
            parts.append(f"SDK methods: {', '.join(self.sdk_methods)}")
        if self.covered_apis:
            parts.append(f"Covered APIs: {'; '.join(self.covered_apis)}")
        if self.target_queries:
            parts.append(f"Example questions: {'; '.join(self.target_queries)}")
        return "\n".join(parts)


async def generate_file_metadata(
    doc_name: str,
    full_text: str,
) -> FileMetadata:
    """
    Generate both built-in and LLM-based metadata for a documentation file.
    """
    settings = get_settings()

    # 1. Built-in metadata
    builtin = extract_builtin_metadata(doc_name, full_text)
    meta = FileMetadata(
        doc_name=doc_name,
        file_category=builtin["file_category"],
        topics=builtin["topics"],
        entities=builtin["entities"],
        endpoints=builtin["endpoints"],
        sdk_methods=builtin["sdk_methods"],
    )

    # 2. LLM-generated metadata (use light model, thinking OFF)
    client = AsyncOpenAI(
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
    )
    preview = full_text[:3000]
    try:
        resp = await client.chat.completions.create(
            model=settings.light_model,
            messages=[
                {
                    "role": "user",
                    "content": _FILE_METADATA_PROMPT.format(
                        doc_name=doc_name, content_preview=preview
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=800,
            extra_body={"enable_thinking": False},
        )
        raw = resp.choices[0].message.content or ""
        raw = raw.strip()
        if raw.startswith("```"):
            raw = "\n".join(
                line for line in raw.split("\n")
                if not line.strip().startswith("```")
            )
        data = json.loads(raw)
        meta.summary = data.get("summary", "")
        meta.keywords = data.get("keywords", [])
        meta.covered_apis = data.get("covered_apis", [])
        meta.target_queries = data.get("target_queries", [])
    except Exception:
        logger.warning("LLM metadata generation failed for %s, using built-in only", doc_name)

    return meta


def enrich_chunks_with_file_metadata(
    chunks: list[DocChunk],
    file_metadata: dict[str, FileMetadata],
) -> list[DocChunk]:
    """
    Inject file-level metadata into each chunk's metadata dict for storage.
    """
    for chunk in chunks:
        fm = file_metadata.get(chunk.doc_name)
        if fm:
            existing = chunk.to_metadata()
            existing["file_category"] = fm.file_category
            existing["file_topics"] = fm.topics
            existing["file_keywords"] = fm.keywords
            existing["file_summary"] = fm.summary
            # We don't overwrite chunk.metadata directly since DocChunk is a dataclass;
            # callers will use chunk.to_metadata() or this enriched dict.
            chunk._enriched_metadata = existing  # type: ignore[attr-defined]
    return chunks

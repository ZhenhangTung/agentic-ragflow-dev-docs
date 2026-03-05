"""
Agentic Search Engine for RAGFlow Developer Documentation.

Implements a multi-step search strategy where an LLM agent:
1. Decomposes complex queries into focused sub-queries
2. Iteratively retrieves documentation across multiple search rounds
3. Evaluates whether gathered context is sufficient
4. Synthesizes a comprehensive answer from all collected evidence
"""
import json
import logging
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from src.config import get_settings
from src.retriever import Retriever, RetrievalResult

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────────

DECOMPOSE_PROMPT = """\
You are a search planning agent for RAGFlow developer documentation.

Given a developer's question, decompose it into 1-4 focused sub-queries that \
together will retrieve all the documentation needed to answer the original question.

Each sub-query should target a specific aspect:
- API endpoints (HTTP method + URL)
- SDK methods (Python class + method)
- Configuration parameters
- Conceptual explanations
- Code examples

Respond with a JSON object:
{
  "sub_queries": ["query1", "query2", ...],
  "reasoning": "Brief explanation of your decomposition strategy"
}

Developer's question: {question}"""

EVALUATE_PROMPT = """\
You are a search evaluation agent for RAGFlow developer documentation.

Given the developer's original question and the documentation chunks retrieved so far, \
determine whether we have enough information to provide a complete answer.

Original question: {question}

Retrieved documentation:
{context}

Respond with a JSON object:
{
  "sufficient": true/false,
  "follow_up_queries": ["query1", ...],
  "reasoning": "What information is missing or why the context is sufficient"
}

Rules:
- Set "sufficient" to true if the retrieved docs cover the question adequately.
- If "sufficient" is false, provide 1-3 targeted follow-up queries to fill gaps.
- If the question is about a specific API or SDK method, ensure we have parameters, \
  request/response formats, and examples.
- An empty follow_up_queries list with sufficient=true means we can proceed to synthesis."""

SYNTHESIZE_PROMPT = """\
You are RAGFlow Developer Docs Assistant, an expert AI assistant specialized in \
RAGFlow API documentation.

You performed an agentic search across multiple documentation sections to answer \
the developer's question. Synthesize a comprehensive answer from all gathered context.

Guidelines:
- Base your answer strictly on the provided documentation context
- Include relevant code examples (curl commands for HTTP API, Python code for SDK)
- Mention specific API endpoints and their HTTP methods
- If information is incomplete, clearly state what is missing
- For chunking-quality questions with insufficient doc evidence, add clearly labeled general ML/NLP library recommendations (for example: sentence-transformers, spaCy, NLTK)
- Cross-reference related endpoints or SDK methods when relevant
- Organize your answer with clear sections when covering multiple aspects

Documentation Context:
{context}

Developer's Question: {question}

Provide a clear, comprehensive, and actionable answer."""


# ── Data Structures ──────────────────────────────────────────────────────

@dataclass
class SearchRound:
    """Record of a single search round."""
    queries: list[str]
    results: list[RetrievalResult]


@dataclass
class AgenticSearchResult:
    """Final result of an agentic search session."""
    answer: str
    rounds: list[SearchRound] = field(default_factory=list)
    all_results: list[RetrievalResult] = field(default_factory=list)
    total_chunks_retrieved: int = 0
    rounds_executed: int = 0


# ── Agentic Search Engine ────────────────────────────────────────────────

class AgenticSearch:
    """
    Multi-step agentic search over RAGFlow documentation.

    Flow:
        Question → Decompose → Search → Evaluate → (loop) → Synthesize
    """

    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self.settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=self.settings.dashscope_api_key,
            base_url=self.settings.dashscope_base_url,
        )
        self.model = self.settings.chat_model

    async def search(
        self,
        question: str,
        max_rounds: int | None = None,
        top_k_per_query: int = 5,
    ) -> AgenticSearchResult:
        """
        Run an agentic search session.

        Args:
            question: The developer's question
            max_rounds: Maximum search rounds (default from config)
            top_k_per_query: Results per sub-query
        """
        max_rounds = max_rounds or self.settings.agentic_max_rounds
        all_results: list[RetrievalResult] = []
        seen_ids: set[int] = set()
        rounds: list[SearchRound] = []

        # Round 1: Decompose the question into sub-queries
        sub_queries = await self._decompose(question)
        logger.info("Decomposed into %d sub-queries: %s", len(sub_queries), sub_queries)

        for round_num in range(max_rounds):
            # Search for each sub-query
            round_results: list[RetrievalResult] = []
            for query in sub_queries:
                results = await self.retriever.search(
                    query=query, top_k=top_k_per_query, search_mode="hybrid"
                )
                for r in results:
                    if r.id not in seen_ids:
                        seen_ids.add(r.id)
                        round_results.append(r)

            all_results.extend(round_results)
            rounds.append(SearchRound(queries=sub_queries, results=round_results))

            logger.info(
                "Round %d: %d new chunks (total: %d)",
                round_num + 1, len(round_results), len(all_results),
            )

            # Last round — skip evaluation
            if round_num >= max_rounds - 1:
                break

            # Evaluate if we have enough context
            sufficient, follow_ups = await self._evaluate(question, all_results)
            if sufficient or not follow_ups:
                logger.info("Context sufficient after round %d", round_num + 1)
                break

            # Prepare next round with follow-up queries
            sub_queries = follow_ups
            logger.info("Follow-up queries: %s", follow_ups)

        # Synthesize final answer from all gathered context
        answer = await self._synthesize(question, all_results)

        return AgenticSearchResult(
            answer=answer,
            rounds=rounds,
            all_results=all_results,
            total_chunks_retrieved=len(all_results),
            rounds_executed=len(rounds),
        )

    # ── LLM Steps ────────────────────────────────────────────────────────

    async def _decompose(self, question: str) -> list[str]:
        """Decompose a complex question into focused sub-queries."""
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": DECOMPOSE_PROMPT.format(question=question)},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        content = resp.choices[0].message.content or ""
        return self._parse_sub_queries(content, question)

    async def _evaluate(
        self, question: str, results: list[RetrievalResult]
    ) -> tuple[bool, list[str]]:
        """Evaluate if gathered context is sufficient."""
        context = self._format_context(results)
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": EVALUATE_PROMPT.format(
                        question=question, context=context
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=512,
        )
        content = resp.choices[0].message.content or ""
        return self._parse_evaluation(content)

    async def _synthesize(
        self, question: str, results: list[RetrievalResult]
    ) -> str:
        """Synthesize a comprehensive answer from all retrieved context."""
        context = self._format_context(results)
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": SYNTHESIZE_PROMPT.format(
                        question=question, context=context
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=3000,
        )
        return resp.choices[0].message.content or ""

    # ── Helpers ───────────────────────────────────────────────────────────

    def _format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieval results as context for the LLM."""
        parts = []
        for i, r in enumerate(results, 1):
            header_parts = []
            if r.doc_name:
                header_parts.append(f"Source: {r.doc_name}")
            if r.section_path:
                header_parts.append(f"Section: {r.section_path}")
            if r.api_method and r.endpoint_url:
                header_parts.append(f"API: {r.api_method} {r.endpoint_url}")
            if r.sdk_signature:
                header_parts.append(f"SDK: {r.sdk_signature}")
            header = " | ".join(header_parts) if header_parts else f"Chunk {i}"
            parts.append(f"--- [{header}] ---\n{r.content}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_sub_queries(text: str, fallback_question: str) -> list[str]:
        """Parse sub-queries from LLM response, with fallback."""
        try:
            data = json.loads(_extract_json(text))
            queries = data.get("sub_queries", [])
            if queries and isinstance(queries, list):
                return [q for q in queries if isinstance(q, str) and q.strip()]
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Failed to parse decomposition, using original question")
        return [fallback_question]

    @staticmethod
    def _parse_evaluation(text: str) -> tuple[bool, list[str]]:
        """Parse evaluation response from LLM."""
        try:
            data = json.loads(_extract_json(text))
            sufficient = bool(data.get("sufficient", False))
            follow_ups = data.get("follow_up_queries", [])
            if isinstance(follow_ups, list):
                follow_ups = [q for q in follow_ups if isinstance(q, str) and q.strip()]
            else:
                follow_ups = []
            return sufficient, follow_ups
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Failed to parse evaluation, assuming sufficient")
            return True, []


def _extract_json(text: str) -> str:
    """Extract JSON from an LLM response that may contain markdown fencing."""
    text = text.strip()
    if text.startswith("```"):
        # Remove markdown code fences
        lines = text.split("\n")
        # Drop first line (```json or ```) and last line (```)
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            if line.strip() == "```" and in_block:
                break
            if in_block:
                json_lines.append(line)
        return "\n".join(json_lines)
    return text

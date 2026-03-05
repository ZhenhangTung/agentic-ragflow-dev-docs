"""
RAG Generator using Qwen3.5-Plus via DashScope.
"""
from openai import AsyncOpenAI
from src.config import get_settings

SYSTEM_PROMPT = """You are RAGFlow Developer Docs Assistant, an expert AI assistant specialized in the RAGFlow API documentation.

Your role is to help developers integrate with RAGFlow by:
1. Answering questions about the RAGFlow HTTP API and Python SDK
2. Providing code examples for common integration patterns
3. Explaining API parameters, request/response formats, and error codes
4. Guiding users through RAGFlow concepts like datasets, documents, chunks, chat assistants, agents, and memory management

Guidelines:
- Always base your answers on the provided documentation context
- Include relevant code examples (curl commands for HTTP API, Python code for SDK)
- Mention specific API endpoints and their HTTP methods
- If the documentation doesn't cover the question, say so clearly
- Use clear, concise language appropriate for developers
- When showing API calls, include all required headers and parameters
- Mention related endpoints or methods that might be useful"""

CONTEXT_PROMPT = """Based on the following RAGFlow documentation excerpts, answer the developer's question.

Documentation Context:
{context}

Developer's Question: {question}

Provide a clear, actionable answer with code examples where appropriate."""


class Generator:
    """RAG answer generator using Qwen via DashScope."""

    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=self.settings.dashscope_api_key,
            base_url=self.settings.dashscope_base_url,
        )
        self.model = self.settings.chat_model
        self._extra_body = (
            {"enable_thinking": True}
            if self.settings.enable_thinking
            else {"enable_thinking": False}
        )

    async def generate(
        self,
        question: str,
        context_chunks: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate a RAG-powered answer using retrieved context chunks.
        """
        # Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source_info = []
            if chunk.get("doc_name"):
                source_info.append(f"Source: {chunk['doc_name']}")
            if chunk.get("section_path"):
                source_info.append(f"Section: {chunk['section_path']}")
            if chunk.get("api_method") and chunk.get("endpoint_url"):
                source_info.append(f"API: {chunk['api_method']} {chunk['endpoint_url']}")
            if chunk.get("sdk_signature"):
                source_info.append(f"SDK: {chunk['sdk_signature']}")

            header = " | ".join(source_info) if source_info else f"Chunk {i}"
            context_parts.append(f"--- [{header}] ---\n{chunk['content']}")

        context = "\n\n".join(context_parts)

        user_message = CONTEXT_PROMPT.format(
            context=context, question=question
        )

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=self._extra_body,
        )

        return resp.choices[0].message.content

    async def generate_stream(
        self,
        question: str,
        context_chunks: list[dict],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        """
        Stream a RAG-powered answer.
        """
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source_info = []
            if chunk.get("doc_name"):
                source_info.append(f"Source: {chunk['doc_name']}")
            if chunk.get("section_path"):
                source_info.append(f"Section: {chunk['section_path']}")
            if chunk.get("api_method") and chunk.get("endpoint_url"):
                source_info.append(f"API: {chunk['api_method']} {chunk['endpoint_url']}")

            header = " | ".join(source_info) if source_info else f"Chunk {i}"
            context_parts.append(f"--- [{header}] ---\n{chunk['content']}")

        context = "\n\n".join(context_parts)
        user_message = CONTEXT_PROMPT.format(
            context=context, question=question
        )

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_body=self._extra_body,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

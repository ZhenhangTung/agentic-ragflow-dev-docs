"""
Embedding service using Alibaba Cloud DashScope text-embedding-v4.
"""
from openai import AsyncOpenAI, BadRequestError
from openai.types import CreateEmbeddingResponse
from src.config import get_settings


class Embedder:
    """Generate embeddings using Qwen text-embedding-v4 via DashScope."""

    MAX_BATCH_SIZE = 10

    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=self.settings.dashscope_api_key,
            base_url=self.settings.dashscope_base_url,
        )
        self.model = self.settings.embedding_model
        self.dimensions = self.settings.embedding_dimensions

    async def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        resp = await self._create_embedding(input_data=text)
        return resp.data[0].embedding

    async def embed_batch(self, texts: list[str], batch_size: int = MAX_BATCH_SIZE) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batches.
        DashScope supports up to 10 texts per request for this endpoint.
        """
        effective_batch_size = max(1, min(batch_size, self.MAX_BATCH_SIZE))
        all_embeddings = []
        for i in range(0, len(texts), effective_batch_size):
            batch = texts[i : i + effective_batch_size]
            resp = await self._create_embedding(input_data=batch)
            batch_embeddings = [d.embedding for d in resp.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    async def _create_embedding(self, input_data: str | list[str]) -> CreateEmbeddingResponse:
        """
        Create embedding with provider compatibility fallback.

        Try a full request first (`dimensions` + `encoding_format`), then retry
        with reduced optional parameters if the provider rejects them.
        """
        request_variants = [
            {"model": self.model, "input": input_data, "dimensions": self.dimensions, "encoding_format": "float"},
            {"model": self.model, "input": input_data, "encoding_format": "float"},
            {"model": self.model, "input": input_data, "dimensions": self.dimensions},
            {"model": self.model, "input": input_data},
        ]

        last_error: Exception | None = None
        for params in request_variants:
            try:
                return await self.client.embeddings.create(**params)
            except BadRequestError as e:
                last_error = e
                body = e.body if isinstance(e.body, dict) else {}
                err = body.get("error", {}) if isinstance(body.get("error"), dict) else {}
                error_param = err.get("param", "")
                error_code = err.get("code", "")
                error_type = err.get("type", "")
                error_msg = err.get("message") or str(e)

                error_param = error_param.lower() if isinstance(error_param, str) else ""
                error_code = error_code.lower() if isinstance(error_code, str) else ""
                error_type = error_type.lower() if isinstance(error_type, str) else ""
                error_msg = error_msg.lower() if isinstance(error_msg, str) else ""

                is_compat_param_error = (
                    error_param in {"dimensions", "encoding_format"}
                    or error_code in {"invalid_parameter", "unknown_parameter"}
                    or error_type in {"invalid_parameter", "unknown_parameter"}
                    or "dimensions" in error_msg
                    or "encoding_format" in error_msg
                )
                if not is_compat_param_error:
                    raise

        if last_error is None:
            raise RuntimeError("Embedding request failed before response was received.")
        raise last_error

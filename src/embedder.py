"""
Embedding service using Alibaba Cloud DashScope text-embedding-v4.
"""
from openai import AsyncOpenAI
from src.config import get_settings


class Embedder:
    """Generate embeddings using Qwen text-embedding-v4 via DashScope."""

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

    async def embed_batch(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in batches.
        DashScope supports up to 25 texts per request for text-embedding-v4.
        """
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = await self._create_embedding(input_data=batch)
            batch_embeddings = [d.embedding for d in resp.data]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    async def _create_embedding(self, input_data):
        request_variants = [
            {"model": self.model, "input": input_data, "dimensions": self.dimensions, "encoding_format": "float"},
            {"model": self.model, "input": input_data, "encoding_format": "float"},
            {"model": self.model, "input": input_data, "dimensions": self.dimensions},
            {"model": self.model, "input": input_data},
        ]

        last_error = None
        for params in request_variants:
            try:
                return await self.client.embeddings.create(**params)
            except Exception as e:
                last_error = e
                msg = str(e).lower()
                if not any(k in msg for k in ("parameter", "unsupported", "unknown", "invalid")):
                    raise

        raise last_error

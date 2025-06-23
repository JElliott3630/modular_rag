from __future__ import annotations
from pathlib import Path
from typing import Sequence
import logging
import yaml
from openai import AsyncOpenAI
from src import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)


class OpenAICompletion:
    def __init__(
        self,
        model: str | None = None,
        max_context: int = 12_000,
        prompts_path: str | None = None,
    ) -> None:
        self._model = model or config.GPT_MODEL
        self._max_ctx = max_context
        cfg = yaml.safe_load(Path(prompts_path or config.PROMPTS_PATH).read_text())["generation"]
        self._system = cfg["system"]
        self._template = cfg["template"]

    async def run(self, query: str, context: Sequence[str]) -> str:
        joined = "\n\n".join(context)[: self._max_ctx]
        user_msg = self._template.format(query=query, context=joined)
        resp = await _client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()

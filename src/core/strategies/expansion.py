from __future__ import annotations
from pathlib import Path
from typing import Sequence
import json
import logging
import yaml
from openai import AsyncOpenAI
from src import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)


class PromptExpansion:
    def __init__(
        self,
        model: str | None = None,
        n: int = 3,
        prompts_path: str | None = None,
    ) -> None:
        self._model = model or config.GPT_MODEL
        self._n = n
        cfg = yaml.safe_load(Path(prompts_path or config.PROMPTS_PATH).read_text())["expansion"]
        self._system = cfg["system"]
        self._template = cfg["template"]

    async def run(self, query: str) -> Sequence[str]:
        user_msg = self._template.format(query=query, n=self._n)
        resp = await _client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
        )
        return json.loads(resp.choices[0].message.content)

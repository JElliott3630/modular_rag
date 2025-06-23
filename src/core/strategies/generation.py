from __future__ import annotations
from pathlib import Path
from typing import Sequence
import logging
import json
import yaml
import openai
from src import config

openai.api_key = config.OPENAI_API_KEY
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OpenAICompletion:
    def __init__(
        self,
        model: str | None = None,
        max_context: int = 12_000,
        prompts_path: str | None = None,
    ) -> None:
        self._model = model or config.GPT_MODEL
        self._max_context = max_context
        path = Path(prompts_path or config.PROMPTS_PATH).resolve()
        prompts = yaml.safe_load(path.read_text())
        gen_cfg = prompts["generation"]
        self._system = gen_cfg["system"]
        self._template = gen_cfg["template"]

    def run(self, query: str, context: Sequence[str]) -> str:
        logger.info("generating answer")
        joined = "\n\n".join(context)[: self._max_context]
        user = self._template.format(query=query, context=joined)
        resp = openai.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": self._system}, {"role": "user", "content": user}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()

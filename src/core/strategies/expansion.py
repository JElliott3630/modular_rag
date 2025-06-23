from __future__ import annotations
from pathlib import Path
from typing import Sequence
import json
import logging
import yaml
import openai
from src import config

openai.api_key = config.OPENAI_API_KEY
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PromptExpansion:
    def __init__(
        self,
        model: str | None = None,
        n: int = 3,
        prompts_path: str | None = None,
    ) -> None:
        self._model = model or config.GPT_MODEL
        self._n = n
        path = Path(prompts_path or config.PROMPTS_PATH).resolve()
        prompts = yaml.safe_load(path.read_text())
        exp_cfg = prompts["expansion"]
        self._system = exp_cfg["system"]
        self._template = exp_cfg["template"]

    def run(self, query: str) -> Sequence[str]:
        logger.info("expanding query")
        user = self._template.format(query=query, n=self._n)
        resp = openai.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": self._system}, {"role": "user", "content": user}],
            temperature=0.7,
        )
        content = resp.choices[0].message.content.strip()
        return json.loads(content)

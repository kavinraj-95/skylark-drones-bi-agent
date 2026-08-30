"""Thin provider adapter for the two LLM calls this system makes.

Kept deliberately small. The app uses an LLM for exactly two things - reading a
question into a `QueryIntent`, and turning a finished `AnalysisResult` into prose -
and neither needs streaming, tools, or conversation state. Wrapping that in one
interface means switching providers is a config change, not a refactor.

What never passes through here is as important as what does: no raw board rows, no
credentials, and nothing the model returns is executed. Structured calls are parsed
into a pydantic model and rejected if they do not fit.
"""

from __future__ import annotations

import json
import logging
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ..config import LLMConfig

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

#: Bounds on model output. Intent extraction is a small JSON object; a founder-facing
#: answer is a few short paragraphs.
#:
#: Generous on purpose. On Gemini 2.5 models internal "thinking" tokens are charged
#: against this same budget, so a limit sized to the visible answer truncates it
#: mid-sentence. Thinking is disabled below for these calls, and the headroom stays as
#: insurance against a model that ignores that.
_STRUCTURED_MAX_TOKENS = 2048
_PROSE_MAX_TOKENS = 4096


class LLMError(RuntimeError):
    """The model could not be reached, or returned something unusable."""

    user_message = (
        "The language model is unavailable right now. The underlying figures are "
        "still correct and shown below."
    )

    def __init__(self, message: str, *, user_message: str | None = None) -> None:
        super().__init__(message)
        if user_message:
            self.user_message = user_message


class LLMClient:
    """Provider-agnostic wrapper around a chat model."""

    def __init__(self, config: LLMConfig):
        self._config = config
        self._client = None

    # -- provider setup ----------------------------------------------------------

    def _gemini(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._config.api_key)
        return self._client

    def _anthropic(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self._config.api_key)
        return self._client

    # -- public API --------------------------------------------------------------

    def structured(self, *, system: str, prompt: str, schema: type[T]) -> T:
        """Ask the model for a JSON object matching `schema`.

        A response that does not validate is an error, not something to patch up. The
        caller decides what to do about it - usually falling back to a deterministic
        interpretation rather than guessing at what the model meant.
        """
        raw = self._generate(
            system=system, prompt=prompt, schema=schema, max_tokens=_STRUCTURED_MAX_TOKENS
        )
        try:
            return schema.model_validate_json(raw)
        except ValidationError:
            # Some providers wrap JSON in prose or fences. One salvage attempt, then stop.
            try:
                start, end = raw.index("{"), raw.rindex("}") + 1
                return schema.model_validate_json(raw[start:end])
            except (ValueError, ValidationError) as exc:
                raise LLMError(
                    f"Model returned output that does not match {schema.__name__}: {exc}"
                ) from exc

    def prose(self, *, system: str, prompt: str) -> str:
        """Ask the model for plain text."""
        return self._generate(
            system=system, prompt=prompt, schema=None, max_tokens=_PROSE_MAX_TOKENS
        ).strip()

    # -- internals ---------------------------------------------------------------

    def _generate(
        self, *, system: str, prompt: str, schema: type[BaseModel] | None, max_tokens: int
    ) -> str:
        if self._config.provider == "anthropic":
            return self._generate_anthropic(
                system=system, prompt=prompt, schema=schema, max_tokens=max_tokens
            )
        return self._generate_gemini(
            system=system, prompt=prompt, schema=schema, max_tokens=max_tokens
        )

    def _generate_gemini(
        self, *, system: str, prompt: str, schema: type[BaseModel] | None, max_tokens: int
    ) -> str:
        from google.genai import types

        config: dict = {
            "system_instruction": system,
            # Deterministic by design: the same question over the same data should
            # produce the same reading and the same explanation.
            "temperature": 0.0,
            "max_output_tokens": max_tokens,
        }

        # Neither call needs extended reasoning - the analysis is already done, and
        # thinking tokens would eat the output budget. Guarded because the option
        # only exists on models that support it.
        try:
            config["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except (AttributeError, TypeError):  # pragma: no cover - SDK/model dependent
            pass
        if schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = schema

        try:
            response = self._gemini().models.generate_content(
                model=self._config.model,
                contents=prompt,
                config=types.GenerateContentConfig(**config),
            )
        except Exception as exc:  # provider SDKs raise a wide variety of types
            raise LLMError(f"Gemini request failed: {exc}") from exc

        text = (response.text or "").strip()
        if not text:
            raise LLMError("Gemini returned an empty response.")
        return text

    def _generate_anthropic(
        self, *, system: str, prompt: str, schema: type[BaseModel] | None, max_tokens: int
    ) -> str:
        instruction = system
        if schema is not None:
            instruction += (
                "\n\nReply with a single JSON object and nothing else. It must match "
                f"this JSON schema:\n{json.dumps(schema.model_json_schema())}"
            )
        try:
            message = self._anthropic().messages.create(
                model=self._config.model,
                max_tokens=max_tokens,
                temperature=0.0,
                system=instruction,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc

        text = "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        ).strip()
        if not text:
            raise LLMError("Anthropic returned an empty response.")
        return text

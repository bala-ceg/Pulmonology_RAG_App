"""
LLM service — initialisation and invocation wrapper for ChatOpenAI.

Provides:
- ``LLMService``   — class encapsulating LLM + embeddings lifecycle.
- ``llm_service``  — module-level singleton; import and use directly.

Usage::

    from services.llm_service import llm_service

    response_text = llm_service.invoke("Explain COPD in simple terms.")
    embeddings    = llm_service.get_embeddings()
    ctx_llm       = llm_service.create_contextual_llm(patient_context="...")
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import Config
from utils.error_handlers import get_logger

logger = get_logger(__name__)


class LLMService:
    """Wrapper around ChatOpenAI and OpenAIEmbeddings.

    A single instance is created at module level (``llm_service``) so the
    underlying objects are constructed once and reused across the application.
    """

    def __init__(self, config: type[Config] = Config) -> None:
        """Initialise the service using values from *config*.

        Args:
            config: A :class:`~config.Config` class (or subclass) to read
                    settings from.  Defaults to the global :class:`Config`.
        """
        self._config = config
        self._llm: ChatOpenAI = self._build_llm()
        logger.info("LLMService initialised (model=%s)", config.LLM_MODEL_NAME)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_llm(self, temperature: float | None = None) -> ChatOpenAI:
        """Return a :class:`ChatOpenAI` built from config values."""
        return ChatOpenAI(
            api_key=self._config.OPENAI_API_KEY,
            base_url=self._config.OPENAI_BASE_URL,
            model_name=self._config.LLM_MODEL_NAME,
            temperature=temperature if temperature is not None else self._config.LLM_DEFAULT_TEMPERATURE,
            request_timeout=self._config.LLM_REQUEST_TIMEOUT,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_llm(self, temperature: float | None = None) -> ChatOpenAI:
        """Return the underlying :class:`ChatOpenAI` instance.

        If *temperature* differs from the default a new instance is created
        so that the shared ``_llm`` is not mutated.

        Args:
            temperature: Override the default temperature.  ``None`` keeps
                         the value from :attr:`Config.LLM_DEFAULT_TEMPERATURE`.

        Returns:
            A :class:`ChatOpenAI` instance.
        """
        if temperature is not None and temperature != self._config.LLM_DEFAULT_TEMPERATURE:
            return self._build_llm(temperature=temperature)
        return self._llm

    def invoke(
        self,
        prompt: str | list[Any],
        temperature: float | None = None,
    ) -> str:
        """Invoke the LLM with *prompt* and return the response as a string.

        Args:
            prompt: A plain string or a list of LangChain message objects.
            temperature: Optional temperature override.

        Returns:
            The LLM response content as a plain ``str``.

        Raises:
            RuntimeError: Wraps any exception raised by the underlying LLM
                          call with a descriptive message.
        """
        try:
            llm = self.get_llm(temperature=temperature)
            response = llm.invoke(prompt)
            # AIMessage and similar objects expose .content; plain strings pass through.
            content: str = getattr(response, "content", response)
            return str(content)
        except Exception as exc:
            logger.error("LLM invocation failed: %s", exc)
            raise RuntimeError(f"LLM invocation failed: {exc}") from exc

    def create_contextual_llm(
        self,
        patient_context: str | None = None,
    ) -> ChatOpenAI:
        """Return a :class:`ChatOpenAI` configured with an optional patient context.

        Mirrors the ``create_contextual_llm()`` function in ``main.py``.
        The system message is stored on the returned instance as
        ``._system_message`` so downstream chains can reference it.

        Args:
            patient_context: Free-text patient context (demographics,
                             conditions, history).  When provided it is
                             prepended to the base medical system message.

        Returns:
            A :class:`ChatOpenAI` instance with ``._system_message`` set.
        """
        base_message = self._config.MEDICAL_SYSTEM_MESSAGE

        if patient_context:
            system_message = (
                f"Patient Context: {patient_context}\n\n"
                f"{base_message} Always consider the patient context when providing "
                "medical advice and recommendations. Tailor your responses to the "
                "specific patient demographics, conditions, and medical history provided."
            )
        else:
            system_message = base_message

        contextual_llm = ChatOpenAI(
            api_key=self._config.OPENAI_API_KEY,
            base_url=self._config.OPENAI_BASE_URL,
            model_name=self._config.LLM_MODEL_NAME,
            temperature=self._config.LLM_DEFAULT_TEMPERATURE,
            request_timeout=self._config.LLM_REQUEST_TIMEOUT,
        )
        contextual_llm._system_message = system_message  # type: ignore[attr-defined]
        return contextual_llm

    def get_embeddings(self) -> OpenAIEmbeddings:
        """Create and return an :class:`OpenAIEmbeddings` instance.

        A new instance is returned on each call because
        :class:`OpenAIEmbeddings` is stateless; callers may cache the result
        themselves if needed.

        Returns:
            An :class:`OpenAIEmbeddings` configured with values from
            :class:`Config`.
        """
        return OpenAIEmbeddings(
            api_key=self._config.OPENAI_API_KEY,
            base_url=self._config.OPENAI_BASE_URL,
            model=self._config.EMBEDDING_MODEL_NAME,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
llm_service = LLMService()

__all__ = ["LLMService", "llm_service"]

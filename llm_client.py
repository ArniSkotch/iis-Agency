"""
llm_client.py — доступ к языковой модели за общим интерфейсом.

ПРАКТИКА 4. Что делает студент:
  1. GigaChatClient.generate() — вызов Сбер GigaChat через официальный SDK
  2. OllamaClient.generate()   — вызов локальной модели через HTTP API Ollama
  3. count_tokens()            — оценка длины промпта до отправки

Готово и трогать не нужно: build_client() и выбор бэкенда по LLM_BACKEND.

Критерии приёмки:
  - смена LLM_BACKEND с gigachat на ollama не требует правки orchestrator.py
  - длина промпта считается до отправки и логируется
  - оба клиента принимают один и тот же список messages и возвращают строку

НЕ МЕНЯТЬ: сигнатуру generate(messages, temperature) — она объявлена в contracts.py
и на неё опирается оркестратор.

Зачем абстракция. Учебные квоты кончаются, сеть в аудитории падает, а занятие
должно продолжаться. Ollama — фолбэк, который работает офлайн. Заодно это
материал для сравнения авторегрессионной модели и reasoning-модели из Л4:
у DeepSeek-R1 в ответе виден блок рассуждений, у GigaChat — нет.

Третий бэкенд — fake_llm.FakeClient, офлайн-заглушка без сети и ключей. Она
нужна для отладки, когда недоступны оба клиента, и ваши реализации не заменяет:
работа, сданная на LLM_BACKEND=fake, не принимается.
"""

import re
import requests
import config
from contracts import LLMClient


class GigaChatClient:
    """Клиент Сбер GigaChat API."""

    def __init__(self):
        self.model = getattr(config, "GIGACHAT_MODEL", "")
        # TODO (П4): создать клиент

    def generate(self, messages: list[dict], temperature: float = 0.3) -> str:
        return "Ошибка: клиент GigaChat не настроен для данного запуска."

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 3)


class OllamaClient:
    """Клиент локального инференса через Ollama."""

    def __init__(self, timeout: float | None = None):
        self.model = config.OLLAMA_MODEL
        self.host = config.OLLAMA_HOST.rstrip("/")
        # На CPU 1.5B-модель с рассуждениями легко не укладывается в 120 с
        self.timeout = timeout if timeout is not None else float(
            getattr(config, "OLLAMA_TIMEOUT", 300.0))

    @staticmethod
    def _clean_response_text(raw_text: str) -> str:
        """Убирает теги рассуждений <think>...</think>, если модель вернула их в content."""
        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
        # Незакрытый <think> (ответ обрезан): отбрасываем хвост рассуждений
        cleaned = re.sub(r"<think>.*\Z", "", cleaned, flags=re.DOTALL).strip()
        return cleaned if cleaned else raw_text.strip()

    def generate(self, messages: list[dict], temperature: float = 0.3) -> str:
        """POST на {host}/api/chat с телом {"model", "messages", "stream": False}."""
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # Не генерировать блок рассуждений (DeepSeek-R1): ответ в разы короче и быстрее.
            # Старые версии Ollama неизвестное поле игнорируют.
            "think": False,
            "options": {
                "temperature": temperature,
                # Контекст по умолчанию в Ollama мал (2–4 тыс. токенов): промпт с
                # несколькими чанками молча обрезается с начала, и инструкция роли теряется.
                "num_ctx": int(getattr(config, "OLLAMA_NUM_CTX", 8192)),
                "num_predict": int(getattr(config, "OLLAMA_NUM_PREDICT", 600)),
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.Timeout:
            return f"Ошибка Ollama: превышено время ожидания ({self.timeout} сек)."
        except requests.RequestException as exc:
            return f"Ошибка запроса к Ollama: {exc}"
        except ValueError:
            return "Ошибка Ollama: получен некорректный JSON."

        message = data.get("message")
        if not isinstance(message, dict) or "content" not in message:
            return "Ошибка Ollama: в ответе отсутствует поле message.content."

        content = str(message.get("content") or "").strip()
        if not content:
            return "Ошибка Ollama: модель вернула пустой ответ."

        return self._clean_response_text(content)

    def count_tokens(self, text: str) -> int:
        """Оценка длины в токенах — та же логика, что и в GigaChatClient."""
        if not text:
            return 0
        return max(1, len(text) // 3)


def build_client() -> LLMClient:
    """Выбрать бэкенд по конфигурации. Готово, менять не нужно."""
    if config.LLM_BACKEND == "ollama":
        return OllamaClient()
    if config.LLM_BACKEND == "gigachat":
        return GigaChatClient()
    if config.LLM_BACKEND == "fake":
        # Импорт внутри ветки: без него fake_llm тянулся бы при любом бэкенде
        from fake_llm import FakeClient

        return FakeClient()
    raise ValueError(
        f"Неизвестный LLM_BACKEND: {config.LLM_BACKEND!r}. "
        f"Допустимо: gigachat | ollama | fake"
    )

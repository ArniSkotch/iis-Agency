"""
mcp_server.py — MCP-сервер, отдающий инструменты работы со знаниями.

ГОТОВ ЦЕЛИКОМ. Студент его не пишет — это инфраструктура протокола (Л3).

Сервер публикует три инструмента:
  search(query, k)                    — семантический поиск по индексу
  search_filtered(query, filters, k)  — то же, но с фильтром по метаданным
  get_history(session_id, n)          — последние n реплик сессии

Оркестратор обращается к ним как MCP-клиент (mcp_client.py), а не прямым импортом.
В этом смысл: инструменты описаны схемой, вызываются по протоколу, и их набор
можно расширить, не трогая ядро.

Запускается автоматически как подпроцесс из mcp_client.py. Вручную — для отладки:
    python mcp_server.py
"""

import sys
from mcp.server.fastmcp import FastMCP

import config

sys.stderr.write("[MCP_SERVER] Инициализация модуля MCP-сервера...\n")
sys.stderr.flush()

mcp = FastMCP("iis-rag")


def _to_chunks(result: dict, verbose = False) -> list[dict]:
    """Разворачивает ответ ChromaDB в плоский список чанков."""
    if verbose:
        sys.stderr.write("[MCP_SERVER] Этап: конвертация результатов ChromaDB в чанки (_to_chunks)...\n")
        sys.stderr.flush()

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    chunks = []
    for text, meta, distance in zip(documents, metadatas, distances):
        meta = meta or {}
        chunks.append({
            "text": text,
            "source": meta.get("source", ""),
            "page": meta.get("page", 0),
            "category": meta.get("category", ""),
            # ChromaDB настроена на косинусное расстояние; сходство = 1 - расстояние
            "score": round(1.0 - float(distance), 4),
        })

    if verbose:
        sys.stderr.write(f"[MCP_SERVER] Успешно сформировано чанков: {len(chunks)}\n")
        sys.stderr.flush()
    return chunks


@mcp.tool()
def search(query: str, k: int = config.TOP_K, verbose = False) -> list[dict]:
    """Семантический поиск по базе знаний. Возвращает k наиболее близких фрагментов."""
    if verbose:
        sys.stderr.write(f"[MCP_SERVER] Вызван инструмент 'search' с запросом: '{query}' (k={k})\n")
        sys.stderr.flush()

    if verbose:
        sys.stderr.write("[MCP_SERVER] Загрузка коллекции из config.get_collection() (здесь поднимается модель и БД)...\n")
        sys.stderr.flush()

    collection = config.get_collection()

    if verbose:
        sys.stderr.write("[MCP_SERVER] Отправка запроса (query) в ChromaDB...\n")
        sys.stderr.flush()
    raw_result = collection.query(query_texts=[query], n_results=k)

    if verbose:
        sys.stderr.write("[MCP_SERVER] Ответ от ChromaDB получен. Передача в конвертер...\n")
        sys.stderr.flush()
    return _to_chunks(raw_result)


@mcp.tool()
def search_filtered(query: str, filters: dict, k: int = config.TOP_K, verbose = False) -> list[dict]:
    """Поиск с фильтром по метаданным.

    filters — словарь вида {"category": "regulations"} или {"source": "nk_rf.pdf"}.
    Пустой словарь равносилен обычному search.
    """
    if verbose:
        sys.stderr.write(f"[MCP_SERVER] Вызван 'search_filtered' с запросом: '{query}', фильтры: {filters}\n")
        sys.stderr.flush()

    if verbose:
        sys.stderr.write("[MCP_SERVER] Загрузка коллекции из config.get_collection()...\n")
        sys.stderr.flush()
    collection = config.get_collection()

    where = filters or None
    if where and len(where) > 1:
        # ChromaDB требует явный оператор при нескольких условиях
        where = {"$and": [{key: value} for key, value in filters.items()]}

    if verbose:
        sys.stderr.write(f"[MCP_SERVER] Выполнение фильтрованного запроса в ChromaDB (where={where})...\n")
        sys.stderr.flush()
    raw_result = collection.query(query_texts=[query], n_results=k, where=where)

    return _to_chunks(raw_result)


@mcp.tool()
def get_history(session_id: str, n: int = config.MEMORY_WINDOW, verbose = False) -> list[dict]:
    """Последние n реплик сессии. Начинает работать после практики 5."""
    if verbose:
        sys.stderr.write(f"[MCP_SERVER] Вызван 'get_history' для сессии {session_id}\n")
        sys.stderr.flush()

    import memory

    history = [
        {"role": m.role, "content": m.content, "timestamp": m.timestamp}
        for m in memory.window(session_id, n)
    ]
    if verbose:
        sys.stderr.write(f"[MCP_SERVER] История получена (сообщений: {len(history)})\n")
        sys.stderr.flush()
    return history


if __name__ == "__main__":
    sys.stderr.write("[MCP_SERVER] Запуск mcp.run() (ожидание запросов через stdio)...\n")
    sys.stderr.flush()
    mcp.run()
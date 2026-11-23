"""
test_suite.py — прогон Gold Dataset и оценка ответов промптом-судьёй.

ПРАКТИКА 6. Что делает студент:
  1. расширить gold_dataset.json с пяти вопросов, написанных на П2, до десяти,
     добавив к каждому эталонный ответ в поле expected
  2. написать JUDGE_PROMPT — промпт-судью по триаде RAG-метрик
  3. реализовать judge() — вызов судьи и разбор его вердикта

Готово: загрузка датасета, прогон через ядро, сводный отчёт.

Критерии приёмки:
  - в gold_dataset.json ровно 10 вопросов, среди них есть заведомо сложные:
    вне исходных данных, на стыке разделов, с неоднозначной формулировкой
  - пять вопросов с П2 сохранены; замена вопроса допускается только с письменным
    объяснением, чем он оказался негоден
  - прогон проходит все 10 без падений
  - судья возвращает три оценки триады на каждый вопрос, а не одну общую
  - в отчёте зафиксирована хотя бы одна пойманная галлюцинация
    (или обоснованно показано, что их нет)
  - обе метрики измерены и разведены: hit-rate@5 (механически, ingest.py, с П2)
    и Context Precision из триады (судьёй, здесь). В отчёте видно, что это
    разные числа, а не одно под двумя именами

Почему судьёй выступает LLM. Классические Accuracy и F1 на свободном тексте
не работают: верный ответ можно сформулировать десятком способов, и посимвольное
сравнение забракует их все. Сильная модель в роли эксперта-оценщика сравнивает
смысл — это и есть LLM-as-a-judge (Л6).

Запуск:
    python test_suite.py
"""

import json
import re
import statistics
import uuid
from pathlib import Path

from contracts import Chunk
from llm_client import build_client
from orchestrator import Orchestrator
from prompts import format_chunks

DATASET = Path(__file__).parent / "gold_dataset.json"

JUDGE_TEMPERATURE = 0.0  # судья детерминирован: температура 0 (требование П6)

JUDGE_PROMPT = """Ты — строгий независимый эксперт-оценщик RAG-системы, которая отвечает на вопросы по пользовательским соглашениям, офертам и законодательству о защите прав потребителей. Сам на вопрос ты не отвечаешь: ты оцениваешь ответ системы.

На вход ты получаешь: тип вопроса, вопрос, эталонный ответ, ответ системы и найденные фрагменты — ровно те, что система передала своей модели (с названием файла и страницей).

Оцени три метрики НЕЗАВИСИМО друг от друга. Каждая — число от 0.0 до 1.0 (шаги 0.0, 0.25, 0.5, 0.75, 1.0). Не выставляй три одинаковых числа по общему впечатлению: ответ может быть верен источнику и при этом не о том, о чём спрашивали, а фрагменты могут быть плохими при хорошем ответе.

1. faithfulness — верность источнику.
Сверяй ответ ТОЛЬКО с найденными фрагментами, не с эталоном и не со своими знаниями. Разбери каждое утверждение ответа (срок, сумма, право, запрет, ссылка на файл и страницу, вердикт «скрытое условие есть / нет», «это законно / незаконно») и проверь, что оно прямо следует из фрагментов.
1.0 — каждое утверждение подтверждено фрагментами. 0.5 — часть утверждений подтверждена, часть добавлена от себя или искажена. 0.0 — ответ противоречит фрагментам или выдуман целиком. Вердикт о риске или законности должен опираться на текст найденных пунктов; вердикт без опоры на них — нарушение верности. Отказ «В базе знаний нет ответа на этот вопрос» ничего не утверждает, поэтому его faithfulness = 1.0.

2. answer_relevance — релевантность ответа вопросу.
Оценивай, отвечает ли ответ именно на заданный вопрос, сравнивая по СМЫСЛУ с эталоном, а не по словам. Верность источнику здесь не оценивается.
1.0 — ответ по существу совпадает с эталоном и отвечает на вопрос полностью. 0.5 — ответ отвечает частично, пропускает важную часть эталона или уходит в сторону. 0.0 — ответ не о том, о чём спрашивали, либо отказ там, где эталон содержит содержательный ответ.
Для вопроса типа cross_section ответ должен охватывать все части, названные в эталоне (по каждому источнику): охват одного источника из двух — не выше 0.5. Для вопроса типа ambiguous допустим ответ, который разбирает оба прочтения или прямо отмечает неоднозначность; ответ по одному разумному прочтению с опорой на текст — не ниже 0.5.

3. context_precision — точность поиска.
Оценивай ТОЛЬКО найденные фрагменты относительно ВОПРОСА (не ответа): какая доля фрагментов содержит информацию, нужную, чтобы на вопрос ответить. Не снижай оценку за то, что ответ не воспользовался подходящим фрагментом.
1.0 — все или почти все фрагменты относятся к вопросу. 0.5 — относится примерно половина. 0.0 — ни один фрагмент не относится к вопросу.

Правило для вопросов типа out_of_scope. Ответа в корпусе заведомо нет, поэтому эталоном служит отказ.
— context_precision не определён: верни 0.0 и напиши об этом в comment.
— faithfulness: проверь, не добавил ли ответ утверждений (сроков, цифр, норм), которых в корпусе заведомо нет. Чистый отказ — 1.0; любое выдуманное утверждение — не выше 0.25.
— answer_relevance: корректный отказ — 1.0; содержательный ответ вместо отказа — 0.0.

ФОРМАТ ВЫВОДА. Верни СТРОГО один JSON-объект и ничего больше: без пояснений до и после, без markdown-обрамления (без ```). Поля: faithfulness, answer_relevance, context_precision — числа; comment — строка на русском, 1–3 предложения: какое утверждение не подтверждено фрагментами (если есть) и почему выставлены именно такие оценки.

Пример вывода (in-scope):
{"faithfulness": 0.5, "answer_relevance": 1.0, "context_precision": 0.75, "comment": "Срок уведомления в ответе не найден ни в одном фрагменте — добавлен от себя. Ответ по смыслу совпадает с эталоном. Один из четырёх фрагментов относится к другому документу."}

Пример вывода (out_of_scope):
{"faithfulness": 1.0, "answer_relevance": 1.0, "context_precision": 0.0, "comment": "Корректный отказ, ничего не выдумано. context_precision не определён для out_of_scope."}
"""

_METRICS = ("faithfulness", "answer_relevance", "context_precision")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _extract_json(raw: str) -> dict:
    """Снять markdown-обрамление и вытащить JSON-объект из ответа судьи."""
    text = _FENCE.sub("", raw.strip()).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("в ответе судьи нет JSON-объекта")
    data = json.loads(text[start:end + 1])
    if not isinstance(data, dict):
        raise ValueError("JSON судьи — не объект")
    return data


def _score(value) -> float:
    """Привести оценку к float и проверить диапазон [0, 1]."""
    x = float(value)
    if not 0.0 <= x <= 1.0:  # NaN тоже не пройдёт это сравнение
        raise ValueError(f"оценка {value!r} вне диапазона 0..1")
    return x


def judge(llm, question: str, answer: str, expected: str,
          chunks: list[Chunk] | None = None,
          case_type: str = "direct") -> dict:
    """Оценить один ответ.

    Фрагменты подаются в промпт через prompts.format_chunks(): судья видит
    источник и страницу каждого и сверяет с ними верность (Faithfulness) и
    точность поиска (Context Precision). Для out_of_scope context_precision
    принудительно 0.0 с пометкой в комментарии.

    Возвращает {"faithfulness": float, "answer_relevance": float,
                "context_precision": float, "comment": str}.
    На сбое вызова или разбора — нули и comment с префиксом "сбой" или "разбор:",
    чтобы прогон не прерывался на одном ответе.
    """
    zeros = {m: 0.0 for m in _METRICS}

    user_content = (
        f"Тип вопроса: {case_type}\n\n"
        f"Вопрос:\n{question}\n\n"
        f"Эталонный ответ:\n{expected or '(эталон не задан)'}\n\n"
        f"Ответ системы:\n{answer}\n\n"
        f"Найденные фрагменты (то, что ушло в модель):\n"
        f"{format_chunks(chunks or [])}\n\n"
        "Верни только JSON."
    )
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        raw = llm.generate(messages, temperature=JUDGE_TEMPERATURE)
    except Exception as exc:  # сеть, лимиты, авторизация — прогон идёт дальше
        return {**zeros, "comment": f"сбой вызова судьи: {exc}"}

    try:
        data = _extract_json(raw)
        verdict = {m: _score(data[m]) for m in _METRICS}
        comment = str(data.get("comment", "")).strip()
    except (ValueError, KeyError, TypeError) as exc:  # JSONDecodeError ⊂ ValueError
        return {**zeros,
                "comment": f"разбор: {type(exc).__name__}: {exc}; "
                           f"ответ судьи: {str(raw)[:120]!r}"}

    if case_type == "out_of_scope":
        # Context Precision для вопросов вне корпуса не определён
        verdict["context_precision"] = 0.0
        if "context_precision" not in comment.lower():
            comment = (comment + " " if comment else "") + \
                "context_precision не определён для out_of_scope."
    return {**verdict, "comment": comment}


def run() -> None:
    """Прогнать датасет и напечатать отчёт. Готово, менять не нужно.

    In-scope и out_of_scope считаются раздельно: у out_of_scope context_precision
    не определён, а hit-rate@5 по ним не считается (см. evaluate_retrieval в
    ingest.py). Сбои разбора судьи не включаются в средние: один нечитаемый
    ответ не должен ронять среднюю оценку без изменения системы.
    """
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    if len(cases) != 10:
        print(f"ВНИМАНИЕ: в датасете {len(cases)} вопросов, нужно 10.\n")

    orchestrator = Orchestrator()
    judge_llm = build_client()

    in_scope = {"faithfulness": [], "answer_relevance": [], "context_precision": []}
    oop = {"faithfulness": [], "answer_relevance": [], "refusals": 0}
    parse_failures: list[int] = []

    print(f"Судья: {type(judge_llm).__name__}; "
          f"генератор: {type(orchestrator.llm).__name__}\n")

    try:
        for case in cases:
            # Своя сессия на вопрос: иначе память от прошлого вопроса исказит ответ
            session_id = f"test:{uuid.uuid4()}"
            answer = orchestrator.handle(session_id, case["question"])
            chunks = orchestrator._context.get(session_id, [])
            case_type = case.get("type", "direct")
            verdict = judge(judge_llm, case["question"], answer,
                            case.get("expected", ""), chunks, case_type)

            f = float(verdict.get("faithfulness", 0.0))
            ar = float(verdict.get("answer_relevance", 0.0))
            cp = float(verdict.get("context_precision", 0.0))
            comment = str(verdict.get("comment", ""))
            broken = comment.startswith(("разбор:", "сбой"))
            if broken:
                parse_failures.append(case["id"])

            if case_type == "out_of_scope":
                if not broken:
                    oop["faithfulness"].append(f)
                    oop["answer_relevance"].append(ar)
                    if f >= 0.5 and ar >= 0.5:
                        oop["refusals"] += 1
            else:
                if not broken:
                    in_scope["faithfulness"].append(f)
                    in_scope["answer_relevance"].append(ar)
                    in_scope["context_precision"].append(cp)

            flag = "ГАЛЛЮЦИНАЦИЯ" if (not broken and f < 0.5) else "ok"
            print(f"[{case['id']:>2}] {flag:<13} {case_type:<13} "
                  f"F={f:.2f} AR={ar:.2f} CP={cp:.2f}"
                  f"{'  СБОЙ РАЗБОРА' if broken else ''}")
            print(f"     вопрос: {case['question']}")
            print(f"     ответ:  {answer[:160]}")
            print(f"     судья:  {comment}\n")
    finally:
        orchestrator.close()

    print(" Сводка ")
    for metric, values in in_scope.items():
        if values:
            print(f"in-scope {metric:<18} {statistics.mean(values):.2f}  "
                  f"(n={len(values)})")
    if oop["faithfulness"]:
        print(f"out_of_scope F-среднее   {statistics.mean(oop['faithfulness']):.2f}")
        print(f"out_of_scope AR-среднее  {statistics.mean(oop['answer_relevance']):.2f}")
        print(f"out_of_scope отказов     {oop['refusals']}/{len(oop['faithfulness'])}")
    hallucinations = sum(1 for v in in_scope["faithfulness"] if v < 0.5)
    print(f"Галлюцинаций (F<0.5)     {hallucinations} "
          f"из {len(in_scope['faithfulness'])}")
    if parse_failures:
        print(f"Сбоев разбора: {len(parse_failures)} (id: {parse_failures})")


if __name__ == "__main__":
    run()

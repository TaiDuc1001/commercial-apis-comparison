from __future__ import annotations

import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent
TASKS_PATH = ROOT / "rag_50_tasks.jsonl"
OUTPUT_DIR = ROOT / "outputs"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODELS = [
    "anthropic/claude-opus-4.8",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.5",
    "openai/gpt-5.4-mini",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3.5-flash",
]
CHEAP_TEST_MODEL = "openai/gpt-5.4-mini"


@dataclass(frozen=True)
class RunConfig:
    name: str
    models: list[str]
    repetitions: int
    task_limit: int | None
    max_tokens: int = 2048
    timeout_seconds: float = 120.0
    retry_backoff_seconds: tuple[float, ...] = (5.0, 15.0)
    concurrency: int = 6


def load_tasks(limit: int | None = None) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with TASKS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks[:limit] if limit else tasks


def api_key() -> str:
    load_dotenv(ROOT / ".env")
    return os.environ["OPENROUTER_API_KEY"]


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def run_id(repetition: int, task_id: str, model: str) -> str:
    return f"rep{repetition:02d}__{task_id}__{model_slug(model)}"


def parse_json_object(content: str) -> tuple[dict[str, Any] | None, str | None]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    if not stripped.startswith("{"):
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, f"invalid json: {exc}"

    if not isinstance(value, dict):
        return None, "json root is not an object"
    return value, None


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_matches(expected: str, actual: str) -> bool:
    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)
    if expected_norm == actual_norm:
        return True
    if expected_norm in actual_norm or actual_norm in expected_norm:
        return True

    expected_words = set(expected_norm.split())
    actual_words = set(actual_norm.split())
    if not expected_words or not actual_words:
        return False
    overlap = len(expected_words & actual_words) / len(expected_words)
    numbers_expected = set(re.findall(r"\d+(?:\.\d+)?", expected_norm))
    numbers_actual = set(re.findall(r"\d+(?:\.\d+)?", actual_norm))
    key_terms_ok = not numbers_expected or numbers_expected.issubset(numbers_actual)
    return overlap >= 0.55 and key_terms_ok


def valid_schema(parsed: dict[str, Any] | None) -> bool:
    if parsed is None:
        return False
    if set(parsed) != {"answer", "citations", "confidence"}:
        return False
    if not isinstance(parsed["answer"], str):
        return False
    citations = parsed["citations"]
    if not isinstance(citations, list) or not all(isinstance(item, str) for item in citations):
        return False
    return parsed["confidence"] in {"low", "medium", "high"}


def normalize_citation(value: Any) -> str | None:
    match = re.search(r"\bC\d+\b", str(value or ""))
    return match.group(0) if match else None


def normalize_citations(values: list[Any]) -> list[str]:
    citations = []
    for value in values:
        citation = normalize_citation(value)
        if citation is not None:
            citations.append(citation)
    return sorted(set(citations))


def normalize_parsed_response(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    if parsed is None:
        return None
    normalized = dict(parsed)
    citations = normalized.get("citations")
    if isinstance(citations, list):
        normalized["citations"] = normalize_citations(citations)
    return normalized


def concise(answer: str) -> bool:
    return 1 <= len(answer.split()) <= 80


def score_response(task: dict[str, Any], content: str) -> dict[str, Any]:
    parsed, parse_error = parse_json_object(content)
    json_valid = valid_schema(parsed)
    expected_answer = task["expected_answer"]
    expected_citations = normalize_citations(task["expected_citations"])
    expected_confidence = task.get("expected_confidence")

    if not json_valid:
        return {
            "score": 0,
            "json_valid": False,
            "answer_correct": False,
            "faithful": False,
            "citation_correct": False,
            "concise": False,
            "parse_error": parse_error,
        }

    answer = parsed["answer"]
    citations = normalize_citations(parsed["citations"])
    parsed["citations"] = citations
    insufficient = expected_answer == "INSUFFICIENT_INFORMATION"

    if insufficient:
        abstain_correct = answer.strip() == "INSUFFICIENT_INFORMATION"
        no_hallucination = abstain_correct
        empty_citations = citations == []
        score = (
            int(abstain_correct) * 50
            + int(no_hallucination) * 30
            + int(empty_citations) * 10
            + int(json_valid) * 10
        )
        return {
            "score": score,
            "json_valid": json_valid,
            "answer_correct": abstain_correct,
            "faithful": no_hallucination,
            "citation_correct": empty_citations,
            "concise": concise(answer),
            "parse_error": None,
        }

    answer_correct = answer_matches(expected_answer, answer)
    citation_correct = citations == expected_citations
    known_citations = set(normalize_citations([chunk["id"] for chunk in task["context_chunks"]]))
    faithful = answer_correct and set(citations).issubset(known_citations)
    clarity = concise(answer)
    confidence_ok = expected_confidence is None or parsed["confidence"] == expected_confidence
    score = (
        int(answer_correct) * 40
        + int(faithful) * 25
        + int(citation_correct) * 20
        + int(json_valid and confidence_ok) * 10
        + int(clarity) * 5
    )
    return {
        "score": score,
        "json_valid": json_valid,
        "answer_correct": answer_correct,
        "faithful": faithful,
        "citation_correct": citation_correct,
        "concise": clarity,
        "parse_error": None,
    }


def send_request(model: str, prompt: str, config: RunConfig) -> tuple[dict[str, Any], float, int]:
    strict_prompt = (
        f"{prompt}\n\n"
        "Output constraints: return one compact JSON object only. "
        "Do not use markdown fences. Do not add explanation outside JSON. "
        "Keep the answer field concise."
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": strict_prompt}],
        "stream": False,
        "max_tokens": config.max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
    }

    started = time.perf_counter()
    with httpx.Client(timeout=config.timeout_seconds) as client:
        response = client.post(OPENROUTER_URL, headers=headers, json=payload)
    elapsed = time.perf_counter() - started
    data = response.json()
    if response.is_error or data.get("error"):
        message = data.get("error", response.text)
        raise RuntimeError(str(message))
    return data, elapsed, response.status_code


def extract_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def extract_usage(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
        "completion_tokens": usage.get("completion_tokens", 0) or 0,
        "cost_usd": usage.get("cost", 0.0) or 0.0,
    }


def run_single(task: dict[str, Any], model: str, repetition: int, config: RunConfig) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    errors: list[str] = []

    for attempt in range(1, len(config.retry_backoff_seconds) + 2):
        try:
            response, latency, http_status = send_request(model, task["prompt"], config)
            content = extract_content(response)
            usage = extract_usage(response)
            scored = score_response(task, content)
            parsed_response = normalize_parsed_response(parse_json_object(content)[0])
            return {
                "run_id": run_id(repetition, task["task_id"], model),
                "task_id": task["task_id"],
                "task_type": task["type"],
                "difficulty": task["difficulty"],
                "model": model,
                "repetition": repetition,
                "status": "success",
                "started_at_utc": started_at,
                "attempt_count": attempt,
                "latency_s": latency,
                "http_status": http_status,
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "cost_usd": usage["cost_usd"],
                "response": content,
                "parsed_response": parsed_response,
                "expected_answer": task["expected_answer"],
                "expected_citations": normalize_citations(task["expected_citations"]),
                **scored,
                "error": None,
            }
        except Exception as exc:
            errors.append(str(exc))
            if attempt <= len(config.retry_backoff_seconds):
                time.sleep(config.retry_backoff_seconds[attempt - 1])

    return {
        "run_id": run_id(repetition, task["task_id"], model),
        "task_id": task["task_id"],
        "task_type": task["type"],
        "difficulty": task["difficulty"],
        "model": model,
        "repetition": repetition,
        "status": "failed",
        "started_at_utc": started_at,
        "attempt_count": len(errors),
        "latency_s": 0.0,
        "http_status": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost_usd": 0.0,
        "response": "",
        "parsed_response": None,
        "expected_answer": task["expected_answer"],
        "expected_citations": normalize_citations(task["expected_citations"]),
        "score": 0,
        "json_valid": False,
        "answer_correct": False,
        "faithful": False,
        "citation_correct": False,
        "concise": False,
        "parse_error": None,
        "error": errors[-1] if errors else "unknown error",
    }


def csv_row(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": record["run_id"],
        "task_id": record["task_id"],
        "task_type": record["task_type"],
        "difficulty": record["difficulty"],
        "model": record["model"],
        "repetition": record["repetition"],
        "status": record["status"],
        "score": record["score"],
        "json_valid": record["json_valid"],
        "answer_correct": record["answer_correct"],
        "faithful": record["faithful"],
        "citation_correct": record["citation_correct"],
        "concise": record["concise"],
        "latency_s": f"{record['latency_s']:.6f}",
        "cost_usd": f"{float(record['cost_usd']):.8f}",
        "prompt_tokens": record["prompt_tokens"],
        "completion_tokens": record["completion_tokens"],
        "expected_answer": record["expected_answer"],
        "expected_citations": json.dumps(record["expected_citations"], ensure_ascii=False),
        "response": record["response"],
        "error": record["error"] or record["parse_error"] or "",
    }


def summarize(records: list[dict[str, Any]], config: RunConfig) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    by_type: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_model.setdefault(record["model"], []).append(record)
        by_type.setdefault(record["task_type"], []).append(record)

    def avg(rows: list[dict[str, Any]], key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0

    return {
        "config": asdict(config),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scheduled_runs": len(records),
        "successful_runs": sum(1 for record in records if record["status"] == "success"),
        "total_cost_usd": sum(float(record["cost_usd"]) for record in records),
        "overall_average_score": avg(records, "score"),
        "by_model": {
            model: {
                "runs": len(rows),
                "success_rate": sum(1 for row in rows if row["status"] == "success") / len(rows) * 100,
                "average_score": avg(rows, "score"),
                "average_latency_s": avg(rows, "latency_s"),
                "total_cost_usd": sum(float(row["cost_usd"]) for row in rows),
            }
            for model, rows in sorted(by_model.items())
        },
        "by_task_type": {
            task_type: {
                "runs": len(rows),
                "average_score": avg(rows, "score"),
                "json_valid_rate": sum(1 for row in rows if row["json_valid"]) / len(rows) * 100,
            }
            for task_type, rows in sorted(by_type.items())
        },
    }


def run_evaluation(config: RunConfig) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    tasks = load_tasks(config.task_limit)
    jobs: list[tuple[int, dict[str, Any], str, int]] = []
    records_by_index: dict[int, dict[str, Any]] = {}
    job_index = 0

    for repetition in range(1, config.repetitions + 1):
        for task in tasks:
            for model in config.models:
                jobs.append((job_index, task, model, repetition))
                job_index += 1

    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = {
            executor.submit(run_single, task, model, repetition, config): index
            for index, task, model, repetition in jobs
        }
        for future in as_completed(futures):
            index = futures[future]
            record = future.result()
            records_by_index[index] = record
            print(
                f"{record['run_id']} {record['status']} "
                f"score={record['score']} cost=${float(record['cost_usd']):.6f}"
            )

    records = [records_by_index[index] for index in sorted(records_by_index)]
    write_outputs(records, config)
    print(f"Wrote {OUTPUT_DIR / (config.name + '.csv')}")
    print(f"Wrote {OUTPUT_DIR / (config.name + '.json')}")


def write_outputs(records: list[dict[str, Any]], config: RunConfig) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    csv_path = OUTPUT_DIR / f"{config.name}.csv"
    json_path = OUTPUT_DIR / f"{config.name}.json"
    fieldnames = list(csv_row(records[0]).keys()) if records else []

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(csv_row(record))

    summary = summarize(records, config)
    summary["records"] = records
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")


def rescore_records(config: RunConfig) -> None:
    path = OUTPUT_DIR / f"{config.name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in load_tasks(config.task_limit)}
    records = []
    for record in data["records"]:
        scored = score_response(tasks[record["task_id"]], record["response"])
        parsed = normalize_parsed_response(parse_json_object(record["response"])[0])
        record.update(scored)
        record["parsed_response"] = parsed
        record["expected_citations"] = normalize_citations(tasks[record["task_id"]]["expected_citations"])
        records.append(record)
    write_outputs(records, config)


def rerun_invalid_results(config: RunConfig) -> None:
    path = OUTPUT_DIR / f"{config.name}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = {task["task_id"]: task for task in load_tasks(config.task_limit)}
    records = data["records"]
    jobs: list[tuple[int, dict[str, Any], str, int]] = []

    for index, record in enumerate(records):
        should_rerun = not record["json_valid"] or int(record["completion_tokens"]) >= 500
        if not should_rerun:
            continue

        task = tasks[record["task_id"]]
        jobs.append((index, task, record["model"], int(record["repetition"])))

    with ThreadPoolExecutor(max_workers=config.concurrency) as executor:
        futures = {
            executor.submit(run_single, task, model, repetition, config): index
            for index, task, model, repetition in jobs
        }
        for future in as_completed(futures):
            index = futures[future]
            new_record = future.result()
            records[index] = new_record
            print(
                f"reran {new_record['run_id']} {new_record['status']} "
                f"score={new_record['score']} cost=${float(new_record['cost_usd']):.6f}"
            )

    write_outputs(records, config)
    print(f"Replaced {len(jobs)} records in {OUTPUT_DIR / (config.name + '.csv')}")
    print(f"Updated {OUTPUT_DIR / (config.name + '.json')}")


def main() -> None:
    run_evaluation(
        RunConfig(
            name="test_results",
            models=[CHEAP_TEST_MODEL],
            repetitions=1,
            task_limit=5,
        )
    )


if __name__ == "__main__":
    main()

"""Run exposed original public tasks against a real, explicitly local model."""
import argparse
import json
from pathlib import Path
import time

from factor.local_models import LocalChatClient
from liner_stability.benchmarks import cases, score_answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="factor-qwen35-9b")
    parser.add_argument("--transport", default="openai", choices=["openai", "lmstudio-native"])
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to overwrite inference evidence")
    args.out.mkdir(parents=True)
    client = LocalChatClient(model=args.model, transport=args.transport)
    outputs, traces = [], []
    start = time.monotonic()
    response_format = {"type": "json_schema", "json_schema": {"name": "benchmark_answer",
        "strict": True, "schema": {"type": "object", "properties": {
            "answer": {"anyOf": [{"type": "number"}, {"type": "string"}, {"type": "null"}]},
            "unit": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
            "required": ["answer", "unit"], "additionalProperties": False}}}
    for task in cases():
        # Only the public prompt enters the request. Evaluator answer stays here.
        prompt = task["prompt"] + '\nReturn JSON only: {"answer": number or label, "unit": string or null}.'
        completion = client.complete([{"role": "user", "content": prompt}], max_tokens=256,
                                     response_format=response_format)
        traces.append({"case_id": task["case_id"], "completion": completion})
        try:
            answer = json.loads(completion["text"])
            if completion["finish_reason"] != "stop":
                raise ValueError("truncated completion")
            outputs.append({"case_id": task["case_id"], "answer": answer.get("answer"),
                            "unit": answer.get("unit")})
        except (ValueError, AttributeError):
            outputs.append({"case_id": task["case_id"], "answer": None, "unit": None})
        print(task["case_id"], completion["finish_reason"], flush=True)
    report = score_answers(cases(), outputs)
    report.update(model=args.model, elapsed_seconds=time.monotonic()-start,
                  transport=args.transport,
                  test_status="PUBLIC_DEVELOPMENT_ONLY", actual_local_inference=True,
                  claims="Structured answer checks only; not physics or agent validation")
    for name, value in [("predictions.json", outputs), ("traces.json", traces), ("metrics.json", report)]:
        (args.out/name).write_text(json.dumps(value, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

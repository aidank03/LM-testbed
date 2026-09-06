"""Small JSON contracts shared by inference, agents, and evaluation.

No domain evaluator is imported here. Model outputs confer no capabilities.
"""
from dataclasses import dataclass
import json
import math
from typing import Any


class ContractError(ValueError):
    pass


class PolicyError(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


def finite(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ContractError(f"{name} is below the supported minimum")
    if maximum is not None and value > maximum:
        raise ContractError(f"{name} exceeds the supported maximum")
    return value


def strict_json(text):
    def bad(value):
        raise ContractError("Nonfinite JSON numbers are prohibited")
    try:
        return json.loads(text, parse_constant=bad)
    except json.JSONDecodeError as exc:
        raise ContractError("Malformed JSON") from exc


def validate_schema(value, schema, path="arguments"):
    """Validate the limited JSON Schema subset used by our registered tools."""
    kinds = schema.get("type")
    kinds = [kinds] if isinstance(kinds, str) else kinds
    if value is None and kinds and "null" in kinds:
        return
    kind = next((k for k in kinds or [] if k != "null"), None)
    if kind == "object":
        if not isinstance(value, dict):
            raise ContractError(f"{path} must be an object")
        props = schema.get("properties", {})
        if set(schema.get("required", [])) - value.keys():
            raise ContractError(f"{path} is missing required fields")
        if schema.get("additionalProperties") is False and value.keys() - props.keys():
            raise ContractError(f"{path} has unknown fields")
        for key in value.keys() & props.keys():
            validate_schema(value[key], props[key], f"{path}.{key}")
    elif kind == "array":
        if not isinstance(value, list) or len(value) > schema.get("maxItems", 100):
            raise ContractError(f"{path} must be a bounded array")
        for item in value:
            validate_schema(item, schema["items"], path+"[]")
    elif kind == "string":
        if not isinstance(value, str) or len(value) > schema.get("maxLength", 8000):
            raise ContractError(f"{path} must be a bounded string")
        if len(value) < schema.get("minLength", 0):
            raise ContractError(f"{path} is too short")
    elif kind in ("number", "integer"):
        finite(value, path, minimum=schema.get("minimum"), maximum=schema.get("maximum"))
        if kind == "integer" and not isinstance(value, int):
            raise ContractError(f"{path} must be an integer")
    elif kind == "boolean":
        if not isinstance(value, bool):
            raise ContractError(f"{path} must be boolean")
    else:
        raise ContractError(f"Unsupported schema type at {path}")
    if "enum" in schema and value not in schema["enum"]:
        raise ContractError(f"{path} is outside the allowed choices")


@dataclass(frozen=True)
class Action:
    tool: str
    arguments: dict[str, Any]

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data) != {"tool", "arguments"}:
            raise ContractError("An action requires exactly tool and arguments")
        if not isinstance(data["tool"], str) or not isinstance(data["arguments"], dict):
            raise ContractError("Invalid action types")
        return cls(**data)


@dataclass(frozen=True)
class RunRequest:
    question: str
    observation: dict[str, Any]
    max_steps: int | None = None
    schema_version: str = "factor-request/1"

    def validate(self):
        if self.schema_version != "factor-request/1":
            raise ContractError("Unsupported request version")
        if not isinstance(self.question, str) or not 1 <= len(self.question) <= 8000:
            raise ContractError("Question must contain 1–8000 characters")
        if not isinstance(self.observation, dict):
            raise ContractError("Observation must be an object")
        if self.max_steps is not None:
            if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or not 1 <= self.max_steps <= 100:
                raise ContractError("max_steps must be an integer in 1–100")


@dataclass(frozen=True)
class RuntimePolicy:
    max_steps: int = 12
    max_tool_calls: int = 10
    max_seconds: float = 300
    max_cloud_cost_usd: float = 0
    allowed_providers: tuple[str, ...] = ("conventional", "local")
    allowed_tools: tuple[str, ...] = ("inspect_observation", "infer_motion", "finish")
    data_egress: bool = False
    max_payload_bytes: int = 40_000
    max_output_tokens: int = 800
    allow_hpc: bool = False

    def __post_init__(self):
        for name in ("max_steps", "max_tool_calls", "max_payload_bytes", "max_output_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContractError(f"{name} must be a positive integer")
        finite(self.max_seconds, "max_seconds", minimum=0.1)
        finite(self.max_cloud_cost_usd, "max_cloud_cost_usd", minimum=0)
        if not isinstance(self.data_egress, bool) or not isinstance(self.allow_hpc, bool):
            raise ContractError("Policy flags must be boolean")
        for name in ("allowed_tools", "allowed_providers"):
            values = getattr(self, name)
            if (not isinstance(values, (list, tuple)) or not values
                    or any(not isinstance(v, str) or not v for v in values)
                    or len(set(values)) != len(values)):
                raise ContractError(f"{name} must be a nonempty list of distinct names")

"""Trace logging helpers for agent runs."""

from typing import Any, Dict


class TraceLogger:
    """Collect a structured trace of a single agent interaction."""

    def __init__(self, prompt: str) -> None:
        self._trace: Dict[str, Any] = {
            "prompt": prompt,
            "llm_output": "",
            "tool_calls": [],
            "tool_results": [],
            "final_response": "",
        }

    def record_llm_output(self, llm_output: str) -> None:
        self._trace["llm_output"] = llm_output

    def record_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        self._trace["tool_calls"].append({"tool": tool_name, "arguments": arguments})

    def record_tool_result(self, result: Dict[str, Any]) -> None:
        self._trace["tool_results"].append(result)

    def record_final_response(self, response: str) -> None:
        self._trace["final_response"] = response

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._trace)

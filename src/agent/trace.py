"""Trace logging helpers for agent runs."""

from typing import Any, Dict, List, Optional


class TraceLogger:
    """Collect a structured trace of a single agent interaction."""

    @staticmethod
    def _normalize_retrieved_documents(
        retrieved_documents: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        normalized_documents: List[Dict[str, Any]] = []
        for index, doc in enumerate(retrieved_documents or [], start=1):
            normalized_documents.append(
                {
                    "doc_id": str(doc.get("doc_id", f"doc_{index}")),
                    "content": str(doc.get("content", "")),
                    "source": str(doc.get("source", "knowledge_base")),
                }
            )
        return normalized_documents

    def __init__(
        self,
        prompt: str,
        retrieved_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._trace: Dict[str, Any] = {
            "prompt": prompt,
            "retrieved_documents": self._normalize_retrieved_documents(
                retrieved_documents
            ),
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

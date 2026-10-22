"""Answer a question from the retrieved excerpts, with verified citations or a refusal.

Flow
    question
      -> retriever.search()               top-5 chunks
      -> Claude, told to call submit_answer(answer, citations[], refused)
      -> guardrails.validate()            ids real? quotes exact?
           ok   -> done
           fail -> send the problems back, one retry
           fail -> return a refusal ("couldn't verify"), never an unverified answer

Why a tool call instead of free text?
    Having the model call `submit_answer` gives us structured JSON every time
    (answer, citations, refused), so nothing has to be parsed out of prose.

The `naive` mode is the baseline for the evaluation: same retrieval, a plain
"answer using the context" prompt, no citations, no refusal rule, no checks.
It's what a first RAG prototype usually looks like.

Run one question:  python -m grounded.answer "How long do I have to report a fraud to get zero liability?"
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field

from grounded import config
from grounded.guardrails import validate

SYSTEM_GROUNDED = """You answer questions about Reserve Bank of India (RBI) regulations for bank customers and staff.
You are given numbered excerpts from three RBI documents. Use ONLY these excerpts, never outside knowledge.

Rules:
- Answer in 1-4 plain sentences. Include the specific numbers, time limits and conditions that apply.
- Support the answer with citations. Each citation is an excerpt id plus a short quote (under 40 words)
  copied EXACTLY, character for character, from that excerpt.
- If the excerpts do not contain the answer, set refused=true and say briefly that the documents
  don't cover it. Do not guess, and do not answer from general knowledge. Partial information
  that doesn't answer the question also means refused=true.
- Always respond by calling submit_answer."""

SYSTEM_NAIVE = """You answer questions about Reserve Bank of India (RBI) regulations.
Use the context provided to answer the question in 1-4 sentences."""

ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit the final answer with exact supporting quotes, or a refusal.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "The answer, or a one-line refusal"},
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string", "description": "Excerpt id, exactly as given"},
                        "quote": {"type": "string", "description": "Exact text copied from that excerpt"},
                    },
                    "required": ["chunk_id", "quote"],
                },
            },
            "refused": {"type": "boolean", "description": "true if the excerpts don't answer the question"},
        },
        "required": ["answer", "citations", "refused"],
    },
}


@dataclass
class Answer:
    question: str
    answer: str
    refused: bool
    citations: list[dict] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    verified: bool = True
    attempts: int = 1
    problems: list[list[str]] = field(default_factory=list)
    rejected: list[list[dict]] = field(default_factory=list)   # citations that failed the checks, per attempt
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


def format_excerpts(chunks) -> str:
    return "\n\n".join(f'<excerpt id="{c.id}">\n{c.text}\n</excerpt>' for c in chunks)


class ClaudeClient:
    """Thin wrapper so tests can swap in a fake with the same .create() method."""

    def __init__(self, model: str | None = None):
        import anthropic
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
        self.client = anthropic.Anthropic()
        self.model = model or config.LLM_MODEL

    def create(self, system: str, messages: list[dict], tools: list[dict] | None = None, max_tokens: int = 800):
        kwargs = {"model": self.model, "max_tokens": max_tokens, "system": system, "messages": messages}
        if tools:
            # Newer Claude models don't support *forcing* a specific tool, so the prompt
            # tells the model to always call it, and answer() handles the rare text reply.
            kwargs["tools"] = tools
        return self.client.messages.create(**kwargs)


def _tool_input(resp) -> dict | None:
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return dict(block.input)
    return None


def _text(resp) -> str:
    return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", None) == "text")


class Answerer:
    def __init__(self, retriever, llm, mode: str = "grounded", k: int = config.TOP_K, max_attempts: int = 2):
        self.retriever, self.llm, self.mode, self.k, self.max_attempts = retriever, llm, mode, k, max_attempts

    def answer(self, question: str) -> Answer:
        start = time.time()
        chunks = [c for c, _ in self.retriever.search(question, self.k)]
        user = f"Excerpts:\n\n{format_excerpts(chunks)}\n\nQuestion: {question}"
        result = Answer(question=question, answer="", refused=False, retrieved=[c.id for c in chunks])

        if self.mode == "naive":
            resp = self.llm.create(SYSTEM_NAIVE, [{"role": "user", "content": user}])
            result.answer = _text(resp).strip()
            result.verified = False
            result.input_tokens, result.output_tokens = resp.usage.input_tokens, resp.usage.output_tokens
            result.seconds = round(time.time() - start, 2)
            return result

        messages = [{"role": "user", "content": user}]
        for attempt in range(1, self.max_attempts + 1):
            resp = self.llm.create(SYSTEM_GROUNDED, messages, tools=[ANSWER_TOOL])
            result.input_tokens += resp.usage.input_tokens
            result.output_tokens += resp.usage.output_tokens
            out = _tool_input(resp)
            result.attempts = attempt
            assistant_turn = {"role": "assistant", "content": [
                b.model_dump(exclude_none=True) if hasattr(b, "model_dump") else b for b in resp.content]}
            if out is None:                                   # replied in prose instead of calling the tool
                result.problems.append(["did not call submit_answer"])
                messages += [assistant_turn, {"role": "user", "content": "Please respond by calling submit_answer."}]
                continue
            problems = validate(out, chunks)
            if not problems:
                result.answer, result.refused = out["answer"].strip(), bool(out["refused"])
                result.citations = out.get("citations") or []
                break
            result.problems.append(problems)
            result.rejected.append(out.get("citations") or [])
            # Show the model what failed and let it try once more.
            tool_id = next(b.id for b in resp.content if getattr(b, "type", None) == "tool_use")
            messages += [
                assistant_turn,
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "is_error": True,
                                              "content": "Rejected: " + "; ".join(problems) + ". Fix and resubmit."}]},
            ]
        else:
            # Still unverifiable after the retry: refuse rather than show an unchecked answer.
            result.answer = "I couldn't find a verifiable answer to this in the documents."
            result.refused, result.verified, result.citations = True, False, []
        result.seconds = round(time.time() - start, 2)
        return result


def build_answerer(chunking: str = "clause", retriever: str = "hybrid", mode: str = "grounded", model: str | None = None):
    from grounded.chunking import build_chunks
    from grounded.embed import CachedEmbedder, Embedder
    from grounded.retrieve import build_retriever

    embedder = CachedEmbedder(Embedder()) if retriever != "bm25" else None
    r = build_retriever(retriever, build_chunks(chunking), embedder)
    return Answerer(r, ClaudeClient(model), mode=mode)


def main():
    ap = argparse.ArgumentParser(description="Ask the policy assistant a question")
    ap.add_argument("question")
    ap.add_argument("--chunking", default=os.environ.get("GROUNDED_CHUNKING", "clause"))
    ap.add_argument("--retriever", default=os.environ.get("GROUNDED_RETRIEVER", "hybrid"))
    args = ap.parse_args()
    print(json.dumps(asdict(build_answerer(args.chunking, args.retriever).answer(args.question)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

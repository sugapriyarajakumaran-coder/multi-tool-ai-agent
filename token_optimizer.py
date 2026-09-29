"""
token_optimizer.py
-------------------
Demonstrates practical token-optimization techniques used when working with
LLM-based chat agents:

1. Token counting with tiktoken (the same style of tokenizer used by GPT models)
2. Redundancy / stop-word compression to cut prompt size before sending to an LLM
3. Sliding-window context management to keep a chatbot inside a token budget
4. Response caching to avoid re-spending tokens on repeated questions

Run directly (`python token_optimizer.py`) to see a demo report printed to
the console and saved to outputs/token_report.txt
"""

import hashlib
import json
import os
import re

# NOTE: tiktoken's real BPE vocab files are downloaded from OpenAI's CDN,
# which is unreachable from this sandboxed environment (no internet access
# to non-package-registry domains). We fall back to a small, self-contained
# regex tokenizer that mimics BPE-style splitting (words, sub-word chunks,
# punctuation, numbers) closely enough for token-count comparisons. Swap
# `count_tokens` for `len(tiktoken.get_encoding("cl100k_base").encode(text))`
# when running with full internet access / a real OpenAI tokenizer.
_TOKEN_PATTERN = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")


def _encode(text: str):
    """Lightweight local stand-in for a BPE tokenizer's .encode()."""
    tokens = []
    for word in _TOKEN_PATTERN.findall(text):
        if len(word) <= 4 or not word.isalpha():
            tokens.append(word)
        else:
            # crude sub-word chunking, similar in spirit to BPE splitting
            # long words into ~4-char pieces
            for i in range(0, len(word), 4):
                tokens.append(word[i:i + 4])
    return tokens


class _LocalEncoding:
    @staticmethod
    def encode(text):
        return _encode(text)


ENCODING = _LocalEncoding()

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on",
    "and", "or", "that", "this", "it", "as", "with", "for", "be", "by",
    "at", "so", "very", "really", "just", "actually", "basically", "like"
}


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def compress_prompt(text: str) -> str:
    """
    Cheap, reversible-ish compression: strips filler / stop words and
    collapses whitespace. Good for system prompts / instructions where
    exact grammar doesn't matter, NOT for content the user must read back.
    """
    words = text.split()
    kept = [w for w in words if w.lower().strip(".,!?") not in STOPWORDS]
    return " ".join(kept)


class SlidingWindowMemory:
    """
    Keeps a chat history under a fixed token budget by dropping the oldest
    turns first (FIFO), always preserving the system prompt.
    """

    def __init__(self, system_prompt: str, max_tokens: int = 200):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.turns = []  # list of (role, text)

    def add(self, role: str, text: str):
        self.turns.append((role, text))
        self._trim()

    def _trim(self):
        while self._total_tokens() > self.max_tokens and len(self.turns) > 1:
            self.turns.pop(0)  # drop oldest turn

    def _total_tokens(self) -> int:
        history_text = " ".join(t for _, t in self.turns)
        return count_tokens(self.system_prompt) + count_tokens(history_text)

    def render(self) -> str:
        lines = [f"[system] {self.system_prompt}"]
        lines += [f"[{role}] {text}" for role, text in self.turns]
        return "\n".join(lines)


class ResponseCache:
    """
    Simple hash-based cache: identical prompts never get re-processed /
    re-billed. In production this would be Redis or a vector-similarity
    cache for near-duplicate prompts.
    """

    def __init__(self):
        self._store = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(prompt: str) -> str:
        return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()

    def get_or_compute(self, prompt: str, compute_fn):
        key = self._key(prompt)
        if key in self._store:
            self.hits += 1
            return self._store[key], True
        self.misses += 1
        result = compute_fn(prompt)
        self._store[key] = result
        return result, False

def optimize_prompt(prompt: str):
    """
    Optimize a user-provided prompt.

    Returns:
        original prompt
        compressed prompt
        original token count
        compressed token count
        savings
        savings percentage
    """
    raw_tokens = count_tokens(prompt)

    compressed = compress_prompt(prompt)

    compressed_tokens = count_tokens(compressed)

    if raw_tokens > 0:
        savings_pct = (raw_tokens - compressed_tokens) / raw_tokens
    else:
        savings_pct = 0

    return {
        "original_prompt": prompt,
        "compressed_prompt": compressed,
        "raw_tokens": raw_tokens,
        "compressed_tokens": compressed_tokens,
        "tokens_saved": raw_tokens - compressed_tokens,
        "savings_pct": savings_pct,
    }
    
def demo():
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(str(msg))

    log("=== 1. Token counting ===")
    sample = (
        "Hello! I was just wondering if you could actually help me "
        "understand what the weather is going to be like tomorrow, please."
    )
    raw_tokens = count_tokens(sample)
    compressed = compress_prompt(sample)
    comp_tokens = count_tokens(compressed)
    log(f"Original ({raw_tokens} tokens): {sample}")
    log(f"Compressed ({comp_tokens} tokens): {compressed}")
    log(f"Savings: {raw_tokens - comp_tokens} tokens "
        f"({(raw_tokens - comp_tokens) / raw_tokens:.0%})")

    log("\n=== 2. Sliding-window context management ===")
    mem = SlidingWindowMemory(
        system_prompt="You are a concise, helpful assistant.",
        max_tokens=60,
    )
    convo = [
        ("user", "Hi, can you tell me about the solar system?"),
        ("assistant", "Sure! It has 8 planets orbiting the Sun."),
        ("user", "Which planet is the biggest?"),
        ("assistant", "Jupiter is the largest planet."),
        ("user", "And the smallest one?"),
        ("assistant", "Mercury is the smallest planet."),
        ("user", "What about dwarf planets like Pluto?"),
    ]
    for role, text in convo:
        mem.add(role, text)
        log(f"  added [{role}] -> window tokens = {mem._total_tokens()}")
    log("\nFinal context window sent to the model:")
    log(mem.render())

    log("\n=== 3. Response caching ===")
    cache = ResponseCache()

    def fake_llm_call(prompt):
        return f"[computed answer for: '{prompt[:30]}...']"

    prompts = [
        "What is the capital of France?",
        "What is the capital of France?",  # duplicate -> cache hit
        "What is the capital of Japan?",
        "what is the capital of france?",  # different case -> still a hit
    ]
    for p in prompts:
        result, was_cached = cache.get_or_compute(p, fake_llm_call)
        log(f"  prompt='{p}' -> {'CACHE HIT' if was_cached else 'COMPUTED'}")
    log(f"\nCache stats: {cache.hits} hits, {cache.misses} misses "
        f"({cache.hits / (cache.hits + cache.misses):.0%} hit rate)")
    log(f"Tokens saved by caching: ~{cache.hits * count_tokens(prompts[0])} tokens")

    OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
    os.makedirs(OUT, exist_ok=True)
    out_path = os.path.join(OUT, "token_report.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(report_lines))

    summary = {
        "raw_tokens": raw_tokens,
        "compressed_tokens": comp_tokens,
        "savings_pct": round((raw_tokens - comp_tokens) / raw_tokens, 3),
        "cache_hit_rate": round(cache.hits / (cache.hits + cache.misses), 3),
    }
    with open(os.path.join(OUT, "token_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    demo()

import hashlib
from typing import Optional
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import pipeline
from dotenv import load_dotenv

load_dotenv()

print("Loading local model (first run may download ~200MB)...")
_pipe = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=100,
    device=-1,  # CPU
)
LOCAL_LLM = HuggingFacePipeline(pipeline=_pipe)
print("Model ready.\n")


# === Model Routing ===

class ModelRouter:

    def __init__(self):
        self.cheap_pipe = pipeline("text-generation", model="gpt2", max_new_tokens=50, device=-1)
        self.expensive_pipe = pipeline("text-generation", model="gpt2", max_new_tokens=150, device=-1)
        self.cheap_model = HuggingFacePipeline(pipeline=self.cheap_pipe)
        self.expensive_model = HuggingFacePipeline(pipeline=self.expensive_pipe)

    def classify_complexity(self, query: str) -> str:
        word_count = len(query.split())
        has_complex_words = any(w in query.lower() for w in [
            "analyze", "explain", "compare", "evaluate", "implications", "reasoning"
        ])
        return "complex" if word_count > 10 or has_complex_words else "simple"

    def invoke(self, query: str) -> tuple[str, str, float]:
        complexity = self.classify_complexity(query)

        if complexity == "simple":
            model = self.cheap_model
            model_name = "flan-t5-small (cheap)"
            cost_per_1k = 0.0001
        else:
            model = self.expensive_model
            model_name = "flan-t5-base (expensive)"
            cost_per_1k = 0.0005

        response = model.invoke(query)
        tokens = len(query.split()) * 1.3
        estimated_cost = (tokens / 1000) * cost_per_1k

        return response, model_name, estimated_cost


def demo_model_routing():
    router = ModelRouter()

    queries = [
        "What is 2 + 2?",
        "Analyze the economic implications of AI on the job market.",
        "What color is the sky?",
    ]

    print("=" * 60)
    print("Model Routing Demo:\n")

    total_cost = 0
    for query in queries:
        result, model, cost = router.invoke(query)
        total_cost += cost
        print(f"Query:     {query}")
        print(f"  Model:     {model}")
        print(f"  Est. Cost: ${cost:.6f}")
        print(f"  Response:  {str(result)[:80]}")
        print()

    print(f"Total Estimated Cost: ${total_cost:.6f}")


# === Semantic Caching ===

class SemanticCache:
    """Cache responses using normalized query hashing."""

    def __init__(self):
        self.cache = {}

    def _hash_query(self, query: str) -> str:
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[str]:
        query_hash = self._hash_query(query)
        if query_hash in self.cache:
            return self.cache[query_hash]["response"]
        return None

    def set(self, query: str, response: str):
        query_hash = self._hash_query(query)
        self.cache[query_hash] = {"query": query, "response": response}


class CachedLLM:
    """LLM wrapper with caching to avoid redundant calls."""

    def __init__(self):
        self.llm = LOCAL_LLM
        self.cache = SemanticCache()
        self.cache_hits = 0
        self.cache_misses = 0

    def invoke(self, query: str) -> tuple[str, bool]:
        cached = self.cache.get(query)
        if cached:
            self.cache_hits += 1
            return cached, True

        self.cache_misses += 1
        response = self.llm.invoke(query)
        self.cache.set(query, response)
        return response, False

    def get_stats(self) -> dict:
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.1%}",
        }


def demo_caching():
    llm = CachedLLM()

    queries = [
        "What is Python?",
        "What is JavaScript?",
        "What is Python?",   # Cache hit
        "What is python?",   # Cache hit (normalized)
        "What is Rust?",
    ]

    print("\n" + "=" * 60)
    print("Caching Demo:\n")

    for query in queries:
        result, from_cache = llm.invoke(query)
        source = "CACHE" if from_cache else "LLM  "
        print(f"[{source}] {query:<30} -> {str(result)[:60]}")

    print(f"\nStats: {llm.get_stats()}")


# === Token Budgeting ===

class TokenBudget:
    """Track and enforce per-request token limits."""

    def __init__(self, max_tokens_per_request: int = 4000):
        self.max_per_request = max_tokens_per_request
        self.usage = {"total_input": 0, "total_output": 0, "requests": 0}

    def estimate_tokens(self, text: str) -> int:
        return int(len(text.split()) * 1.3)

    def check_budget(self, text: str) -> tuple[bool, int]:
        tokens = self.estimate_tokens(text)
        return tokens <= self.max_per_request, tokens

    def record_usage(self, input_tokens: int, output_tokens: int):
        self.usage["total_input"] += input_tokens
        self.usage["total_output"] += output_tokens
        self.usage["requests"] += 1

    def get_stats(self) -> dict:
        total = self.usage["total_input"] + self.usage["total_output"]
        return {
            **self.usage,
            "total_tokens": total,
            "avg_per_request": total / max(self.usage["requests"], 1),
        }


class BudgetedLLM:
    """LLM with per-request token budget enforcement."""

    def __init__(self, max_tokens: int = 4000):
        self.llm = LOCAL_LLM
        self.budget = TokenBudget(max_tokens_per_request=max_tokens)

    def invoke(self, query: str) -> str:
        within_budget, tokens = self.budget.check_budget(query)

        if not within_budget:
            raise ValueError(
                f"Query exceeds token budget: {tokens} > {self.budget.max_per_request}"
            )

        response = self.llm.invoke(query)
        output_tokens = self.budget.estimate_tokens(str(response))
        self.budget.record_usage(tokens, output_tokens)
        return response

    def get_stats(self) -> dict:
        return self.budget.get_stats()


def demo_token_budgeting():
    llm = BudgetedLLM(max_tokens=100)

    queries = [
        "What is AI?",
        "Explain " + "very " * 100 + "complex topic",  # Over budget
    ]

    print("\n" + "=" * 60)
    print("Token Budgeting Demo:\n")

    for query in queries:
        try:
            result = llm.invoke(query)
            print(f"✅ {query[:50]} -> {str(result)[:60]}")
        except ValueError as e:
            print(f"❌ {query[:50]}... -> {e}")

    print(f"\nUsage stats: {llm.get_stats()}")


if __name__ == "__main__":
    demo_model_routing()
    demo_caching()
    demo_token_budgeting()
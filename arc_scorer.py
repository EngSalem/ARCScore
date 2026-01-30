"""
ARCScore: Atomic Recall Computation Scorer

This module provides an independent scoring object that:
1. Takes article elements (components with types)
2. Retrieves their decomposition from cache if available
3. Computes recall via prompt-based LLM evaluation

The score is based on how many atomic facts in the reference are recalled in the generated summary.
"""

import os
import json
import asyncio
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from openai import OpenAI
import openai
import time
from tqdm.asyncio import tqdm_asyncio


@dataclass
class AtomicFact:
    """Represents an atomic fact with metadata."""
    fact_id: str
    text: str
    component_type: str  # e.g., "argument", "evidence", "claim"
    source: str  # article name or identifier


@dataclass
class DecompositionResult:
    """Result of decomposing an article element."""
    article_id: str
    component_type: str
    component_text: str
    atomic_facts: List[AtomicFact]
    decomposed_at: float  # timestamp


class DecompositionCache:
    """
    Manages caching of atomic fact decompositions.
    
    Cache structure (cache.json):
    {
        "article_id__component_type": {
            "component_text": "...",
            "atomic_facts": [{"fact_id": "...", "text": "...", ...}, ...],
            "decomposed_at": 1234567890
        }
    }
    """

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "decompositions.json")
        os.makedirs(cache_dir, exist_ok=True)
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        """Load cache from disk."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        """Save cache to disk."""
        with open(self.cache_file, "w") as f:
            json.dump(self._cache, f, indent=2)

    def _make_key(self, article_id: str, component_type: str) -> str:
        """Generate a cache key."""
        return f"{article_id}__{component_type}"

    def get(self, article_id: str, component_type: str) -> Optional[DecompositionResult]:
        """Retrieve a decomposition from cache."""
        # Backwards-compatible single-key lookup
        key = self._make_key(article_id, component_type)
        if key in self._cache:
            data = self._cache[key]
            facts = [AtomicFact(**f) for f in data["atomic_facts"]]
            return DecompositionResult(
                article_id=article_id,
                component_type=component_type,
                component_text=data["component_text"],
                atomic_facts=facts,
                decomposed_at=data["decomposed_at"]
            )

        # If keys are indexed (e.g. article__Type__0), return the first matching decomposition
        prefix = f"{article_id}__{component_type}__"
        for k, data in self._cache.items():
            if k.startswith(prefix):
                facts = [AtomicFact(**f) for f in data["atomic_facts"]]
                return DecompositionResult(
                    article_id=article_id,
                    component_type=component_type,
                    component_text=data["component_text"],
                    atomic_facts=facts,
                    decomposed_at=data["decomposed_at"]
                )
        return None

    def get_components(self, article_id: str, component_type: str) -> List[DecompositionResult]:
        """Retrieve all decomposition instances for an article+component_type.

        This supports indexed keys like `article__Type__0`, `article__Type__1`, etc.
        Returns an empty list if none found.
        """
        prefix = f"{article_id}__{component_type}__"
        results: List[DecompositionResult] = []
        for k, data in self._cache.items():
            if k == f"{article_id}__{component_type}" or k.startswith(prefix):
                facts = [AtomicFact(**f) for f in data["atomic_facts"]]
                results.append(DecompositionResult(
                    article_id=article_id,
                    component_type=component_type,
                    component_text=data.get("component_text", ""),
                    atomic_facts=facts,
                    decomposed_at=data.get("decomposed_at", 0)
                ))
        return results

    def get_all_components(self, article_id: str) -> Dict[str, List[DecompositionResult]]:
        """Retrieve all decomposition instances for an article grouped by component_type.

        Returns a dict: component_type -> list[DecompositionResult]
        """
        results: Dict[str, List[DecompositionResult]] = {}
        prefix = f"{article_id}__"
        for k, data in self._cache.items():
            if not k.startswith(prefix):
                continue
            # key format: article__Component__idx or article__Component
            parts = k.split("__")
            if len(parts) < 2:
                continue
            comp = parts[1]
            facts = [AtomicFact(**f) for f in data["atomic_facts"]]
            decomp = DecompositionResult(
                article_id=article_id,
                component_type=comp,
                component_text=data.get("component_text", ""),
                atomic_facts=facts,
                decomposed_at=data.get("decomposed_at", 0)
            )
            results.setdefault(comp, []).append(decomp)
        return results

    def put(self, decomposition: DecompositionResult):
        """Cache a decomposition."""
        key = self._make_key(decomposition.article_id, decomposition.component_type)
        self._cache[key] = {
            "component_text": decomposition.component_text,
            "atomic_facts": [asdict(f) for f in decomposition.atomic_facts],
            "decomposed_at": decomposition.decomposed_at
        }
        self._save_cache()

    def clear(self):
        """Clear all cache."""
        self._cache = {}
        self._save_cache()

    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            "total_entries": len(self._cache),
            "cache_file": self.cache_file
        }


class PromptLoader:
    """Load prompt templates from prompts/ directory."""

    def __init__(self, prompts_dir: str = None):
        if prompts_dir is None:
            prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
        self.prompts_dir = prompts_dir
        self._prompts = {}

    def load_prompt(self, name: str) -> str:
        """Load a prompt template by name (e.g., 'fact_verifier' -> 'fact_verifier.md')."""
        if name in self._prompts:
            return self._prompts[name]
        
        prompt_file = os.path.join(self.prompts_dir, f"{name}.md")
        if not os.path.exists(prompt_file):
            raise FileNotFoundError(f"Prompt not found: {prompt_file}")
        
        with open(prompt_file, "r") as f:
            prompt = f.read()
        self._prompts[name] = prompt
        return prompt


class ARCScorer:
    """
    Main Atomic Recall Computation scorer.
    
    Workflow:
    1. Initialize with LLM credentials and cache
    2. For each article element (component), retrieve or compute its atomic decomposition
    3. Compare atomic facts against a summary using LLM-based fact verification
    4. Compute recall: (facts verified in summary) / (total facts in reference)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        cache_dir: str = None,
        prompts_dir: str = None,
        temperature: float = 0,
        is_reasoning: bool = False,
        verbose: bool = False
    ):
        """
        Initialize ARCScorer.
        
        Args:
            api_key: OpenAI API key (or other LLM provider)
            base_url: Base URL for LLM API
            model_name: Model to use for fact verification
            cache_dir: Directory for decomposition cache (default: ARCScore/cache)
            prompts_dir: Directory for prompt templates (default: ARCScore/prompts)
            temperature: Decoding temperature (0 = greedy)
            is_reasoning: Whether model supports reasoning (e.g., o1)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.is_reasoning = is_reasoning
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.cache = DecompositionCache(cache_dir)
        self.prompt_loader = PromptLoader(prompts_dir)
        self.decomposition_fn = None  # Optional callback for decomposition on cache miss
        self.verbose = verbose # if true print the output of the LLM verifier

    async def verify_fact_in_summary(self, fact: str, summary: str) -> Tuple[int, str]:
        """
        Verify if a fact is present/supported in a summary using LLM.
        
        Returns:
            (decision, reason) where decision ∈ {1, 0} (1=supported, 0=not-supported/missing)
        """
        prompt = self.prompt_loader.load_prompt("atomic_fact_binary_verifier")
        formatted_prompt = prompt.format(argument=fact, summary=summary)

        messages = [
            {"role": "system", "content": "You are a helpful assistant that verifies if facts are present in texts."},
            {"role": "user", "content": formatted_prompt}
        ]

        max_tokens = 4096 if self.is_reasoning else 512
        retry_interval_exp = 1

        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )
                response = completion.choices[0].message.content
                return self._parse_verification_response(response), response

            except openai.RateLimitError:
                wait_time = min(max(4, 0.5 * (2 ** retry_interval_exp)), 1024)
                print(f"Rate limit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                retry_interval_exp += 1

            except (openai.APIConnectionError, openai.APIError) as e:
                print(f"API error: {e}. Retrying in 1s...")
                time.sleep(1)

            except Exception as e:
                print(f"Error verifying fact: {e}")
                return 0, None

    def _parse_verification_response(self, response: str) -> int:
        """Parse LLM response to extract binary decision (1 or 0)."""
        response_lower = response.lower()
        if self.verbose:
            print("Verification response:", response)
        # Look for explicit markers
        if "(1, 'supported')" in response or '(1, "supported")' in response:
            return 1
        if "(0, 'missing')" in response or '(0, "missing")' in response or \
           "(0, 'not-factual')" in response or '(0, "not-factual")' in response:
            return 0
        
        # Fallback: heuristic keywords
        if "supported" in response_lower and "not" not in response_lower:
            return 1
        if "missing" in response_lower or "not found" in response_lower or "not supported" in response_lower:
            return 0
        
        # Default
        return 0

    async def compute_recall_batch(
        self,
        atomic_facts: List[AtomicFact],
        summary: str,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Compute recall for a batch of atomic facts against a summary.
        
        Recall = (# facts verified in summary) / (total facts)
        
        Args:
            atomic_facts: List of atomic facts to verify
            summary: Generated summary to check against
            batch_size: Async batch size
            
        Returns:
            {
                "total_facts": N,
                "verified_facts": M,
                "recall": M/N,
                "verification_results": [...]
            }
        """
        if not atomic_facts:
            return {
                "total_facts": 0,
                "verified_facts": 0,
                "recall": 0.0,
                "verification_results": []
            }

        # Split into batches
        batches = []
        for i in range(0, len(atomic_facts), batch_size):
            batches.append(atomic_facts[i:i+batch_size])

        verified_count = 0
        results = []

        # Process batches asynchronously
        for batch in tqdm_asyncio(batches, desc="Verifying facts", unit="batch"):
            tasks = [self.verify_fact_in_summary(fact.text, summary) for fact in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for fact, (decision, response) in zip(batch, batch_results):
                if isinstance(decision, int):
                    verified_count += decision
                    results.append({
                        "fact_id": fact.fact_id,
                        "fact_text": fact.text,
                        "decision": decision,
                        "response": response[:200] if response else None  # truncate for storage
                    })
                else:
                    results.append({
                        "fact_id": fact.fact_id,
                        "fact_text": fact.text,
                        "decision": 0,
                        "error": str(decision)
                    })

        total_facts = len(atomic_facts)
        recall = verified_count / total_facts if total_facts > 0 else 0.0

        return {
            "total_facts": total_facts,
            "verified_facts": verified_count,
            "recall": recall,
            "verification_results": results
        }

    # async def score_summary(
    #     self,
    #     article_id: str,
    #     summary: str,
    #     batch_size: int = 10
    # ) -> Dict[str, Any]:
    #     """
    #     Score a generated summary against an article (bottom-up recall).
        
    #     Process:
    #     1. Load all component types and instances for the article from cache
    #     2. If not found and decomposition_fn is set, call it to load/compute decompositions
    #     3. For each component instance, verify atomic facts against the summary
    #     4. Compute per-component recall and return average across all components
        
    #     Args:
    #         article_id: Unique identifier for the article
    #         summary: Generated summary to evaluate
    #         batch_size: Batch size for async fact verification
            
    #     Returns:
    #         {
    #             "article_id": ...,
    #             "summary_length": ...,
    #             "num_components": ...,
    #             "average_recall": ...,
    #             "per_component_results": [
    #                 {
    #                     "component_type": ...,
    #                     "component_text": ...,
    #                     "total_facts": ...,
    #                     "verified_facts": ...,
    #                     "recall": ...,
    #                     "verification_results": [...]
    #                 },
    #                 ...
    #             ]
    #         }
    #     """
    #     # Retrieve all components for the article
    #     all_components = self.cache.get_all_components(article_id)

    #     # If not found in cache and decomposition_fn is set, try decomposition
    #     if not all_components and self.decomposition_fn:
    #         try:
    #             print(f"Article {article_id} not in cache. Running decomposition...")
    #             all_components = self.decomposition_fn(article_id)
    #         except Exception as e:
    #             print(f"Error during decomposition for {article_id}: {e}")
    #             all_components = {}

    #     per_component_results = []
    #     # Iterate over component types and their instances
    #     for comp_type, instances in all_components.items():
    #         for instance in instances:
    #             recall_result = await self.compute_recall_batch(instance.atomic_facts, summary, batch_size)
    #             per_component_results.append({
    #                 "component_type": comp_type,
    #                 "component_text": instance.component_text,
    #                 "total_facts": recall_result["total_facts"],
    #                 "verified_facts": recall_result["verified_facts"],
    #                 "recall": recall_result["recall"],
    #                 "verification_results": recall_result["verification_results"]
    #             })

    #     # Compute average recall across all component instances
    #     if per_component_results:
    #         avg_recall = sum(r.get("recall", 0.0) for r in per_component_results) / len(per_component_results)
    #     else:
    #         avg_recall = 0.0

    #     return {
    #         "article_id": article_id,
    #         "summary_length": len(summary.split()),
    #         "num_components": len(per_component_results),
    #         "average_recall": avg_recall,
    #         "per_component_results": per_component_results
    #     }
    
    async def score_summary(
        self,
        article_id: str,
        summary: str,
    ) -> Dict[str, Any]:
        """
        Score a generated summary against an article by:
        1. Loading ALL components for the article.
        2. Flattening ALL atomic facts across all components.
        3. Firing ONE big asyncio.gather for verify_fact_in_summary calls.
        4. Reconstructing per-component recall.
        """

        # ---------------------------------------------------------
        # 1. Load components from cache or decomposition_fn
        # ---------------------------------------------------------
        all_components = self.cache.get_all_components(article_id)

        if not all_components and self.decomposition_fn:
            try:
                print(f"Article {article_id} not in cache. Running decomposition...")
                all_components = self.decomposition_fn(article_id)
            except Exception as e:
                print(f"Error during decomposition for {article_id}: {e}")
                all_components = {}

        # Convert to a list of component instances to preserve ordering
        comp_instances = []
        for comp_type, instances in all_components.items():
            for inst in instances:
                comp_instances.append(inst)

        if not comp_instances:
            return {
                "article_id": article_id,
                "summary_length": len(summary.split()),
                "num_components": 0,
                "average_recall": 0.0,
                "per_component_results": [],
            }

        # ---------------------------------------------------------
        # 2. Flatten ALL facts across ALL components
        #    flat_facts[i] = (comp_idx, AtomicFact)
        # ---------------------------------------------------------
        flat_facts = []
        for comp_idx, inst in enumerate(comp_instances):
            for fact in inst.atomic_facts:
                flat_facts.append((comp_idx, fact))

        if not flat_facts:
            # No facts = trivial zero recall for all components
            per_component_results = []
            for inst in comp_instances:
                per_component_results.append({
                    "component_type": inst.component_type,
                    "component_text": inst.component_text,
                    "total_facts": 0,
                    "verified_facts": 0,
                    "recall": 0.0,
                    "verification_results": [],
                })

            return {
                "article_id": article_id,
                "summary_length": len(summary.split()),
                "num_components": len(comp_instances),
                "average_recall": 0.0,
                "per_component_results": per_component_results,
            }

        # ---------------------------------------------------------
        # 3. Fire ONE global aio gather for ALL facts
        # ---------------------------------------------------------
        tasks = [
            self.verify_fact_in_summary(fact.text, summary)
            for (_, fact) in flat_facts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
     
        # ---------------------------------------------------------
        # 4. Reconstruct per-component statistics
        # ---------------------------------------------------------
        comp_stats = [
            {"total": 0, "verified": 0, "results": []}
            for _ in comp_instances
        ]

        for (comp_idx, fact), res in zip(flat_facts, results):
            comp_stats[comp_idx]["total"] += 1

            # Handle exceptions
            if isinstance(res, Exception):
                comp_stats[comp_idx]["results"].append({
                    "fact_id": fact.fact_id,
                    "fact_text": fact.text,
                    "decision": 0,
                    "error": str(res),
                })
                continue

            decision, response = res
            if not isinstance(decision, int):
                decision = 0

            comp_stats[comp_idx]["verified"] += decision

            comp_stats[comp_idx]["results"].append({
                "fact_id": fact.fact_id,
                "fact_text": fact.text,
                "decision": decision,
                "response": response[:200] if response else None,
            })

        # ---------------------------------------------------------
        # 5. Build per-component results w/ recall
        # ---------------------------------------------------------
        per_component_results = []
        for inst, stats in zip(comp_instances, comp_stats):
            total = stats["total"]
            verified = stats["verified"]
            recall = verified / total if total > 0 else 0.0

            per_component_results.append({
                "component_type": inst.component_type,
                "component_text": inst.component_text,
                "total_facts": total,
                "verified_facts": verified,
                "recall": recall,
                "verification_results": stats["results"],
            })

        # ---------------------------------------------------------
        # 6. Compute article-level average recall
        # ---------------------------------------------------------
        avg_recall = (
            sum(r["recall"] for r in per_component_results) / len(per_component_results)
            if per_component_results else 0.0
        )

        return {
            "article_id": article_id,
            "summary_length": len(summary.split()),
            "num_components": len(per_component_results),
            "average_recall": avg_recall,
            "per_component_results": per_component_results,
        }


    async def _verify_facts_chunk(self, texts: List[str], summary: str) -> Dict[str, Dict[str, Any]]:
        """
        Verify a chunk of unique fact texts against the summary in a single LLM call.

        Returns a mapping: fact_text -> {"decision": 0|1, "response": raw_response}
        """
        if not texts:
            return {}

        # Build a concise prompt enumerating facts and asking for JSON result
        enumerated = "\n".join([f"{i+1}) {t}" for i, t in enumerate(texts)])
        prompt_template = (
            "Summary:\n{summary}\n\n"
            "Facts to verify:\n{facts}\n\n"
            "Instruction: For each numbered fact above, output a JSON object mapping the index to 1 if the fact is supported by the summary, or 0 if not."
            "Return ONLY the JSON object (no extra commentary). Example: {\"1\":1, \"2\":0}"
        )
        prompt = prompt_template.format(summary=summary, facts=enumerated)

        messages = [
            {"role": "system", "content": "You are a precise assistant that evaluates whether short facts are supported by a given summary."},
            {"role": "user", "content": prompt}
        ]

        retry = 1
        while True:
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=1024
                )
                response = completion.choices[0].message.content

                # Try to extract JSON substring
                try:
                    # Find first '{' and last '}'
                    start = response.find('{')
                    end = response.rfind('}')
                    if start != -1 and end != -1 and end > start:
                        json_str = response[start:end+1]
                        parsed = json.loads(json_str)
                    else:
                        parsed = json.loads(response)
                except Exception:
                    # Fallback: try to parse simple patterns '1: 1' etc
                    parsed = {}
                    import re
                    for i, _ in enumerate(texts, start=1):
                        # look for patterns like '"1":1' or '1) 1' or '1: 1'
                        m = re.search(rf'"?{i}"?\s*[:\)\-]?\s*([01])', response)
                        if m:
                            parsed[str(i)] = int(m.group(1))

                results: Dict[str, Dict[str, Any]] = {}
                for idx, text in enumerate(texts, start=1):
                    decision = int(parsed.get(str(idx), 0))
                    results[text] = {"decision": decision, "response": response[:1000]}

                return results

            except Exception as e:
                if retry > 3:
                    # give up and fallback to substring heuristic
                    out = {}
                    for t in texts:
                        out[t] = {"decision": 1 if t.strip() and t.strip() in summary else 0, "response": None}
                    return out
                time.sleep(min(2 ** retry, 30))
                retry += 1

    async def score_summary_batched(
        self,
        article_id: str,
        summary: str,
        facts_per_batch: int = 50,
        concurrency: int = 2,
        weighted: bool = False
    ) -> Dict[str, Any]:
        """
        Faster scoring: verify all unique atomic facts for the article in batched LLM calls,
        then compute per-component-instance recall and average them.

        Args:
            article_id: article identifier
            summary: generated summary
            facts_per_batch: number of unique facts to verify per LLM call
            concurrency: number of concurrent chunk calls
            weighted: if True, compute weighted average by number of facts per component
        """
        # Retrieve components
        all_components = self.cache.get_all_components(article_id)

        if not all_components and self.decomposition_fn:
            try:
                all_components = self.decomposition_fn(article_id)
            except Exception:
                all_components = {}

        # Build component instances list and mappings
        comp_instances: List[Tuple[str, DecompositionResult]] = []  # (comp_key, instance)
        for comp_type, instances in all_components.items():
            for idx, inst in enumerate(instances):
                comp_key = f"{comp_type}__{idx}"
                comp_instances.append((comp_key, inst))

        # Flatten facts and dedupe by text
        text_to_fact_ids: Dict[str, List[str]] = {}
        factid_to_comp: Dict[str, str] = {}
        for comp_key, inst in comp_instances:
            for fact in inst.atomic_facts:
                text = fact.text.strip()
                text_to_fact_ids.setdefault(text, []).append(fact.fact_id)
                factid_to_comp[fact.fact_id] = comp_key

        unique_texts = [t for t in text_to_fact_ids.keys()]

        # Chunk unique_texts
        chunks = [unique_texts[i:i+facts_per_batch] for i in range(0, len(unique_texts), facts_per_batch)]

        # Semaphore for concurrency
        sem = asyncio.Semaphore(concurrency)

        async def call_chunk(chunk_texts: List[str]):
            async with sem:
                return await self._verify_facts_chunk(chunk_texts, summary)

        # Launch all chunk tasks
        tasks = [asyncio.create_task(call_chunk(ch)) for ch in chunks]
        responses = []
        if tasks:
            for fut in tqdm_asyncio.as_completed(tasks, desc="Batched verification", unit="chunk"):
                res = await fut
                responses.append(res)

        # Merge responses into text_decision map
        text_decision: Dict[str, int] = {}
        for r in responses:
            for txt, info in r.items():
                text_decision[txt] = int(info.get("decision", 0))

        # Compute per-component-instance recalls
        per_component_results = []
        for comp_key, inst in comp_instances:
            total = len(inst.atomic_facts)
            verified = 0
            verification_results = []
            for fact in inst.atomic_facts:
                dec = text_decision.get(fact.text.strip(), 0)
                verified += int(dec)
                verification_results.append({
                    "fact_id": fact.fact_id,
                    "fact_text": fact.text,
                    "decision": int(dec)
                })

            recall = verified / total if total > 0 else 0.0
            per_component_results.append({
                "component_type": inst.component_type,
                "component_text": inst.component_text,
                "total_facts": total,
                "verified_facts": verified,
                "recall": recall,
                "verification_results": verification_results
            })

        # Aggregate article-level score
        if per_component_results:
            if weighted:
                # weighted by number of facts per component
                weights = [r["total_facts"] for r in per_component_results]
                weighted_sum = sum(r["recall"] * w for r, w in zip(per_component_results, weights))
                avg_recall = weighted_sum / sum(weights) if sum(weights) > 0 else 0.0
            else:
                avg_recall = sum(r["recall"] for r in per_component_results) / len(per_component_results)
        else:
            avg_recall = 0.0

        return {
            "article_id": article_id,
            "summary_length": len(summary.split()),
            "num_components": len(per_component_results),
            "average_recall": avg_recall,
            "per_component_results": per_component_results
        }

    def set_decomposition(
        self,
        article_id: str,
        component_type: str,
        component_text: str,
        atomic_facts: List[Dict[str, str]]
    ):
        """
        Manually set/cache a decomposition for an article component.
        
        Args:
            article_id: Article identifier
            component_type: Type of component
            component_text: Text of the component
            atomic_facts: List of dicts with keys: fact_id, text, component_type, source
        """
        facts = [AtomicFact(**f) for f in atomic_facts]
        decomp = DecompositionResult(
            article_id=article_id,
            component_type=component_type,
            component_text=component_text,
            atomic_facts=facts,
            decomposed_at=time.time()
        )
        self.cache.put(decomp)

    def cache_stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return self.cache.stats()

    def set_decomposition_fn(self, decomposition_fn):
        """
        Set a decomposition function to be called when article_id is not in cache.
        
        The function signature should be:
            decomposition_fn(article_id: str) -> Dict[str, List[DecompositionResult]]
        
        That is, given an article_id, it returns a dict mapping component_type to list of DecompositionResult instances.
        The function is responsible for:
        - Loading or computing the decomposition from an external source
        - Caching the results (by calling self.cache.put for each decomposition)
        - Returning the dict of decompositions grouped by component_type
        
        Example:
            def my_decomposer(article_id):
                # Load decomposition from DB or file
                decomp = load_decomposition(article_id)
                # Cache it
                scorer.cache.put(decomp)
                return {decomp.component_type: [decomp]}
            
            scorer.set_decomposition_fn(my_decomposer)
        """
        self.decomposition_fn = decomposition_fn

"""
Example usage of ARCScore module with real CANLII data.

This demonstrates:
1. Loading real summaries from pickled CANLII outputs
2. Loading decompositions from the pre-built cache
3. Computing bottom-up recall per (article_id, sentence-role)
4. Averaging recall across all components in an article

NOTE: This example uses cached decompositions (pre-computed from CSV).
For actual LLM-based fact verification, you would need valid API credentials.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from arc_scorer import ARCScorer, AtomicFact, DecompositionResult
import asyncio
import pickle
import json


async def main():
    """Example: score a real CANLII summary using bottom-up ARCScore."""
    
    # Initialize scorer (mock API for dry-run - no LLM calls)
    scorer = ARCScorer(
        api_key="your_api_key_here",  # replace with valid key for real use
        base_url="https://api.openai.com/v1",  # mock local endpoint
        model_name="gpt-4o-mini", # replace with desired model (e.g., "gpt-4o", "gpt-4o-mini", etc.)
        verbose=True
    )
    
    # Show cache stats
    stats = scorer.cache_stats()
    print(f"✓ Cache loaded: {stats['total_entries']} decomposition instances")
    print(f"  Cache file: {stats['cache_file']}\n")
    
    # Load a real summary from CANLII outputs
    # article_id, summary = load_sample_summary()
    # hard code them for now to test cache
    article_id = "1995canlii6138.txt"
    summary = """ This court case, Diane Soron vs. Albert Lavoie et al, involved a plaintiff who claimed she had been injured by a car accident when she was a child. The plaintiff, now an adult, consulted a lawyer 5 years after reaching majority and sought to revive her original claim. The defendants, the lawyers who initially advised her, applied to dismiss the case due to the delay and lack of evidence.

Justice Wright considered several factors in making his decision. He had sympathy for the defendants, who had destroyed their files and had a key witness, Dr. Leakos, pass away. However, Wright also noted that the initial lawyer, Judge Lavoie, may have continued to represent the plaintiff after the initial consultation, contradicting his claim that he refused to act. Furthermore, it was unlikely that all medical records from the accident were destroyed, as hospital and patient records should still exist.

Wright was also influenced by the principle that infants (children) with legitimate claims should not be penalized for the mistakes of their previous legal representatives. He noted that the defendants had not adequately accounted for the contingent nature of the situation.

Despite the challenges the plaintiff's claim faces, such as the accident being a rear-end collision in bad weather and the plaintiff being a guest passenger, Wright dismissed the defendants' application to dismiss the case, but reserved the right to award costs to the trial judge. """
    
    if article_id and summary:
        print(f"✓ Loaded real summary from CANLII dataset")
        print(f"  Article ID: {article_id}")
        print(f"  Summary length: {len(summary.split())} words")
        print(f"  Summary preview: {summary[:200]}...\n")
        
        # Check if article is in cache
        all_components = scorer.cache.get_all_components(article_id)
        if all_components:
            print(f"✓ Found {sum(len(v) for v in all_components.values())} decomposition instances:")
            for comp_type, instances in all_components.items():
                print(f"    {comp_type}: {len(instances)} instances")
            print()
            
            # Show per-component details
            print("Components:")
            for comp_type, instances in all_components.items():
                for i, inst in enumerate(instances):
                    print(f"  [{comp_type}_{i}] {len(inst.atomic_facts)} facts")
                    print(f"       Text: {inst.component_text[:80]}...")
            print()
        else:
            print(f"⚠ Article {article_id} not found in cache\n")
    else:
        print("⚠ Could not load sample summary from pickle")
        article_id = "1995canlii6138.txt"  # fallback to a known article from cache
        summary = "The defendants applied to stay the action. Infants with bona fide causes are privileged suitors."
        print(f"  Using fallback example: article_id={article_id}\n")
    
    # Score the summary (bottom-up: per component then average)
    print(f"Computing ARCScore for article {article_id}...")
    print("(Note: This is a dry-run - no actual LLM calls are made)\n")
    
    result = await scorer.score_summary(article_id, summary)
    result = await scorer.score_summary_batched()
    print(f"ARCScore Results:")
    print(f"  Article ID: {result['article_id']}")
    print(f"  Summary length: {result['summary_length']} words")
    print(f"  Number of components: {result['num_components']}")
    print(f"  Average Recall (article-level): {result['average_recall']:.2%}")
    
    if result['per_component_results']:
        print(f"\n  Per-Component Breakdown:")
        for comp_result in result['per_component_results']:
            print(f"    [{comp_result['component_type']}]")
            print(f"      Text: {comp_result['component_text'][:60]}...")
            print(f"      Total facts: {comp_result['total_facts']}")
            print(f"      Verified facts: {comp_result['verified_facts']}")
            print(f"      Recall: {comp_result['recall']:.2%}")
    
    print("\n✓ ARCScore example completed (dry-run mode)")


if __name__ == "__main__":
    asyncio.run(main())

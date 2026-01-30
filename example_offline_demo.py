"""
Offline demonstration of ARCScore with cached decompositions (no LLM calls).

This shows:
1. Loading decompositions from the pre-built cache
2. Inspecting atomic facts per article and component
3. Simulating recall results (without actual LLM verification)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from arc_scorer import ARCScorer


def main():
    """Demonstrate ARCScore offline with cached decompositions."""
    
    # Initialize scorer (no API needed for offline demo)
    scorer = ARCScorer(
        api_key="mock-key",
        base_url="http://localhost:8000/v1",
        model_name="mock-model"
    )
    
    # Show cache stats
    stats = scorer.cache_stats()
    print("=" * 70)
    print("ARCScore Cache Statistics")
    print("=" * 70)
    print(f"Total decomposition instances: {stats['total_entries']}")
    print(f"Cache file: {stats['cache_file']}\n")
    
    # Pick an article from cache
    article_id = "1995canlii6138.txt"
    print(f"Inspecting article: {article_id}")
    print("-" * 70)
    
    # Get all components for this article
    all_components = scorer.cache.get_all_components(article_id)
    
    if not all_components:
        print(f"⚠ Article {article_id} not found in cache")
        return
    
    print(f"\n✓ Found components for article:")
    total_facts = 0
    for comp_type, instances in all_components.items():
        print(f"\n  Component Type: {comp_type}")
        print(f"  Instances: {len(instances)}")
        
        for i, inst in enumerate(instances):
            print(f"\n    [{i}] Component Text:")
            print(f"        {inst.component_text}")
            print(f"        Atomic Facts ({len(inst.atomic_facts)}):")
            
            for fact in inst.atomic_facts:
                print(f"          - {fact.text}")
                total_facts += 1
    
    print(f"\n" + "=" * 70)
    print(f"Total atomic facts for {article_id}: {total_facts}")
    print(f"Component breakdown:")
    for comp_type, instances in all_components.items():
        num_facts = sum(len(inst.atomic_facts) for inst in instances)
        print(f"  - {comp_type}: {num_facts} facts across {len(instances)} instance(s)")
    
    # Show another article
    print(f"\n" + "=" * 70)
    article_id2 = "1996canlii6881.txt"
    print(f"Inspecting article: {article_id2}")
    print("-" * 70)
    
    all_components2 = scorer.cache.get_all_components(article_id2)
    if all_components2:
        print(f"\n✓ Found components for article:")
        print(f"  Component types: {list(all_components2.keys())}")
        for comp_type, instances in all_components2.items():
            num_facts = sum(len(inst.atomic_facts) for inst in instances)
            print(f"    {comp_type}: {len(instances)} instance(s), {num_facts} facts")
    
    # Show cache retrieval methods
    print(f"\n" + "=" * 70)
    print("Cache Retrieval Methods Demo")
    print("=" * 70)
    
    # Get all Issue instances
    issues = scorer.cache.get_components(article_id, "Issue")
    print(f"\nget_components('{article_id}', 'Issue') -> {len(issues)} instances")
    for i, issue in enumerate(issues):
        print(f"  [{i}] {issue.component_text[:60]}...")
    
    # Get single component (backward compatible)
    single = scorer.cache.get(article_id, "Conclusion")
    if single:
        print(f"\nget('{article_id}', 'Conclusion') -> {single.component_type}")
        print(f"  Text: {single.component_text[:60]}...")
        print(f"  Facts: {len(single.atomic_facts)}")
    
    print(f"\n✓ ARCScore offline demo completed")


if __name__ == "__main__":
    main()

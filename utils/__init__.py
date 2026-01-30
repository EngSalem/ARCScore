"""Utility functions for ARCScore module."""

import json
from typing import List, Dict, Any
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from arc_scorer import AtomicFact


def load_atomic_facts_from_csv(csv_path: str, fact_column: str = "arg_atomic_facts_norm") -> Dict[str, List[AtomicFact]]:
    """
    Load atomic facts from a CSV file (e.g., from fact-checking dataset).
    
    Expected CSV columns:
    - name: article identifier
    - fact_column (default "arg_atomic_facts_norm"): atomic fact text
    
    Returns:
        Dict mapping article_id -> List[AtomicFact]
    """
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    facts_by_article = {}
    
    for idx, row in df.iterrows():
        article_id = str(row["name"])
        fact_text = str(row[fact_column])
        
        if article_id not in facts_by_article:
            facts_by_article[article_id] = []
        
        fact = AtomicFact(
            fact_id=f"{article_id}_fact_{idx}",
            text=fact_text,
            component_type="argument",
            source=article_id
        )
        facts_by_article[article_id].append(fact)
    
    return facts_by_article


def save_recall_results(results: List[Dict[str, Any]], output_path: str):
    """
    Save recall computation results to a JSON file.
    
    Args:
        results: List of result dicts from ARCScorer.score_summary
        output_path: Path to save JSON output
    """
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def aggregate_recall_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Aggregate recall scores across multiple summaries.
    
    Args:
        results: List of result dicts
        
    Returns:
        Dict with keys: "mean_recall", "median_recall", "total_facts", "total_verified"
    """
    if not results:
        return {}
    
    recalls = [r["recall"] for r in results]
    total_facts = sum(r["total_facts"] for r in results)
    total_verified = sum(r["verified_facts"] for r in results)
    
    return {
        "mean_recall": sum(recalls) / len(recalls),
        "median_recall": sorted(recalls)[len(recalls) // 2],
        "total_facts": total_facts,
        "total_verified": total_verified,
        "overall_recall": total_verified / total_facts if total_facts > 0 else 0.0,
        "n_summaries": len(results)
    }

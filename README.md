# 🎯 ARCScore: Argument Representation and Coverage Analysis

[![Paper](https://img.shields.io/badge/Paper-arXiv:2505.23654-b31b1b.svg)](https://arxiv.org/abs/2505.23654)
[![Conference](https://img.shields.io/badge/EACL-2025-4b44ce.svg)](https://2025.eacl.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**ARCScore** is an evaluation metric for zero-shot long document summarization with instruction-following LLMs. It measures how well a summary covers the atomic facts present in argument-annotated source documents by computing bottom-up recall across different argument components.

---

## 📖 Overview

ARCScore addresses a critical challenge in evaluating document summaries: measuring **coverage** at the atomic fact level while considering the **argumentative structure** of documents. Traditional metrics like ROUGE or BERTScore fail to capture whether key arguments, evidence, and reasoning are adequately represented in summaries.

### Key Features

- ✅ **Atomic Fact Verification**: Uses LLMs to verify if atomic facts from source documents appear in summaries
- 🔍 **Component-Level Analysis**: Evaluates coverage across different argument roles (e.g., issues, holdings, reasoning, evidence)
- 📊 **Bottom-Up Scoring**: Computes recall per component, then aggregates to document level
- 💾 **Efficient Caching**: Stores decompositions to avoid redundant LLM calls
- 🔌 **Flexible API Support**: Works with OpenAI, Azure, or any OpenAI-compatible endpoints

### How It Works

![ARCScore Bottom-Up Architecture](figs/arc_bottom_up_enlarged.pdf)

<!-- Alternative PNG format (convert PDF to PNG for better compatibility):
![ARCScore Bottom-Up Architecture](figs/arc_bottom_up_enlarged.png)
-->

ARCScore operates in two phases:

1. **Decomposition Phase** (offline): Each annotated component (e.g., an argument or piece of evidence) is decomposed into atomic facts
2. **Verification Phase** (online): For each atomic fact, an LLM verifies whether it's present/supported in the generated summary
3. **Scoring**: Recall is computed per component, then averaged across all components in the document

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ARCScore.git
cd ARCScore

# Install dependencies
pip install openai tqdm
```

**Requirements:**
- Python 3.8+
- `openai` (>=1.0.0)
- `tqdm`
- An OpenAI API key or compatible LLM endpoint

### Basic Usage

```python
import asyncio
from arc_scorer import ARCScorer

async def main():
    # Initialize the scorer
    scorer = ARCScorer(
        api_key="your_api_key_here",
        base_url="https://api.openai.com/v1",
        model_name="gpt-4o-mini",
        verbose=True
    )
    
    # Score a summary against a cached article
    article_id = "1995canlii6138.txt"
    summary = """This court case involved a plaintiff who claimed injury from a 
    childhood car accident. The plaintiff consulted a lawyer 5 years after reaching 
    majority and sought to revive her claim. Justice Wright dismissed the defendants' 
    application to dismiss the case, noting that infants with legitimate claims 
    should not be penalized for previous legal representatives' mistakes."""
    
    # Compute ARCScore
    result = await scorer.score_summary(article_id, summary)
    
    print(f"ARCScore: {result['average_recall']:.2%}")
    print(f"Components evaluated: {result['num_components']}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📚 Detailed Example

Here's a complete example showing how to work with real data and cache:

```python
"""
Example usage of ARCScore with CANLII legal case summaries.
Demonstrates loading cached decompositions and computing bottom-up recall.
"""

import asyncio
from arc_scorer import ARCScorer

async def main():
    # Initialize scorer with your LLM credentials
    scorer = ARCScorer(
        api_key="your_api_key_here",
        base_url="https://api.openai.com/v1",
        model_name="gpt-4o-mini",
        verbose=True
    )
    
    # Check cache statistics
    stats = scorer.cache_stats()
    print(f"✓ Cache loaded: {stats['total_entries']} decomposition instances")
    print(f"  Cache file: {stats['cache_file']}\n")
    
    # Example article and summary
    article_id = "1995canlii6138.txt"
    summary = """This court case, Diane Soron vs. Albert Lavoie et al, involved 
    a plaintiff who claimed she had been injured by a car accident when she was a 
    child. The plaintiff, now an adult, consulted a lawyer 5 years after reaching 
    majority and sought to revive her original claim. The defendants applied to 
    dismiss the case due to the delay and lack of evidence.

    Justice Wright considered several factors in making his decision. He had sympathy 
    for the defendants, who had destroyed their files and had a key witness pass away. 
    However, Wright noted that the initial lawyer may have continued to represent the 
    plaintiff after the initial consultation. Furthermore, medical records from the 
    accident should still exist.

    Wright was influenced by the principle that infants with legitimate claims should 
    not be penalized for mistakes of previous legal representatives. Despite the 
    challenges, Wright dismissed the defendants' application to dismiss the case."""
    
    # Check what components are available in cache
    all_components = scorer.cache.get_all_components(article_id)
    if all_components:
        print(f"✓ Found {sum(len(v) for v in all_components.values())} "
              f"decomposition instances:")
        for comp_type, instances in all_components.items():
            print(f"    {comp_type}: {len(instances)} instances")
        print()
        
        # Show detailed breakdown of components
        print("Component Details:")
        for comp_type, instances in all_components.items():
            for i, inst in enumerate(instances):
                print(f"  [{comp_type}_{i}] {len(inst.atomic_facts)} atomic facts")
                print(f"       Text: {inst.component_text[:80]}...")
        print()
    
    # Compute ARCScore (bottom-up: per-component then averaged)
    print(f"Computing ARCScore for article {article_id}...\n")
    
    result = await scorer.score_summary(article_id, summary)
    
    # Display results
    print(f"=" * 70)
    print(f"ARCScore Results:")
    print(f"=" * 70)
    print(f"  Article ID: {result['article_id']}")
    print(f"  Summary length: {result['summary_length']} words")
    print(f"  Number of components: {result['num_components']}")
    print(f"  Average Recall (article-level): {result['average_recall']:.2%}")
    print()
    
    # Per-component breakdown
    if result['per_component_results']:
        print(f"  Per-Component Breakdown:")
        print(f"  " + "-" * 66)
        for comp_result in result['per_component_results']:
            print(f"    [{comp_result['component_type']}]")
            print(f"      Text: {comp_result['component_text'][:60]}...")
            print(f"      Total facts: {comp_result['total_facts']}")
            print(f"      Verified facts: {comp_result['verified_facts']}")
            print(f"      Recall: {comp_result['recall']:.2%}")
            print()

if __name__ == "__main__":
    asyncio.run(main())
```

**Expected Output:**
```
✓ Cache loaded: 847 decomposition instances
  Cache file: /path/to/ARCScore/cache/decompositions_cache.json

✓ Found 15 decomposition instances:
    Issue: 2 instances
    Holding: 3 instances
    Reasoning: 7 instances
    Evidence: 3 instances

Computing ARCScore for article 1995canlii6138.txt...

======================================================================
ARCScore Results:
======================================================================
  Article ID: 1995canlii6138.txt
  Summary length: 187 words
  Number of components: 15
  Average Recall (article-level): 73.42%

  Per-Component Breakdown:
  ------------------------------------------------------------------
    [Issue]
      Text: Whether the plaintiff's claim should be dismissed due to...
      Total facts: 5
      Verified facts: 4
      Recall: 80.00%
    
    [Holding]
      Text: The defendants' application to dismiss is denied...
      Total facts: 3
      Verified facts: 3
      Recall: 100.00%
    ...
```

---

## 🗂️ Project Structure

```
ARCScore/
├── arc_scorer.py              # Main ARCScore implementation
├── example_usage_real_data.py # Complete usage example
├── example_offline_demo.py    # Offline demo without API calls
├── run_generate_arcscores.sh  # Batch scoring script
├── cache/                     # Cached decompositions
│   ├── decompositions_cache.json
│   └── readme.md
├── prompts/                   # LLM prompt templates
│   └── atomic_fact_binary_verifier.md
├── figs/                      # Figures and diagrams
│   └── arc_bottom_up_enlarged.pdf
└── utils/                     # Utility functions
    └── __init__.py
```

---

## 📊 CANLII Data Access

### Annotated Dataset

The ARCScore evaluation was conducted on **legal case summaries** from the Canadian Legal Information Institute (CanLII) dataset, with manual annotations of argument roles (issues, holdings, reasoning, evidence, etc.).

⚠️ **Data Acquisition Requirements:**

1. **Unannotated Data**: First, you must obtain the raw CanLII data by signing a data agreement with the [Canadian Legal Information Institute (CanLII)](https://www.canlii.org/en/)

2. **Annotated Data**: After obtaining the unannotated data, please contact **Dr. Kevin D. Ashley** ([ashley@pitt.edu](mailto:ashley@pitt.edu)) to request access to:
   - Argument role annotations for summaries
   - Argument role annotations for articles
   - Manual decompositions of arguments into atomic facts

3. **Model Outputs**: We will share our model's decompositions and outputs **after you have obtained the necessary permissions** from CanLII and Dr. Ashley.

### Cache Usage

The `cache/` directory contains pre-computed atomic fact decompositions for the annotated dataset. These decompositions allow you to:
- Evaluate summaries without re-decomposing source documents
- Reproduce results from the paper
- Compare different summarization models efficiently

To use the cache with your own data, see the [Cache Management](#-cache-management) section below.

---

## 🔧 Advanced Usage

### Custom Cache Management

```python
from arc_scorer import ARCScorer, DecompositionResult, AtomicFact

# Initialize with custom cache directory
scorer = ARCScorer(
    api_key="your_key",
    base_url="https://api.openai.com/v1",
    model_name="gpt-4o",
    cache_dir="./my_custom_cache"
)

# Add a custom decomposition to cache
custom_decomp = DecompositionResult(
    article_id="doc123",
    component_type="Argument",
    component_text="The court ruled that...",
    atomic_facts=[
        AtomicFact(
            fact_id="fact1",
            text="The court ruled in favor of the plaintiff",
            component_type="Argument",
            source="doc123"
        )
    ],
    decomposed_at=time.time()
)

scorer.cache.put(custom_decomp)
```

### Batch Processing

```python
async def score_multiple_summaries(scorer, articles_and_summaries):
    """Score multiple summaries in parallel."""
    tasks = [
        scorer.score_summary(article_id, summary)
        for article_id, summary in articles_and_summaries
    ]
    results = await asyncio.gather(*tasks)
    return results

# Usage
articles = [
    ("doc1.txt", "Summary of document 1..."),
    ("doc2.txt", "Summary of document 2..."),
    ("doc3.txt", "Summary of document 3..."),
]

results = asyncio.run(score_multiple_summaries(scorer, articles))
for result in results:
    print(f"{result['article_id']}: {result['average_recall']:.2%}")
```

### Using Alternative LLM Providers

```python
# Azure OpenAI
scorer = ARCScorer(
    api_key="your_openai_key",
    base_url="url",
    model_name="gpt-4",
    verbose=True
)

# Local LLM (e.g., via VLLM )
scorer = ARCScorer(
    api_key="not-needed",
    base_url="http://localhost:1234/v1",
    model_name="local-model",
    verbose=True
)
```

---

## 📝 Citation

If you use ARCScore in your research, please cite our paper:

```bibtex
@article{elaraby2025arc,
  title={ARC: Argument Representation and Coverage Analysis for Zero-Shot Long Document Summarization with Instruction Following LLMs},
  author={Elaraby, Mohamed and Litman, Diane},
  journal={arXiv preprint arXiv:2505.23654},
  year={2025},
  note={Accepted at EACL 2025 Main Conference}
}
```

**Paper:** [arXiv:2505.23654](https://arxiv.org/abs/2505.23654)  
**Conference:** EACL 2025 (Main Conference)

---

## 🛠️ API Reference

### `ARCScorer`

Main class for computing ARCScores.

**Constructor:**
```python
ARCScorer(
    api_key: str,              # LLM API key
    base_url: str,             # LLM API base URL
    model_name: str,           # Model name (e.g., "gpt-4o-mini")
    cache_dir: str = None,     # Cache directory (default: ./cache)
    prompts_dir: str = None,   # Prompts directory (default: ./prompts)
    temperature: float = 0,    # Sampling temperature
    is_reasoning: bool = False,# Whether model supports reasoning
    verbose: bool = False      # Print verification details
)
```

**Key Methods:**

- `async score_summary(article_id: str, summary: str) -> Dict`: Score a single summary
- `cache_stats() -> Dict`: Get cache statistics
- `cache.get_all_components(article_id: str) -> Dict`: Get all components for an article

### Return Format

The `score_summary()` method returns:

```python
{
    "article_id": str,              # Document identifier
    "summary_length": int,          # Word count
    "num_components": int,          # Number of components evaluated
    "average_recall": float,        # Article-level recall (0-1)
    "per_component_results": [      # Detailed breakdown
        {
            "component_type": str,   # e.g., "Issue", "Holding"
            "component_text": str,   # Original text
            "total_facts": int,      # Number of atomic facts
            "verified_facts": int,   # Facts found in summary
            "recall": float          # Component-level recall (0-1)
        },
        ...
    ]
}
```



## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs or issues
- Suggest new features
- Submit pull requests
- Improve documentation



## 👥 Contact

For questions about:
- **Code and technical issues**: Open a GitHub issue
- **Dataset access**: Contact Dr. Kevin D. Ashley ([ashley@pitt.edu](mailto:ashley@pitt.edu))
- **Research collaboration**: Contact Mohamed Elaraby ([ashley@pitt.edu](mailto:mse30@pitt.edu))


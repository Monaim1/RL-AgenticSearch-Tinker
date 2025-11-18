# Prior-Art Patent Search Agent

An agentic, multi-turn, tool-using system for automated prior-art search over patent corpora. This project implements a reinforcement learning-based agent that leverages the Harvard USPTO Patent Dataset to identify relevant prior patents for a given invention description.


## Project Overview

This system builds an intelligent agent that can:
- **Search** across a large patent database using semantic similarity
- **Iteratively refine** search queries over multiple turns
- **Lookup** detailed patent information
- **Rank and return** the most relevant prior patents

The agent uses:
- **ChromaDB** for semantic search with embeddings
- **DSPy** for structured LLM prompting
- **ART (Agent Reinforcement Training)** for training with RULER (LLM-as-judge) rewards
- **Qwen3 or equivalent LLMs** for the agent backbone

## Architecture

### Components

1. **Patent Database (ChromaDB)**
   - Persistent vector database with 100+ patents from Harvard USPTO dataset
   - Embeddings generated using `sentence-transformers/all-mpnet-base-v2`
   - HNSW index with cosine similarity for fast retrieval

2. **Search Tools**
   - `search_patents(query, n_results=10)` - Semantic search over patent abstracts
   - `lookup_patent(publication_number)` - Retrieve full patent details
   - `return_final_answer(answer, patent_ids)` - Format and submit final answer

3. **Agent Loop**
   - Multi-turn conversation with max 6 turns
   - System prompt guides agent to use tools iteratively
   - Termination on `FINAL_ANSWER` or max turns reached

4. **Reward Function (RULER)**
   - LLM-as-judge evaluation of agent trajectories
   - Relative ranking of multiple trajectories
   - Trains on correctness: does agent find the target patent


### Environment Variables

Create a `.env` file in the project root:

```env
WANDB_API_KEY=your_wandb_key
GEMINI_API_KEY=your_gemini_api_key
GEMINI_API_BASE=https://generativelanguage.googleapis.com/v1beta
OPENAI_API_KEY=your_openai_key  # For inference
```


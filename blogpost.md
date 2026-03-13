# Teaching a Small LLM to Search Patents: A Journey Through Multi-Step RL

Back in November 2025, I set out to build something that sounded straightforward on paper: an agent that could search a patent database and find relevant prior art. The kind of thing a patent attorney does when they need to figure out if an invention is actually novel. You give it a description of an invention, and it searches, reads, refines its queries, and comes back with the right patents.

What I didn't expect was how hard it would be to train the agent using reinforcement learning. Not the conceptual part -- that was clear. The tooling was the bottleneck.

## The Problem

Prior art search is genuinely hard. You can't just do a single keyword search and call it a day. Patent language is dense, technical, and deliberately broad. A patent about "a disposable absorbent article with a sensor device and a bi-dimensional code for monitoring vital signs" might be the answer to the query "wearable health monitoring in diapers" -- but a naive embedding search won't always surface it on the first try.

What you really need is an agent that can:

1. Take a natural language query about an invention
2. Search a patent database using semantic similarity
3. Look at the results, realize they're not quite right, and reformulate the query
4. Drill into specific patents to read their full claims and abstracts
5. Decide when it has found the right answer

This is multi-step reasoning with tool use. The agent needs to learn *when* to search again, *how* to reformulate queries, and *when* to stop.

## The Setup

### Patent Database

I used the Harvard USPTO Patent Dataset -- a collection of real patent filings. Each patent has a publication number, title, abstract, claims, IPC/CPC classification codes, and filing dates. I loaded about 700 patents into ChromaDB, a vector database, embedding the abstracts using `sentence-transformers/all-mpnet-base-v2` with HNSW indexing and cosine similarity.

The agent gets three tools:

- **`search_patents(query, n_results=10)`** -- semantic search over patent abstracts, returns titles, publication numbers, and similarity scores
- **`lookup_patent(publication_number)`** -- retrieves full patent details (abstract, claims, metadata)
- **`return_final_answer(answer, patent_ids)`** -- submits the agent's final answer

### Synthesizing Training Data

Here's the trick: I needed queries that are hard enough to require multi-step search, but where I know the ground truth answer. So I used DSPy with Gemini 2.0 Flash to generate 3 search queries per patent abstract. Each query is a natural language reformulation of what someone might type when looking for that patent as prior art.

This gave me ~2,000 training examples where each row is: `(publication_number, query, abstract)`. The query is designed to be related to the patent but not a trivial copy of the title -- forcing the agent to actually work to find it.

### The Agent Loop

The agent runs for up to 6 turns. Each turn, it decides whether to search, look up a specific patent, or return its final answer. The reward is binary: 1.0 if the gold patent ID appears in the agent's returned list, 0.0 otherwise.

Here's what a successful episode looks like:

```
Turn 1: search_patents("wearable sensor health monitoring")
        → returns 10 patents, none exactly right
Turn 2: search_patents("vital signs sensor disposable article")
        → returns better results, one looks promising
Turn 3: lookup_patent("US20180271722A1-20180927")
        → reads full abstract, confirms relevance
Turn 4: return_final_answer(answer="...", patent_ids=["US20180271722A1-20180927"])
        → reward = 1.0
```

The agent learns to reformulate queries and verify before committing to an answer.

## The RL Training Journey

This is where things got interesting -- and frustrating.

### Attempt 1: TRL

My first instinct was TRL (Transformer Reinforcement Learning from Hugging Face). It's the most popular RL library for language models, and it has examples for tool-use environments. But multi-turn, multi-step RL with tool calls turned out to be painful. TRL is built primarily for single-turn RLHF -- the kind where you generate one response and score it. Wiring up a multi-turn environment where the model calls tools, gets results, and continues reasoning required a lot of custom plumbing. The abstractions didn't quite fit.

### Attempt 2: PrimeIntellect (PRIME-RL)

PrimeIntellect's PRIME-RL framework was more promising. They ship environment abstractions designed for agentic tasks -- you define your environment, the agent interacts with it, and the framework handles the RL loop. But at the time (late 2025), the environment trainer wasn't fully mature. I hit edge cases and missing features that made it hard to get a stable training run. (It's gotten better since then -- the current version is solid.)

### Attempt 3: OpenPipe ART

I landed on OpenPipe's ART (Agent Reinforcement Training). The key selling point: serverless RL. You define your rollout function, your reward, and ART handles inference, training, and logging on their infrastructure.

Here's the core of my training loop:

```python
model = art.TrainableModel(
    project="Patent-Search",
    name="agent-001",
    base_model="OpenPipe/Qwen3-14B-Instruct"
)

for batch in training_iterator:
    # Create trajectory groups: 4 rollouts per scenario
    train_groups = []
    for scenario in batch.items:
        train_groups.append(
            art.TrajectoryGroup(
                rollout(model, scenario)
                for _ in range(4)
            )
        )

    # Run all rollouts, compute rewards
    finished = await art.gather_trajectory_groups(train_groups)

    # Train on the trajectories
    await model.train(finished, config=art.TrainConfig(lr=1e-5))
```

Each training step runs 2 scenarios with 4 rollouts each. The model generates 4 different trajectories for the same query, gets rewarded for the ones that find the correct patent, and learns from the contrast. This is essentially a form of REINFORCE with multiple samples -- the model sees what works and what doesn't for the same problem.

I also experimented with **RULER** (Relative Universal LLM-Elicited Rewards), OpenPipe's LLM-as-judge approach. Instead of just checking if the patent ID matches, RULER uses an LLM to rank multiple trajectories relative to each other. This gives richer signal than binary rewards -- the agent can learn that "found a related patent but not the exact one" is better than "returned nothing."

### What Made Multi-Step RL Hard

The fundamental challenge across all three frameworks: **multi-turn tool-use RL is a different beast than single-turn RLHF.**

In single-turn RL, you generate one response and score it. The trajectory is short, the credit assignment is straightforward, and the action space is just "generate tokens."

In multi-step agentic RL:

- **The trajectory is a conversation.** The model's action at turn 3 depends on what it did at turn 1 and what the tools returned. Credit assignment across turns is hard.
- **The action space includes tool selection.** The model isn't just generating text -- it's deciding which tool to call, with what arguments, at each step.
- **Episodes have variable length.** Some queries need 1 search, others need 4. The model has to learn when to stop.
- **Tool outputs are stochastic from the model's perspective.** The same search query always returns the same results, but the model doesn't know that in advance. It has to reason about what it learned.

Most RL frameworks in late 2025 were optimized for the single-turn case. Getting them to handle multi-turn tool-use required either significant custom code or waiting for the frameworks to mature.

## Architecture Decisions

A few things I'd highlight:

**Small model, focused task.** I used Qwen3-0.6B and Qwen3-14B at different points. The 0.6B model is surprisingly capable once trained on this specific task. You don't need GPT-4 for patent search if you train a small model with good RL signal.

**Async everything.** The rollout function is fully async, with a semaphore limiting concurrent ChromaDB queries to 20. This matters because during training, you're running many rollouts in parallel.

**Binary reward with RULER upgrade path.** Starting with a simple "did you find the right patent?" reward gets you surprisingly far. RULER adds nuance but the binary signal is enough to bootstrap.

**Synthetic queries via DSPy.** Using an LLM to generate training queries from patent abstracts is a clean way to create a large training set without manual labeling. The queries are realistic enough to train on but you always know the ground truth.

## What I Learned

1. **The RL tooling gap for agentic tasks is real.** As of late 2025, training a multi-turn tool-using agent with RL required more framework wrestling than actual research. This is changing fast -- both PrimeIntellect and OpenPipe have improved significantly.

2. **Start with the simplest reward that works.** Binary correctness rewards are ugly but effective. You can always add LLM-as-judge later.

3. **Small models can do agentic tasks.** With the right training signal, a 0.6B model can learn to search, reformulate, verify, and answer. The key is that RL teaches the *policy* -- when to use which tool -- not general knowledge.

4. **Synthetic data generation is underrated.** Using Gemini to generate realistic search queries from patent abstracts gave me 2,000+ training examples in minutes. The alternative was manual labeling, which would have taken weeks.

5. **Vector databases + RL agents are a natural fit.** The agent doesn't need to memorize patents. It just needs to learn how to search effectively. The database handles retrieval; the agent handles strategy.

## What's Next

The natural extensions are scaling the patent database (Harvard USPTO has millions of patents), adding more tools (citation graph traversal, classification-based filtering), and trying larger base models. The training infrastructure for multi-step RL is maturing quickly -- what was painful in November 2025 is getting dramatically easier.

The bigger picture: this pattern -- RL-trained agent + vector database + domain-specific tools -- generalizes well beyond patents. Legal research, academic literature review, compliance checking -- anywhere you need an agent that can iteratively search, verify, and synthesize across a large corpus.

The hard part was never the idea. It was getting the training loop to work.

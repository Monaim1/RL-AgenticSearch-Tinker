## Current Flow

prior_art_search/training_loop_qwen.py (lines 14-117) drives training by slicing the CSV into train/val, batching via art.iterate_dataset, spawning async multi-rollout batches, logging validation, then calling model.train on judged TrajectoryGroups. ART handles optimizer state, checkpointing, and the (serverless) inference endpoints.
prior_art_search/rollout.py (lines 61-172) is your “environment”: it builds the conversation history, calls AsyncOpenAI for each turn, fan-outs to search_patents / lookup_patent / return_final_answer, and turns the final answer into a scalar reward (traj.reward, traj.metrics["correct"]).
prior_art_search/prior_art_tools.py (lines 10-60) exposes async helpers over Chroma; the rollout imports them directly, so all tool execution stays in-process while the policy itself runs on OpenPipe’s hosted model.

## Migrating to TRL + GRPO

Pick & load the base policy locally. Use AutoModelForCausalLM.from_pretrained + AutoTokenizer (e.g., your Qwen checkpoint). Instantiate a frozen reference model for KL control; TRL’s GRPOConfig has reference_model/reference_tokenizer hooks.

Turn your dataset into a TRL prompt set. Recreate get_train_val_sets() as a Hugging Face Dataset with columns like query, publication_number. TRL expects a column name for prompts; store the whole system+user preamble string that rollout() currently composes so each episode starts identically.

Rebuild the rollout as a Python env. Implement something like PatentSearchEnv that mirrors rollout():

Maintain messages (system/user/assistant/tool) arrays exactly as before.
On each step, call the local HF model via generate (with stopping criteria that end on </tool_call> or similar). If you still want OpenAI-style JSON tool calls, tokenize a structured output format (e.g., <tool>name|args</tool>).
Deserialize tool calls, run the same async helpers (they’re already written), append tool responses, and continue until return_final_answer triggers or MAX_TURNS is hit.
Return the entire transcript plus the model’s logprobs so TRL can compute advantages.
Define the reward function. Port judge_correctness() into a pure Python callable that takes the env output (final answer + context) and returns scalar rewards. With GRPO you typically emit both per-example rewards and auxiliary scores (e.g., penalties for exceeding turn budgets). Keep the binary exact-match reward for parity, but this is a good point to add shaping (bonus for mentioning the gold patent in intermediate steps, penalty for invalid tool outputs).

Wire into GRPOTrainer.

Create a generator that yields batches of prompts (queries). For each, run env.run_episode(policy, tokenizer) to get input_ids, attention_mask, generated_ids, logprobs, rewards, etc.
Feed those into TRL’s GRPOTrainer via trainer.step([samples]). GRPO expects grouped rollouts per prompt (mirroring ART’s TrajectoryGroup), so set config.num_generations_per_step = training_config["rollouts_per_group"].
Replace ART’s async gather with synchronous or asyncio-driven rollout collection; the trainer simply needs lists of response_tensors and rewards.
Validation & logging. Instead of ART’s model.log(...), periodically freeze the policy, run deterministic env episodes over val_df, and compute metrics (accuracy, average turns). Use TRL’s trainer.log_stats or push to W&B manually (env already exports WANDB_API_KEY in prior_art_search/local_patent_db.py (lines 12-15)).

Serving vs. training separation. With OpenPipe the inference endpoint was managed for you. In TRL, after fine-tuning, save checkpoints (trainer.save_model(...)) and spin up your own serving stack (e.g., text-generation-inference or a simple FastAPI wrapper) that exposes the same tool-calling contract for production.

## Key Considerations

Tool-call formatting: HF models won’t natively emit OpenAI’s JSON schema. Decide on a lightweight grammar (XML tags, markdown fenced JSON) and train/reward against that format so parsing stays robust.
Batching rollouts: GRPO benefits from multiple samples per prompt; keep the idea of groups_per_step, but now it controls how many env episodes you collect before a trainer update.
Async vs. sync: TRL’s trainer loop is synchronous; you can still use asyncio to run multiple env episodes concurrently, but gather the tensors before calling trainer.step.
Scaling: Without OpenPipe’s serverless backend, you’re responsible for distributed training/inference (e.g., Accelerate configs, FSDP). Start with single-GPU GRPO to validate the pipeline, then shard as needed.
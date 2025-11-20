from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Dict, List, Optional

import torch
from transformers import PreTrainedTokenizerBase

from trl.trainer.grpo_trainer import GRPOTrainer
from trl.trainer.utils import pad

from prior_art_search.prior_art_tools import lookup_patent, search_patents


logger = logging.getLogger(__name__)


@dataclass
class FinalAnswer:
    answer: str
    patent_ids: List[str]


@dataclass
class AssistantAction:
    name: str
    arguments: Dict[str, Any]


@dataclass
class ConversationSegment:
    role: str  # "assistant" or "tool"
    content: str


@dataclass
class PatentSearchEpisode:
    prompt_text: str
    segments: List[ConversationSegment]
    final_answer: Optional[FinalAnswer]
    scenario: Dict[str, Any]


@dataclass
class PatentSearchEnvConfig:
    max_turns: int = 6
    turn_max_new_tokens: int = 196
    temperature: float = 0.9
    top_p: float = 0.95
    tool_result_limit: int = 5
    tool_response_char_limit: int = 1200


class PatentSearchEnv:
    """Drives the multi-turn tool-use interaction for a single patent search scenario."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: PatentSearchEnvConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or PatentSearchEnvConfig()
        self.system_prompt = dedent(
            """
            You are a prior-art search specialist. You can issue JSON-formatted tool calls to
            investigate a new invention description. Every assistant turn MUST be a JSON object
            with the following schema:
              {
                "action": "search_patents" | "lookup_patent" | "return_final_answer",
                "arguments": { ... }
              }

            - For search_patents, provide: {"query": string, "n_results": int (default 5)}
            - For lookup_patent, provide: {"publication_number": string}
            - For return_final_answer, provide: {"answer": string, "patent_ids": [string, ...]}

            After every tool call the environment will append a TOOL block containing the
            JSON response. Use that information before calling return_final_answer.
            Keep the reasoning tight and stay under {max_turns} tool interactions.
            """
        ).strip().format(max_turns=self.config.max_turns)

    def _initial_prompt(self, scenario: Dict[str, Any]) -> str:
        user_prompt = f"New invention description or query:\n{scenario['query'].strip()}"
        return f"{self.system_prompt}\n\nUSER:\n{user_prompt}\n\nASSISTANT:\n"

    async def _call_tool(self, action: AssistantAction) -> Any:
        if action.name == "search_patents":
            query = action.arguments.get("query")
            if not query:
                raise ValueError("search_patents requires a 'query'")
            n_results = int(action.arguments.get("n_results", 5))
            n_results = max(1, min(15, n_results))
            results = await search_patents(query=query, n_results=n_results)
            return results[: self.config.tool_result_limit]

        if action.name == "lookup_patent":
            publication_number = action.arguments.get("publication_number")
            if not publication_number:
                raise ValueError("lookup_patent requires 'publication_number'")
            return await lookup_patent(publication_number=publication_number)

        raise ValueError(f"Unknown tool: {action.name}")

    def _execute_tool(self, action: AssistantAction) -> str:
        async def runner() -> str:
            payload = await self._call_tool(action)
            serialized = json.dumps(payload, ensure_ascii=False)
            if len(serialized) > self.config.tool_response_char_limit:
                serialized = serialized[: self.config.tool_response_char_limit] + " ..."
            return f"TOOL[{action.name}]: {serialized}"

        try:
            return asyncio.run(runner())
        except RuntimeError:
            # Already inside an event loop → use a dedicated loop
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(runner())
            finally:
                new_loop.close()

    def _extract_json_block(self, text: str) -> str:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Assistant response does not contain JSON")
        return text[start : end + 1]

    def _parse_action(self, text: str) -> AssistantAction:
        try:
            payload = json.loads(self._extract_json_block(text))
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Failed to parse assistant JSON: %s", exc)
            return AssistantAction(
                name="return_final_answer",
                arguments={
                    "answer": text.strip(),
                    "patent_ids": [],
                },
            )

        action_name = payload.get("action")
        arguments = payload.get("arguments") or {}
        if action_name not in {"search_patents", "lookup_patent", "return_final_answer"}:
            logger.warning("Invalid tool name '%s' in assistant response", action_name)
            action_name = "return_final_answer"
        return AssistantAction(name=action_name, arguments=arguments)

    def run_episode(self, model: torch.nn.Module, scenario: Dict[str, Any]) -> PatentSearchEpisode:
        device = next(model.parameters()).device
        prompt_text = self._initial_prompt(scenario)
        history_text = prompt_text
        segments: List[ConversationSegment] = []
        final_answer: Optional[FinalAnswer] = None

        for turn in range(self.config.max_turns):
            inputs = self.tokenizer(
                history_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.turn_max_new_tokens,
                    do_sample=True,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=False,
                )
            generated_ids = outputs.sequences[0, inputs["input_ids"].shape[1] :]
            assistant_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if not assistant_text:
                break

            segments.append(ConversationSegment(role="assistant", content=assistant_text + "\n"))
            action = self._parse_action(assistant_text)

            if action.name == "return_final_answer":
                final_answer = FinalAnswer(
                    answer=action.arguments.get("answer", assistant_text),
                    patent_ids=[str(pid) for pid in action.arguments.get("patent_ids", [])],
                )
                history_text += assistant_text + "\n"
                break

            try:
                tool_response = self._execute_tool(action)
            except Exception as exc:  # pylint: disable=broad-except
                logger.error("Tool execution failed: %s", exc)
                break

            segments.append(ConversationSegment(role="tool", content=tool_response + "\n"))
            history_text = history_text + assistant_text + "\n" + tool_response + "\n\nASSISTANT:\n"

        return PatentSearchEpisode(
            prompt_text=prompt_text,
            segments=segments,
            final_answer=final_answer,
            scenario=scenario,
        )

    def segments_to_tokens(self, segments: List[ConversationSegment]) -> tuple[List[int], List[int], str]:
        if not segments:
            segments = [
                ConversationSegment(
                    role="assistant",
                    content='{"action": "return_final_answer", "arguments": {"answer": "", "patent_ids": []}}\n',
                )
            ]
        completion_tokens: List[int] = []
        completion_mask: List[int] = []
        completion_text_parts: List[str] = []
        for segment in segments:
            segment_text = segment.content
            completion_text_parts.append(segment_text)
            tokenized = self.tokenizer(
                segment_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.squeeze(0).tolist()
            completion_tokens.extend(tokenized)
            mask_value = 1 if segment.role == "assistant" else 0
            completion_mask.extend([mask_value] * len(tokenized))
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            completion_tokens.append(eos_id)
            completion_mask.append(1)
        return completion_tokens, completion_mask, "".join(completion_text_parts)


def parse_final_answer_from_completion(completion: str) -> Optional[FinalAnswer]:
    try:
        last_block = completion.rsplit("return_final_answer", 1)[-1]
        json_start = last_block.find("{")
        json_end = last_block.rfind("}")
        if json_start == -1 or json_end == -1 or json_end <= json_start:
            return None
        payload = json.loads(last_block[json_start : json_end + 1])
        answer_text = payload.get("answer", "").strip()
        patent_ids = [str(pid) for pid in payload.get("patent_ids", [])]
        return FinalAnswer(answer=answer_text, patent_ids=patent_ids)
    except Exception:  # pylint: disable=broad-except
        return None


def judge_correctness(gold_publication: str, final_answer: Optional[FinalAnswer]) -> float:
    if not gold_publication or final_answer is None:
        return 0.0
    return float(str(gold_publication) in (final_answer.patent_ids or []))


def patent_reward_function(
    prompts: List[str],
    completions: List[str],
    publication_number: List[str],
    **_: Any,
) -> List[float]:
    rewards: List[float] = []
    for completion, gold in zip(completions, publication_number):
        final_answer = parse_final_answer_from_completion(completion)
        rewards.append(judge_correctness(gold, final_answer))
    return rewards


class PatentSearchGRPOTrainer(GRPOTrainer):
    """GRPO trainer that swaps the default single-shot generation with the patent search env."""

    def __init__(
        self,
        env: PatentSearchEnv,
        scenario_column: str = "scenario",
        *args,
        **kwargs,
    ) -> None:
        self.patent_env = env
        self.scenario_column = scenario_column
        super().__init__(*args, **kwargs)

    def _generate_and_score_completions(self, inputs: list[dict[str, Any]]):  # type: ignore[override]
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"
        unwrap_model = self.accelerator.unwrap_model(self.model)
        if getattr(self, "use_vllm", False):
            raise RuntimeError("PatentSearchGRPOTrainer does not support vLLM-based generation for now.")

        prompt_ids_raw: List[torch.Tensor] = []
        prompt_mask_raw: List[torch.Tensor] = []
        completion_ids_raw: List[torch.Tensor] = []
        completion_mask_raw: List[torch.Tensor] = []
        completion_ids_list: List[List[int]] = []
        prompts_text: List[str] = []
        completions_text: List[str] = []

        for row in inputs:
            scenario = row.get(self.scenario_column) or {}
            if isinstance(scenario, str):
                scenario = json.loads(scenario)
            episode = self.patent_env.run_episode(unwrap_model, scenario)
            prompts_text.append(episode.prompt_text)
            prompt_ids = self.processing_class(
                episode.prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.squeeze(0)
            prompt_ids_raw.append(prompt_ids.to(device))
            prompt_mask_raw.append(torch.ones_like(prompt_ids, device=device))

            completion_tokens, completion_mask, completion_text = self.patent_env.segments_to_tokens(episode.segments)
            completion_ids_tensor = torch.tensor(completion_tokens, device=device)
            completion_mask_tensor = torch.tensor(completion_mask, device=device)
            completion_ids_raw.append(completion_ids_tensor)
            completion_mask_raw.append(completion_mask_tensor)
            completion_ids_list.append(completion_tokens)
            completions_text.append(completion_text)

        prompt_ids = pad(prompt_ids_raw, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask_raw, padding_value=0, padding_side="left")
        completion_ids = pad(completion_ids_raw, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask_raw, padding_value=0, padding_side="right")

        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids_raw], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids_raw], device=device)
        agg_prompt_lengths = self.accelerator.gather(prompt_lengths)
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        total_prompt_tokens = agg_prompt_lengths.sum()
        total_completion_tokens = agg_completion_lengths.sum()
        if mode == "train":
            self.state.num_input_tokens_seen += (total_prompt_tokens + total_completion_tokens).item()
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]
        self._metrics[mode]["completions/mean_length"].append(agg_completion_lengths.float().mean().item())
        self._metrics[mode]["completions/min_length"].append(agg_completion_lengths.float().min().item())
        self._metrics[mode]["completions/max_length"].append(agg_completion_lengths.float().max().item())

        eos_and_pad = [self.eos_token_id, self.pad_token_id]
        is_truncated = torch.tensor(
            [1 if len(ids) == 0 or ids[-1] not in eos_and_pad else 0 for ids in completion_ids_list],
            device=device,
        )
        completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()
        self._metrics[mode]["completions/clipped_ratio"].append(is_truncated.float().mean().item())

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        forward_kwargs: Dict[str, Any] = {}
        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            old_per_token_logps = None
            if self.args.gradient_accumulation_steps % (self.args.steps_per_generation * self.num_iterations) != 0:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    **forward_kwargs,
                )

            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                        **forward_kwargs,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size,
                            **forward_kwargs,
                        )
            else:
                ref_per_token_logps = None

        prompts = [row["prompt"] for row in inputs]
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        rewards_per_func = self._calculate_rewards(inputs, prompts, completions, completion_ids_list)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError("scale_rewards must be 'batch', 'group', or 'none'")

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()
        advantages = advantages[process_slice]

        for i, name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{name}/mean"].append(mean_rewards)
            std_reward = torch.nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{name}/std"].append(std_reward)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        self._logs["prompt"].extend(prompts_text)
        self._logs["completion"].extend(completions_text)
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        num_items_in_batch = sum(len(ids) for ids in completion_ids_list)
        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output

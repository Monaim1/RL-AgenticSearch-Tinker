import json
import logging
import inspect
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Dict, List, Optional

from langchain_core.utils.function_calling import convert_to_openai_tool
from openai import AsyncOpenAI
from pydantic import BaseModel

import art
from prior_art_search.prior_art_tools import search_patents, lookup_patent


MAX_TURNS = 6


# Patent data models
class Patent(BaseModel):
    publication_number: str
    title: str
    abstact: Optional[str] = None
    main_ipcr_label: Optional[str] = None
    main_cpc_label: List[str] = []
    decision: List[str] = []
    patent_issue_date: List[str] = []


@dataclass
class SearchResult:
    message_id: str
    snippet: str

class FinalAnswer(BaseModel):
    answer: str
    patent_ids: List[str]

class ProjectTrajectory(art.Trajectory):
    # Reuse the Trajectory type from `art`, only
    # adding the final_answer field specific to this project.
    final_answer: Optional[FinalAnswer] = None


@dataclass
class JudgeResponse:
    accept: bool


def return_final_answer(answer: str, patent_ids: List[str]) -> FinalAnswer:
    """Return the final answer and the patent IDs that support it."""
    return FinalAnswer(answer=answer, patent_ids=patent_ids)


async def judge_correctness(scenario_row: Dict[str, Any], final_answer: FinalAnswer) -> JudgeResponse:
    gold = str(scenario_row["publication_number"])
    accept = gold in (final_answer.patent_ids or [])
    return JudgeResponse(accept=accept)


# Multi‑turn rollout for prior‑art search
async def rollout(model: art.Model, search_scenario: Dict[str, Any]) -> ProjectTrajectory:
    scenario = search_scenario

    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "query": scenario.get("query")},
    )

    system_prompt = dedent(
        f"""
        You are a prior-art search agent. You are given a new invention description
        or search query and a set of tools you can use to perform prior-art search.

        Use the tools to search for relevant prior patents in the patent database
        and identify the most relevant patent publication numbers.

        You may take up to {MAX_TURNS} turns; if your first search does not find
        the answer, refine your queries and try again.
        You should only return publication numbers of patents that are relevant
        to the invention description or query provided.

        Tools:
        1. search_patents(query: str, n_results: int = 10) -> list[dict]:
           Search for top n_results relevant patents using embedding similarity.
        2. lookup_patent(publication_number: str) -> dict:
           Lookup patent details by publication number.
        3. return_final_answer(answer: str, patent_ids: list[str]) -> FinalAnswer:
           Return the final answer and the list of patent publication numbers
           that support your answer.
        """
    )

    user_prompt_parts = [f"New invention description or query:\n{scenario['query']}"]
    user_prompt = "\n".join(user_prompt_parts)

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tools = [search_patents, lookup_patent, return_final_answer]
    tools_by_name = {t.__name__: t for t in tools}

    # Use the same tool representation as in the `art` examples.
    traj.tools = [convert_to_openai_tool(t) for t in tools]

    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    for _ in range(MAX_TURNS):
        response = await client.chat.completions.create(
            model=model.get_inference_name(),
            temperature=1.0,
            messages=traj.messages(),
            tools=traj.tools,
        )

        response_message = response.choices[0].message
        traj.messages_and_choices.append(response.choices[0])

        # No tool calls → end episode
        if not response_message.tool_calls:
            return traj

        try:
            for tool_call in response_message.tool_calls:
                tool_name: str = tool_call.function.name
                if tool_name not in tools_by_name:
                    continue

                tool_args = json.loads(tool_call.function.arguments or "{}")
                tool_to_call = tools_by_name[tool_name]

                # Support async and sync tools
                if inspect.iscoroutinefunction(tool_to_call):
                    result = await tool_to_call(**tool_args)
                else:
                    result = tool_to_call(**tool_args)

                traj.messages_and_choices.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(result, default=str),
                    }
                )

                if tool_name == "return_final_answer":
                    traj.final_answer = result

                    if traj.final_answer:
                        correctness_judge_response = await judge_correctness(
                            scenario, traj.final_answer
                        )
                        traj.metrics["correct"] = float(
                            correctness_judge_response.accept
                        )
                    return traj

        except Exception as e:
            logging.error(f"Error executing tool call: {e}")
            return traj

    return traj

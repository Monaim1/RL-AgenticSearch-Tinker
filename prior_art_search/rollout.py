import logging

from openai import AsyncOpenAI
from litellm import acompletion
import weave
from langchain_core.utils.function_calling import convert_to_openai_tool
from tenacity import retry, stop_after_attempt

import art
from art.utils.strip_logprobs import strip_logprobs

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
    patent_ids: list[str]

@dataclass
class ProjectTrajectory:
    reward: float = 0.0
    messages: List[dict] = field(default_factory=list)
    tools: List[dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    final_answer: Optional[FinalAnswer] = None  # you already define this

    def add_message(self, role: str, content: str, **extra: Any) -> None:
        msg = {"role": role, "content": content}
        msg.update(extra)
        self.messages.append(msg)

@dataclass
class JudgeResponse:
    accept: bool


def return_final_answer(
        answer: str, 
        reference_message_ids: list[str]) -> FinalAnswer:
        """Return the final answer and the message IDs of the emails that were used to generate the answer."""
        return FinalAnswer(answer=answer, source_ids=reference_message_ids)

async def judge_correctness(scenario_row, final_answer: FinalAnswer) -> JudgeResponse:
    gold = str(scenario_row["publication_number"])
    accept = gold in (final_answer.patent_ids or [])
    return JudgeResponse(accept=accept)


# Multi‑turn rollout for prior‑art search 
async def rollout(model, SearchScenario):
    scenario = SearchScenario
    
    traj = ProjectTrajectory(
        reward=0.0,
        metadata={
            "SearchScenario": scenario.id,
            "step": email_scenario.step,
        },
    )
    system_prompt = dedent(
        f"""
        You are a prior-art search agent. You are given a new invention description
        or search query and a set of tools you can use to perform prior-art search.

        Use the tools to search for relevant prior patents in the patent database
        and identify the most relevant patent publication numbers.

        You may take up to {MAX_TURNS} turns; if your first search does not find
        the answer, refine your queries and try again.
        you should only return publication numbers of patents that are relevant
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

    # if "abstract" in scenario and isinstance(scenario["abstract"], str):
    #     user_prompt_parts.append(f"\nOptional additional context (abstract):\n{scenario['abstract']}")
    user_prompt = "\n".join(user_prompt_parts)

    traj.add_message("system", system_prompt)
    traj.add_message("user", user_prompt)

    tools = [search_patents, lookup_patent, return_final_answer]
    tools_by_name = {t.__name__: t for t in tools}

    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="whatever"  
        )
    for _ in range(MAX_TURNS):
        response = await client.chat.completions.create(
            model=model.get_inference_name(),
            temperature=1.0,
            messages=traj.messages,
            tools=traj.tools,
        )

        choice = response.choices[0]
        message = choice.message

        # Record assistant message
        traj.messages.append(message.model_dump(exclude_none=True))

        # No tool calls → end episode
        if not message.tool_calls:
            return traj

        try:
            for tool_call in message.tool_calls:
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

                traj.messages.append(
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
            print(f"Error executing tool call: {e}")
            return traj

    return traj
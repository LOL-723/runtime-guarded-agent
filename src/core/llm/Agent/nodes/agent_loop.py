import json
from typing import Any

from pydantic import ValidationError

from llm.Agent.memory import OneRunMemory
from llm.Agent.nodes.universal import _available_tools, _chat_completion, add_log
from llm.Agent.prompt import AGENT_LOOP_PROMPT
from llm.Agent.state import (
    AgentFailure,
    AgentState,
    MAX_REACT_TURNS_PER_STEP,
    MAX_REPLAN_COUNT,
    MAX_STEP_REPLAN_COUNT,
    AgentLoopResult,
    AgentLoopSignal,
    PlanStep,
    PlanStepState,
)
from llm.tools import TOOL_REGISTRY


def agent_loop_node(state: AgentState) -> AgentState:
    try:
        current_step_index, current_step = _current_step(state)
        question = state.get("question", "")
        step_id = current_step["step_id"]
        task = current_step["task"]
        memory = OneRunMemory(state)
        consecutive_think_count = 0
        turn_index = 0

        while turn_index < MAX_REACT_TURNS_PER_STEP:
            decision = _decide_next_loop(
                state=state,
                memory=memory,
                question=question,
                step_id=step_id,
                task=task,
            )
            turn_index += 1
            loop_result = AgentLoopResult(
                thought=decision["thought"],
                decide_type=decision["decide_type"],
                Signal=decision["Signal"],
                no_finding=decision["no_finding"],
                tool_name=decision["tool_name"],
                arguments=decision["arguments"],
                answer=decision["answer"],
            ).model_dump()
            _emit_agent_thought_trace(
                state=state,
                step_id=step_id,
                turn_index=turn_index,
                decision=decision,
            )

            signal = decision["Signal"]
            if signal == "overturning":
                return _handle_overturning_signal(
                    state=state,
                    step_id=step_id,
                    task=task,
                    turn_index=turn_index,
                    loop_result=loop_result,
                    memory=memory,
                )
            if signal == "overthink":
                signal_result = _handle_signal(
                    state=state,
                    step_id=step_id,
                    memory=memory,
                )
                if signal_result is not None:
                    return signal_result
                memory.reset_react_results()
                consecutive_think_count = 0
                turn_index = 0
                continue
            if signal == "tool_error":
                raise ValueError("Signal tool_error must be triggered by a real tool observation error")
            if signal is not None:
                raise ValueError(f"Signal is not implemented yet: {signal}")

            no_finding_signal = memory.update_no_finding(
                step_id=step_id,
                no_finding=decision["no_finding"],
            )
            if no_finding_signal is not None:
                return _handle_finding_missing_signal(
                    state=state,
                    step_id=step_id,
                    task=task,
                    turn_index=turn_index,
                    loop_result=loop_result,
                    memory=memory,
                )

            decide_type = decision["decide_type"]
            if decide_type == "think":
                consecutive_think_count += 1
                if consecutive_think_count >= 3:
                    signal_result = _handle_signal(
                        state=state,
                        step_id=step_id,
                        memory=memory,
                    )
                    if signal_result is not None:
                        return signal_result
                    memory.reset_react_results()
                    consecutive_think_count = 0
                    turn_index = 0
                    continue
                memory.append_loop_result(loop_result)
                continue

            if decide_type == "tool_call":
                consecutive_think_count = 0
                observation = _execute_tool_call(
                    tool_name=decision["tool_name"],
                    arguments=decision["arguments"],
                    document_id=state.get("document_id"),
                )
                loop_result["observation"] = observation
                memory.append_loop_result(loop_result)
                tool_error = _observation_error(observation)
                if tool_error is not None:
                    loop_result["Signal"] = "tool_error"
                    tool_name = _required_value(
                        decision["tool_name"],
                        "tool_name",
                        str,
                        allow_empty=False,
                    )
                    memory.record_failed_tool(tool_name)
                    if memory.agent_depth >= 1:
                        failed_state = memory.apply_to_state(state)
                        return _agent_loop_failed(
                            failed_state,
                            f"tool_error cannot create subagent at agent_depth {memory.agent_depth}",
                        )
                    subagent_update = _run_tool_error_subagent(
                        parent_state=state,
                        memory=memory,
                        question=question,
                        step_id=step_id,
                        task=task,
                    )
                    if subagent_update.get("agent_status") == "failed":
                        memory.append_subagent_result(
                            step_id=step_id,
                            task=task,
                            status="failed",
                            error=subagent_update.get("error"),
                        )
                        failed_state = memory.apply_to_state(state)
                        return _agent_loop_failed(
                            failed_state,
                            "tool_error subagent failed: "
                            f"{subagent_update.get('error') or 'unknown error'}",
                        )
                    subagent_answer = _subagent_answer(subagent_update, step_id)
                    memory.append_subagent_result(
                        step_id=step_id,
                        task=task,
                        status="done",
                        result=subagent_answer,
                    )
                    updated_plan = _update_step_result(
                        plan=list(state.get("plan", [])),
                        current_step_index=current_step_index,
                        result=subagent_answer,
                    )
                    memory.append_step_result(
                        step_id=step_id,
                        task=task,
                        result=subagent_answer,
                    )
                    tool_calls = memory.tool_calls()
                    memory.reset_react_results()
                    return {
                        "plan": updated_plan,
                        "current_react_turn_count": turn_index,
                        **memory.state_fields(),
                        "tool_calls": tool_calls,
                        "phase": "reacting",
                        "agent_status": "running",
                        "logs": add_log(
                            state=state,
                            node="agent_loop_node",
                            message="step completed by tool_error subagent",
                            extra={
                                "current_step_id": step_id,
                                "failed_tools": memory.failed_tools,
                            },
                        ),
                    }
                continue

            if decide_type == "finish":
                consecutive_think_count = 0
                answer = _required_value(
                    decision["answer"],
                    "answer",
                    str,
                    allow_empty=False,
                )
                updated_plan = _update_step_result(
                    plan=list(state.get("plan", [])),
                    current_step_index=current_step_index,
                    result=answer,
                )
                memory.append_step_result(
                    step_id=step_id,
                    task=task,
                    result=answer,
                )
                tool_calls = memory.tool_calls()
                memory.reset_react_results()
                return {
                    "plan": updated_plan,
                    "current_react_turn_count": turn_index,
                    **memory.state_fields(),
                    "tool_calls": tool_calls,
                    "phase": "reacting",
                    "agent_status": "running",
                    "logs": add_log(
                        state=state,
                        node="agent_loop_node",
                        message="step agent loop completed",
                        extra={
                            "current_step_id": step_id,
                            "agent_loop_turn_count": turn_index,
                        },
                    ),
                }

            if decide_type == "fail":
                raise ValueError("agent loop decision returned fail")

        raise ValueError(
            f"agent loop exceeded {MAX_REACT_TURNS_PER_STEP} turns before step completed"
        )
    except Exception as exc:
        return _agent_loop_failed(state, str(exc))


def _current_step(state: AgentState) -> tuple[int, PlanStepState]:
    plan = state.get("plan", [])
    current_step_index = state.get("current_step_index")
    current_step_id = state.get("current_step_id")

    if current_step_index is None:
        raise ValueError("current_step_index is required")
    if not isinstance(current_step_index, int):
        raise ValueError("current_step_index must be an integer")
    if current_step_index < 0 or current_step_index >= len(plan):
        raise ValueError("current_step_index is out of range")
    if not current_step_id:
        raise ValueError("current_step_id is required")

    current_step = plan[current_step_index]
    if current_step["step_id"] != current_step_id:
        raise ValueError("current step cursor does not match plan")

    return current_step_index, current_step


def _decide_next_loop(
    state: AgentState,
    memory: OneRunMemory,
    question: str,
    step_id: str,
    task: str,
) -> dict[str, Any]:
    payload = {
        "question": question,
        "current_step_id": step_id,
        "task": task,
        "completed_steps": memory.step_results,
        "react_results": memory.react_results,
        "previous_thought": memory.previous_thought(),
        "no_finding_count": memory.no_finding_counts.get(step_id, 0),
        "current_correction_instruction": state.get("current_correction_instruction"),
        "overthink_count": memory.overthink_counts.get(step_id, 0),
        "failed_tools": memory.failed_tools,
        "agent_depth": memory.agent_depth,
        "available_tools": _available_tools(memory.failed_tools),
    }
    data = _chat_json(
        system_prompt=AGENT_LOOP_PROMPT,
        payload=payload,
    )
    thought = _required_value(data.get("thought"), "thought", str, allow_empty=False)
    decide_type = _required_value(
        data.get("decide_type"),
        "decide_type",
        str,
        allow_empty=False,
    )
    if decide_type not in {"think", "tool_call", "finish", "fail"}:
        raise ValueError(f"unknown decide_type: {decide_type}")
    signal = _optional_signal(data.get("Signal"))
    no_finding = _optional_binary_int(data.get("no_finding", 0), "no_finding")

    tool_name = data.get("tool_name")
    if decide_type == "tool_call":
        tool_name = _required_value(
            tool_name,
            "tool_name",
            str,
            allow_empty=False,
        )
        if tool_name not in TOOL_REGISTRY or tool_name in memory.failed_tools:
            raise ValueError(f"unknown tool_name: {tool_name}")
    elif tool_name is not None:
        raise ValueError("tool_name must be null unless decide_type is tool_call")

    arguments = _required_value(
        data.get("arguments", {}),
        "arguments",
        dict,
    )
    answer = _required_value(
        data.get("answer", ""),
        "answer",
        str,
    )

    return {
        "thought": thought,
        "decide_type": decide_type,
        "Signal": signal,
        "no_finding": no_finding,
        "tool_name": tool_name,
        "arguments": arguments,
        "answer": answer,
    }


def _print_agent_thought_trace(
    step_id: str,
    turn_index: int,
    decision: dict[str, Any],
) -> None:
    signal = decision.get("Signal") or "none"
    tool_name = decision.get("tool_name") or "none"
    print(
        "[Agent Thought] "
        f"step={step_id} "
        f"turn={turn_index} "
        f"decision={decision.get('decide_type')} "
        f"signal={signal} "
        f"tool={tool_name}\n"
        f"{decision.get('thought', '')}",
        flush=True,
    )


def _emit_agent_thought_trace(
    state: AgentState,
    step_id: str,
    turn_index: int,
    decision: dict[str, Any],
) -> None:
    observer = state.get("_event_callback")
    if not callable(observer):
        return
    observer(
        {
            "type": "agent.loop.thought",
            "step_id": step_id,
            "turn": turn_index,
            "decision": decision.get("decide_type"),
            "signal": decision.get("Signal") or "none",
            "tool": decision.get("tool_name") or "none",
            "thought": decision.get("thought", ""),
        }
    )


def _optional_binary_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} type mismatch")
    if value not in {0, 1}:
        raise ValueError(f"{field} must be 0 or 1")
    return value


def _optional_signal(value: Any) -> AgentLoopSignal | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Signal type mismatch")
    if value not in {"overthink", "tool_error", "overturning", "finding_missing"}:
        raise ValueError(f"unknown Signal: {value}")
    return value


def _handle_signal(
    state: AgentState,
    step_id: str,
    memory: OneRunMemory,
) -> AgentState | None:
    next_count = memory.record_overthink(step_id)
    if next_count > 1:
        failed_state = memory.apply_to_state(state)
        return _agent_loop_failed(
            failed_state,
            "agent loop triggered overthink more than once in the same plan step",
        )
    return None


def _handle_overturning_signal(
    state: AgentState,
    step_id: str,
    task: str,
    turn_index: int,
    loop_result: dict[str, Any],
    memory: OneRunMemory,
) -> AgentState:
    replan_count = int(state.get("replan_count", 0) or 0)
    if replan_count >= MAX_REPLAN_COUNT:
        failed_state = memory.apply_to_state(state)
        failed_state["replan_count"] = replan_count
        return _agent_loop_failed(
            failed_state,
            "agent loop triggered overturning replan more than once",
        )

    trigger_trace = memory.trigger_trace(loop_result)
    last_observation = memory.last_tool_observation()
    tool_calls = memory.tool_calls()
    memory.reset_react_results()
    return {
        "planner_mode": "replan",
        "replan_count": replan_count + 1,
        "replan_context": {
            "signal": "overturning",
            "current_step_id": step_id,
            "current_step": {"step_id": step_id, "task": task},
            "react_results": trigger_trace,
            "last_tool_observation": last_observation,
        },
        "last_tool_observation": last_observation,
        "current_react_turn_count": turn_index,
        **memory.state_fields(),
        "tool_calls": tool_calls,
        "phase": "replanning",
        "agent_status": "running",
        "logs": add_log(
            state=state,
            node="agent_loop_node",
            message="overturning signal requested full replan",
            extra={
                "current_step_id": step_id,
                "replan_count": replan_count + 1,
            },
        ),
    }


def _handle_finding_missing_signal(
    state: AgentState,
    step_id: str,
    task: str,
    turn_index: int,
    loop_result: dict[str, Any],
    memory: OneRunMemory,
) -> AgentState:
    step_replan_count = int(state.get("step_replan_count", 0) or 0)
    if step_replan_count >= MAX_STEP_REPLAN_COUNT:
        failed_state = memory.apply_to_state(state)
        failed_state["step_replan_count"] = step_replan_count
        return _agent_loop_failed(
            failed_state,
            "agent loop triggered finding_missing step replan more than once",
        )

    trigger_trace = memory.trigger_trace(loop_result)
    memory.react_results = trigger_trace
    return {
        "planner_mode": "step_replan",
        "step_replan_count": step_replan_count + 1,
        "replan_context": {
            "signal": "finding_missing",
            "current_step_id": step_id,
            "current_step": {"step_id": step_id, "task": task},
            "react_results": trigger_trace,
            "no_finding_count": memory.no_finding_counts.get(step_id, 0),
        },
        "current_react_turn_count": turn_index,
        **memory.state_fields(),
        "phase": "replanning",
        "agent_status": "running",
        "logs": add_log(
            state=state,
            node="agent_loop_node",
            message="finding_missing signal requested step replan",
            extra={
                "current_step_id": step_id,
                "step_replan_count": step_replan_count + 1,
                "no_finding_count": memory.no_finding_counts.get(step_id, 0),
            },
        ),
    }


def _observation_error(observation: str) -> str | None:
    try:
        data = json.loads(observation)
    except json.JSONDecodeError as exc:
        raise ValueError("tool observation returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("tool observation returned non-object JSON")
    error = data.get("error")
    if error is None:
        return None
    if not isinstance(error, str):
        raise ValueError("tool observation error type mismatch")
    if not error.strip():
        return None
    return error


def _run_tool_error_subagent(
    parent_state: AgentState,
    memory: OneRunMemory,
    question: str,
    step_id: str,
    task: str,
) -> AgentState:
    subagent_step = PlanStep(step_id=step_id, task=task).model_dump()
    subagent_state = memory.subagent_state(
        parent_state=parent_state,
        question=question,
        step_id=step_id,
        task=task,
        subagent_step=subagent_step,
    )
    return agent_loop_node(subagent_state)


def _subagent_answer(subagent_update: AgentState, step_id: str) -> str:
    for step_result in reversed(subagent_update.get("step_results", [])):
        if step_result.get("step_id") != step_id:
            continue
        result = step_result.get("result")
        if isinstance(result, str) and result.strip():
            return result
    raise ValueError("tool_error subagent returned no usable result")


def _execute_tool_call(
    tool_name: str | None,
    arguments: dict[str, Any],
    document_id: str | None,
) -> str:
    if tool_name is None:
        raise ValueError("tool_name is required for tool_call")

    tool_arguments = dict(arguments)
    if tool_name == "retrieve_uploaded_document" and "document_id" not in tool_arguments:
        tool_arguments["document_id"] = document_id

    try:
        result = TOOL_REGISTRY[tool_name](**tool_arguments)
    except Exception as exc:
        return json.dumps(
            {
                "tool_name": tool_name,
                "result": None,
                "error": str(exc),
            },
            ensure_ascii=False,
            default=str,
        )

    return json.dumps(
        {
            "tool_name": tool_name,
            "result": result,
            "error": None,
        },
        ensure_ascii=False,
        default=str,
    )


def _chat_json(system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    content = _chat_completion(
        system_prompt=system_prompt,
        user_message=json.dumps(payload, ensure_ascii=False),
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("agent loop returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("agent loop returned non-object JSON")
    return data


def _update_step_result(
    plan: list[PlanStepState],
    current_step_index: int,
    result: str,
) -> list[PlanStepState]:
    updated_plan = list(plan)
    step = updated_plan[current_step_index]
    try:
        updated_plan[current_step_index] = PlanStep(
            step_id=step["step_id"],
            task=step["task"],
            status="done",
            result=result,
            retry_count=step.get("retry_count", 0),
        ).model_dump()
    except (TypeError, ValidationError) as exc:
        raise ValueError("current plan step is invalid") from exc

    return updated_plan


def _required_value(
    value: Any,
    field: str,
    expected_type: type[Any],
    *,
    allow_empty: bool = True,
) -> Any:
    if not isinstance(value, expected_type):
        raise ValueError(f"{field} type mismatch")
    if isinstance(value, str) and not allow_empty and not value.strip():
        raise ValueError(f"{field} cannot be empty")
    return value


def _agent_loop_failed(state: AgentState, error: str) -> AgentState:
    message = f"agent loop failed: {error}"
    failure = AgentFailure(
        reason="react_failed",
        message=message,
        node="agent_loop_node",
        target_step_id=state.get("current_step_id"),
    )
    failed_update: AgentState = {
        "phase": "failed",
        "agent_status": "failed",
        "error": message,
        "failure": failure.model_dump(),
        "logs": add_log(
            state=state,
            node="agent_loop_node",
            message="agent loop failed",
            extra={"error": error},
        ),
    }
    if "failed_tools" in state:
        failed_update["failed_tools"] = state["failed_tools"]
    if "subagent_results" in state:
        failed_update["subagent_results"] = state["subagent_results"]
    if "overthink_counts" in state:
        failed_update["overthink_counts"] = state["overthink_counts"]
    if "no_finding_counts" in state:
        failed_update["no_finding_counts"] = state["no_finding_counts"]
    if "replan_count" in state:
        failed_update["replan_count"] = state["replan_count"]
    if "step_replan_count" in state:
        failed_update["step_replan_count"] = state["step_replan_count"]
    return failed_update

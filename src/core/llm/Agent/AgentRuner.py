import json
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, TextIO

from llm import langgraph
from llm.Agent.memory import ContextMemory, OneRunMemory
from llm.Agent.state import AgentState


RUNS_DIR = Path("runs")
Event = dict[str, Any]
EventHandler = Callable[[Event], None]


@dataclass(frozen=True)
class AgentRunResult:
    run_id: str
    status: str
    answer: str
    events_path: Path


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        self._subscribers.append(handler)

    def publish(self, event: Event) -> None:
        for handler in self._subscribers:
            handler(event)


class EventWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file: TextIO | None = None

    def __enter__(self) -> "EventWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *args: object) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def subscribe(self, bus: EventBus) -> None:
        bus.subscribe(self.handle)

    def handle(self, event: Event) -> None:
        if self._file is None:
            return
        self._file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._file.flush()


class StdoutPrinter:
    def __init__(self) -> None:
        self._started_at = 0.0

    def handle(self, event: Event) -> None:
        event_type = event.get("type")
        if event_type == "run.started":
            self._started_at = time.monotonic()
            self._print(f"[run] {event.get('run_id')} {event.get('goal')}")
        elif event_type == "agent.node.started":
            self._print(f"[node] {event.get('node')} started")
        elif event_type == "agent.node.finished":
            status = event.get("status") or "unknown"
            self._print(f"[node] {event.get('node')} {status}")
        elif event_type == "agent.step.started":
            self._print(f"[step] {event.get('step_id')} {event.get('task')}")
        elif event_type == "agent.loop.thought":
            self._print(
                "[Agent Thought] "
                f"step={event.get('step_id')} "
                f"turn={event.get('turn')} "
                f"decision={event.get('decision')} "
                f"signal={event.get('signal')} "
                f"tool={event.get('tool')}\n"
                f"{event.get('thought', '')}",
            )
        elif event_type == "agent.log":
            self._print(f"[log] {event.get('source')}: {event.get('message')}")
        elif event_type == "agent.answer":
            self._print(str(event.get("answer", "")))
        elif event_type == "run.finished":
            elapsed = time.monotonic() - self._started_at if self._started_at else 0.0
            self._print(f"[run] {event.get('status')} {elapsed:.1f}s")

    def _print(self, text: str) -> None:
        try:
            print(text, flush=True)
        except UnicodeEncodeError:
            encoding = sys.stdout.encoding or "utf-8"
            safe_text = text.encode(encoding, errors="replace").decode(encoding)
            print(safe_text, flush=True)


class AgentRuner:
    def __init__(
        self,
        *,
        runs_dir: str | Path | None = None,
        extra_handlers: list[EventHandler] | None = None,
        context_memory_path: str | Path | None = None,
    ) -> None:
        self.runs_dir = Path(runs_dir) if runs_dir is not None else RUNS_DIR
        self.extra_handlers = list(extra_handlers or [])
        self.context_memory_path = (
            Path(context_memory_path) if context_memory_path is not None else None
        )

    def run(self, goal: str) -> AgentRunResult:
        if not goal or not goal.strip():
            raise ValueError("goal cannot be empty")

        run_id = new_run_id()
        run_path = self.runs_dir / run_id
        events_path = run_path / "events.jsonl"
        bus = EventBus()
        for handler in self.extra_handlers:
            bus.subscribe(handler)

        status = "failed"
        answer = ""
        error: str | None = None

        with EventWriter(events_path) as writer:
            writer.subscribe(bus)
            bus.publish(_event("run.started", run_id, goal=goal))
            try:
                answer = self._run_agent(goal=goal, run_id=run_id, bus=bus)
                bus.publish(_event("agent.answer", run_id, answer=answer))
                status = "finished"
            except Exception as exc:
                error = str(exc)
                bus.publish(_event("agent.error", run_id, error=error))
            finally:
                bus.publish(
                    _event(
                        "run.finished",
                        run_id,
                        status=status,
                        answer=answer,
                        error=error,
                    )
                )

        if status != "finished":
            raise RuntimeError(error or "agent run failed")
        return AgentRunResult(
            run_id=run_id,
            status=status,
            answer=answer,
            events_path=events_path,
        )

    def _run_agent(self, *, goal: str, run_id: str, bus: EventBus) -> str:
        agent_state = OneRunMemory.initial_state(question=goal, document_id=None)
        agent_state["_event_callback"] = lambda event: bus.publish(
            _event(
                str(event.get("type", "agent.event")),
                run_id,
                **{
                    key: value
                    for key, value in event.items()
                    if key not in {"type", "run_id", "ts"}
                },
            )
        )
        agent_state["context_memory"] = ContextMemory(self.context_memory_path).load()

        for _ in range(langgraph.MAX_AGENT_NODE_ITERATIONS):
            should_plan = (
                agent_state.get("planner_mode") in {"replan", "step_replan"}
                or "plan" not in agent_state
            )
            if should_plan:
                agent_state = self._run_node(
                    run_id=run_id,
                    bus=bus,
                    state=agent_state,
                    node_name="planner_node",
                    node=langgraph.planner_node,
                )
                self._raise_if_failed(agent_state)
                continue

            agent_state = self._run_node(
                run_id=run_id,
                bus=bus,
                state=agent_state,
                node_name="select_next_step_node",
                node=langgraph.select_next_step_node,
            )
            self._raise_if_failed(agent_state)
            if agent_state.get("should_continue_next") == "finish":
                answer = langgraph._summarize_agent_answer(agent_state)
                ContextMemory(self.context_memory_path).remember(
                    question=goal,
                    final_answer=answer,
                )
                return answer

            current_step = _current_step(agent_state)
            if current_step is not None:
                bus.publish(
                    _event(
                        "agent.step.started",
                        run_id,
                        step_id=current_step.get("step_id"),
                        task=current_step.get("task"),
                    )
                )

            agent_state = self._run_node(
                run_id=run_id,
                bus=bus,
                state=agent_state,
                node_name="agent_loop_node",
                node=langgraph.agent_loop_node,
            )
            self._raise_if_failed(agent_state)

            finished_step = _current_step(agent_state)
            if finished_step is not None and finished_step.get("status") == "done":
                bus.publish(
                    _event(
                        "agent.step.finished",
                        run_id,
                        step_id=finished_step.get("step_id"),
                        result=finished_step.get("result"),
                    )
                )

        raise RuntimeError("agent exceeded graph iteration limit")

    def _run_node(
        self,
        *,
        run_id: str,
        bus: EventBus,
        state: AgentState,
        node_name: str,
        node: Callable[[AgentState], AgentState],
    ) -> AgentState:
        previous_log_count = len(state.get("logs", []))
        bus.publish(_event("agent.node.started", run_id, node=node_name))
        update = node(state)
        merged = langgraph._merge_agent_state(state, update)
        for log_item in merged.get("logs", [])[previous_log_count:]:
            bus.publish(
                _event(
                    "agent.log",
                    run_id,
                    source=log_item.get("node", node_name),
                    message=log_item.get("message", ""),
                    extra={
                        key: value
                        for key, value in log_item.items()
                        if key not in {"node", "message"}
                    },
                )
            )
        bus.publish(
            _event(
                "agent.node.finished",
                run_id,
                node=node_name,
                status=merged.get("agent_status", "running"),
                phase=merged.get("phase"),
            )
        )
        return merged

    def _raise_if_failed(self, agent_state: AgentState) -> None:
        if agent_state.get("agent_status") == "failed":
            raise RuntimeError(str(agent_state.get("error") or "agent failed"))


def new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{timestamp}-{suffix}"


def _event(event_type: str, run_id: str, **payload: Any) -> Event:
    return {
        "type": event_type,
        "run_id": run_id,
        "ts": datetime.now(UTC).isoformat(),
        **payload,
    }


def _current_step(agent_state: AgentState) -> dict[str, Any] | None:
    current_step_id = agent_state.get("current_step_id")
    if not current_step_id:
        return None
    for step in agent_state.get("plan", []):
        if step.get("step_id") == current_step_id:
            return step
    return None

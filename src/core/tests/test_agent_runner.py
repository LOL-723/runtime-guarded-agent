import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from llm.Agent.AgentRuner import AgentRuner, StdoutPrinter
from llm.Agent.nodes import agent_loop


class AgentRunerTests(unittest.TestCase):
    def test_runner_writes_events_jsonl_for_successful_agent_run(self) -> None:
        def fake_planner(state):
            return {
                "plan": [
                    {
                        "step_id": "step_1",
                        "task": "answer the goal",
                        "status": "pending",
                        "result": None,
                        "retry_count": 0,
                    }
                ],
                "plan_revision": 1,
                "plan_updates": [],
                "planner_mode": "initial",
                "agent_status": "running",
                "logs": [{"node": "planner_node", "message": "plan created"}],
            }

        def fake_select(state):
            if state.get("step_results"):
                return {
                    "should_continue_next": "finish",
                    "current_step_index": 1,
                    "current_step_id": None,
                    "agent_status": "running",
                    "logs": state.get("logs", [])
                    + [{"node": "select_next_step_node", "message": "done"}],
                }
            return {
                "should_continue_next": "continue",
                "current_step_index": 0,
                "current_step_id": "step_1",
                "agent_status": "running",
                "logs": state.get("logs", [])
                + [{"node": "select_next_step_node", "message": "selected"}],
            }

        def fake_loop(state):
            state["_event_callback"](
                {
                    "type": "agent.loop.thought",
                    "step_id": "step_1",
                    "turn": 1,
                    "decision": "finish",
                    "signal": "none",
                    "tool": "none",
                    "thought": "ready to answer",
                }
            )
            plan = list(state["plan"])
            plan[0] = dict(plan[0], status="done", result="step answer")
            return {
                "plan": plan,
                "step_results": [
                    {
                        "step_id": "step_1",
                        "task": "answer the goal",
                        "result": "step answer",
                    }
                ],
                "agent_status": "running",
                "logs": state.get("logs", [])
                + [{"node": "agent_loop_node", "message": "step completed"}],
            }

        seen_events = []
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = AgentRuner(
                runs_dir=Path(temp_dir) / "runs",
                context_memory_path=Path(temp_dir) / "context_memory.jsonl",
                extra_handlers=[seen_events.append],
            )
            with (
                patch("llm.langgraph.planner_node", side_effect=fake_planner),
                patch("llm.langgraph.select_next_step_node", side_effect=fake_select),
                patch("llm.langgraph.agent_loop_node", side_effect=fake_loop),
                patch("llm.langgraph._summarize_agent_answer", return_value="final answer"),
            ):
                result = runner.run("complex task")

            self.assertEqual(result.status, "finished")
            self.assertEqual(result.answer, "final answer")
            self.assertTrue(result.events_path.exists())
            file_events = [
                json.loads(line)
                for line in result.events_path.read_text(encoding="utf-8").splitlines()
            ]

        event_types = [event["type"] for event in file_events]
        self.assertEqual(event_types[0], "run.started")
        self.assertIn("agent.step.started", event_types)
        self.assertIn("agent.loop.thought", event_types)
        self.assertIn("agent.step.finished", event_types)
        self.assertIn("agent.answer", event_types)
        self.assertEqual(event_types[-1], "run.finished")
        self.assertEqual([event["type"] for event in seen_events], event_types)

    def test_agent_loop_emits_thought_callback_without_legacy_printer(self) -> None:
        emitted_events = []
        state = {
            "question": "question",
            "plan": [
                {
                    "step_id": "step_1",
                    "task": "answer",
                    "status": "pending",
                    "result": None,
                    "retry_count": 0,
                }
            ],
            "current_step_index": 0,
            "current_step_id": "step_1",
            "logs": [],
            "_event_callback": emitted_events.append,
        }
        decision = {
            "thought": "ready to answer",
            "decide_type": "finish",
            "Signal": None,
            "no_finding": 0,
            "tool_name": None,
            "arguments": {},
            "answer": "final",
        }

        with (
            patch.object(agent_loop, "_decide_next_loop", return_value=decision),
            patch.object(agent_loop, "_print_agent_thought_trace") as legacy_printer,
        ):
            update = agent_loop.agent_loop_node(state)

        legacy_printer.assert_not_called()
        self.assertEqual(
            emitted_events,
            [
                {
                    "type": "agent.loop.thought",
                    "step_id": "step_1",
                    "turn": 1,
                    "decision": "finish",
                    "signal": "none",
                    "tool": "none",
                    "thought": "ready to answer",
                }
            ],
        )
        self.assertEqual(update["plan"][0]["status"], "done")
        self.assertEqual(update["plan"][0]["result"], "final")

    def test_stdout_printer_renders_thought_event_like_legacy_trace(self) -> None:
        printer = StdoutPrinter()
        stream = StringIO()

        with redirect_stdout(stream):
            printer.handle(
                {
                    "type": "agent.loop.thought",
                    "step_id": "step_1",
                    "turn": 2,
                    "decision": "tool_call",
                    "signal": "none",
                    "tool": "retrieve_uploaded_document",
                    "thought": "I need to inspect the document.",
                }
            )

        self.assertEqual(
            stream.getvalue(),
            "[Agent Thought] "
            "step=step_1 "
            "turn=2 "
            "decision=tool_call "
            "signal=none "
            "tool=retrieve_uploaded_document\n"
            "I need to inspect the document.\n",
        )

    def test_stdout_printer_replaces_unencodable_characters(self) -> None:
        printer = StdoutPrinter()
        ascii_stdout = type("AsciiStdout", (), {"encoding": "ascii"})()

        with (
            patch("sys.stdout", ascii_stdout),
            patch(
                "builtins.print",
                side_effect=[
                    UnicodeEncodeError("ascii", "hello 😊", 6, 7, "unsupported"),
                    None,
                ],
            ) as mocked_print,
        ):
            printer._print("hello 😊")

        mocked_print.assert_any_call("hello ?", flush=True)


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from unittest.mock import patch

from llm.Agent.memory import append_plan_visualization
from llm.Agent.nodes import agent_loop_node, planner_node
from llm.Agent.state import (
    AgentState,
    MAX_PLAN_STEPS,
    PlanStep,
)


class AgentStateTest(unittest.TestCase):
    def _run_agent_loop_with_responses(self, responses: list[str]) -> AgentState:
        plan = [PlanStep(step_id="step_1", task="Validate model output").model_dump()]

        with patch("llm.Agent.nodes.agent_loop._chat_completion", side_effect=responses):
            return agent_loop_node(
                {
                    "question": "Validate model output",
                    "plan": plan,
                    "current_step_index": 0,
                    "current_step_id": "step_1",
                    "logs": [],
                }
            )

    @staticmethod
    def _loop_response(
        *,
        thought: str = "Inspect current evidence",
        decide_type: str = "finish",
        signal: str | None = None,
        no_finding: int = 0,
        answer: str = "done",
    ) -> str:
        return json.dumps(
            {
                "thought": thought,
                "decide_type": decide_type,
                "Signal": signal,
                "no_finding": no_finding,
                "tool_name": None,
                "arguments": {},
                "answer": answer,
            }
        )

    def test_overturning_signal_requests_single_full_replan(self) -> None:
        observation = json.dumps(
            {
                "tool_name": "search",
                "result": "the earlier assumption is false",
                "error": None,
            }
        )
        state: AgentState = {
            "question": "Investigate failure",
            "plan": [PlanStep(step_id="step_1", task="Check likely cause").model_dump()],
            "current_step_index": 0,
            "current_step_id": "step_1",
            "react_results": [
                {
                    "thought": "Assume the cause is configuration",
                    "decide_type": "tool_call",
                    "Signal": None,
                    "no_finding": 0,
                    "tool_name": "search",
                    "arguments": {},
                    "observation": observation,
                    "answer": "",
                }
            ],
            "logs": [],
        }

        with patch(
            "llm.Agent.nodes.agent_loop._chat_completion",
            return_value=self._loop_response(
                thought="The tool result overturns the current plan",
                decide_type="think",
                signal="overturning",
                answer="",
            ),
        ):
            result = agent_loop_node(state)

        self.assertEqual(result["agent_status"], "running")
        self.assertEqual(result["planner_mode"], "replan")
        self.assertEqual(result["replan_count"], 1)
        self.assertEqual(result["last_tool_observation"], observation)
        self.assertEqual(result["react_results"], [])

    def test_overturning_signal_fails_after_single_replan_used(self) -> None:
        state: AgentState = {
            "question": "Investigate failure",
            "plan": [PlanStep(step_id="step_1", task="Check likely cause").model_dump()],
            "current_step_index": 0,
            "current_step_id": "step_1",
            "replan_count": 1,
            "logs": [],
        }

        with patch(
            "llm.Agent.nodes.agent_loop._chat_completion",
            return_value=self._loop_response(
                decide_type="think",
                signal="overturning",
                answer="",
            ),
        ):
            result = agent_loop_node(state)

        self.assertEqual(result["agent_status"], "failed")
        self.assertIn("overturning replan more than once", result["error"])






if __name__ == "__main__":
    unittest.main()

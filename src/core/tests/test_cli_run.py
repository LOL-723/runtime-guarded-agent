import unittest
from unittest.mock import patch

from cli import commands


class CliRunCommandTests(unittest.TestCase):
    def test_run_command_dispatches_to_single_process_runner(self) -> None:
        with (
            patch("core.llm.Agent.AgentRuner.StdoutPrinter") as printer_class,
            patch("core.llm.Agent.AgentRuner.AgentRuner") as runner_class,
        ):
            printer = printer_class.return_value
            runner = runner_class.return_value

            exit_code = commands.run_command(["hello", "agent"])

        self.assertEqual(exit_code, 0)
        runner_class.assert_called_once_with(extra_handlers=[printer.handle])
        runner.run.assert_called_once_with("hello agent")


if __name__ == "__main__":
    unittest.main()

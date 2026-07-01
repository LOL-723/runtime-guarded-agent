import argparse
from typing import Callable

from cli.formatting import format_ping_reply
from core.ipc.client import CoreClient, get_default_host, get_default_port
from core.ipc.daemon_process import ensure_daemon_running


CommandHandler = Callable[[list[str]], int]


def ping_command(argv: list[str]) -> int:
    host = get_default_host()
    port = get_default_port()
    ensure_daemon_running(host=host, port=port)
    rpc_result = CoreClient(host=host, port=port).request(
        "core.ping",
        {"client": "sorrow-cli/0.1.0"},
    )
    print(format_ping_reply(host=host, port=port, rpc_result=rpc_result), flush=True)
    return 0


def shutdown_command(argv: list[str]) -> int:
    host = get_default_host()
    port = get_default_port()
    try:
        rpc_result = CoreClient(host=host, port=port).request(
            "core.shutdown",
            {"client": "sorrow-cli/0.1.0"},
        )
    except OSError:
        print(f"Daemon is not running at {host}:{port}.", flush=True)
        return 0
    print(
        "Daemon shutting down: "
        f"{host}:{port} uptime={_format_seconds(rpc_result.result.get('uptime_ms', 0))}",
        flush=True,
    )
    return 0


def run_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="sorrow run")
    parser.add_argument("goal", nargs="+")
    args = parser.parse_args(argv)
    goal = " ".join(args.goal).strip()

    from core.llm.Agent.AgentRuner import AgentRuner, StdoutPrinter

    printer = StdoutPrinter()
    runner = AgentRuner(extra_handlers=[printer.handle])
    try:
        runner.run(goal)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Agent run failed: {exc}", flush=True)
        return 1
    return 0


def _format_seconds(uptime_ms: object) -> str:
    try:
        return f"{float(uptime_ms) / 1000:.1f}s"
    except (TypeError, ValueError):
        return "unknown"


COMMANDS: dict[str, CommandHandler] = {
    "ping": ping_command,
    "run": run_command,
    "shutdown": shutdown_command,
}

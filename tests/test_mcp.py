"""
Tests of ``biapy.run_subcommand`` and of the MCP server (``biapy mcp``, in ``biapy/agent/mcp_server.py``): its tools
give agents the same report as the ``biapy`` subcommands with ``--json``.

How to run
----------
``run_subcommand`` needs nothing more than BiaPy; the server tests need the MCP SDK (``pip install "biapy[mcp]"``)
and are skipped without it::

    pytest tests/test_mcp.py -q
"""
import json

import pytest

from biapy import run_subcommand
from biapy._biapy import VALID_WORKFLOWS

REPORT_KEYS = ["schema_version", "ok", "errors", "warnings", "data"]


######################
# run_subcommand     #
######################
def test_run_subcommand(capsys):
    """The report of a subcommand, without printing it."""
    report = run_subcommand("list-workflows")
    assert list(report) == REPORT_KEYS
    assert report["ok"]
    assert report["data"]["workflows"] == VALID_WORKFLOWS
    assert capsys.readouterr().out == ""


def test_run_subcommand_failure():
    report = run_subcommand("get-config-template", ["--workflow", "NOPE", "--ndim", "2D"])
    assert not report["ok"]
    assert "workflow" in report["errors"][0]


def test_run_subcommand_wrong_arguments():
    """Wrong arguments give a report with argparse's message, instead of exiting."""
    report = run_subcommand("get-config-template", ["--ndim", "2D"])
    assert list(report) == REPORT_KEYS
    assert not report["ok"]
    assert "--workflow" in report["errors"][0]


def test_run_subcommand_unknown():
    with pytest.raises(ValueError):
        run_subcommand("no-such-command")


##############
# MCP server #
##############
def _server():
    pytest.importorskip("mcp")
    from biapy.agent.mcp_server import server

    return server


def _call(server, tool, **arguments):
    """Calls a tool as a client does and returns its report."""
    import anyio

    result = anyio.run(server.call_tool, tool, arguments)
    return result.structured_content or json.loads(result.content[0].text)


def test_tools():
    """One tool for each subcommand, with its arguments."""
    import anyio

    tools = {t.name: t for t in anyio.run(_server().list_tools)}
    assert set(tools) == {
        "list_workflows", "inspect_image", "get_config_template", "create_config", "validate_config", "check_data",
        "search_bmz_models", "plan", "run", "job_status", "stop_job", "inspect_results",
    }
    assert set(tools["get_config_template"].input_schema["required"]) == {"workflow", "ndim"}
    assert tools["inspect_results"].annotations.read_only_hint
    assert tools["stop_job"].annotations.destructive_hint


def test_tool_same_report_as_subcommand():
    server = _server()
    assert _call(server, "list_workflows") == run_subcommand("list-workflows")
    report = _call(server, "get_config_template", workflow="SEMANTIC_SEG", ndim="2D")
    assert report == run_subcommand("get-config-template", ["--workflow", "SEMANTIC_SEG", "--ndim", "2D"])


def test_tool_failure(tmp_path):
    """A failure is a report with ok false and why, not an error of the tool."""
    report = _call(_server(), "inspect_image", path=str(tmp_path / "missing.tif"))
    assert list(report) == REPORT_KEYS
    assert not report["ok"]
    assert report["errors"]


def test_setup_print(capsys):
    """The configuration for any agent: this environment's 'biapy' with its full path, and 'mcp'."""
    _server()
    from biapy.agent.mcp_server import serve, server_command

    command = server_command()
    assert command[-1] == "mcp"
    serve(["--setup", "print"])
    out = capsys.readouterr().out
    assert json.dumps(command[0]) in out
    assert "[mcp_servers.biapy]" in out


def test_setup_without_agent_commands(monkeypatch, tmp_path):
    """Without the 'claude' and 'codex' commands, their configuration files are written."""
    _server()
    from biapy.agent import mcp_server

    def which(name, path=None):  # no 'claude' nor 'codex' command; this environment's 'biapy'
        return None if name in ("claude", "codex") else "/env/bin/biapy"

    monkeypatch.setattr(mcp_server.shutil, "which", which)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".mcp.json").write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
    mcp_server.setup("claude", str(tmp_path))
    config = json.loads((tmp_path / ".mcp.json").read_text())
    assert config["mcpServers"]["biapy"] == {"command": "/env/bin/biapy", "args": ["mcp"]}
    assert "other" in config["mcpServers"]  # what was there is kept

    mcp_server.setup("codex")
    toml = (tmp_path / ".codex" / "config.toml").read_text()
    assert '[mcp_servers.biapy]\ncommand = "/env/bin/biapy"\nargs = ["mcp"]' in toml
    assert "already has" in mcp_server.setup("codex")  # not added twice


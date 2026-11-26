"""
MCP server of BiaPy: the ``biapy`` subcommands as tools for AI agents (Claude Code, Claude Desktop, Codex, Cursor...).

Each tool runs a subcommand with :func:`biapy.run_subcommand` and returns its report, the same JSON the command prints
with ``--json`` (``schema_version``, ``ok``, ``errors``, ``warnings`` and ``data``): the logic is only in the
subcommands, the server only gives them to agents.

Install it and add it to the agent::

    pip install "biapy[mcp]"
    biapy mcp --setup claude     # or codex; 'print' shows the configuration for any other agent

The agent's client then runs ``biapy mcp`` and talks to it through stdin/stdout.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import sysconfig
from typing import List, Optional

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from biapy import __version__, run_subcommand

INSTRUCTIONS = """\
BiaPy runs deep-learning workflows on microscopy images: semantic and instance segmentation, detection, denoising,
super-resolution, self-supervised learning, classification and image to image. Every tool returns the same report:
'ok', 'errors' (what is wrong), 'warnings' and 'data'.

Usual order:
1. inspect_image on one image (2D or 3D, channels) and list_workflows to choose the workflow.
2. create_config to write a configuration adapted to the data (or get_config_template and edit it), or
   search_bmz_models to start from a pretrained model.
3. plan: it validates the configuration and checks its data (no need to call validate_config or check_data before),
   and says what running it would do and whether the run already exists.
4. run starts the job in the background and returns at once; follow it with job_status (training epoch, test images,
   errors) and stop it with stop_job.
5. inspect_results when it ends: metrics, statistics of the predictions and warnings about what is plainly wrong.

For a job that already exists, start with inspect_results (or job_status while it runs). validate_config and
check_data are for checking a configuration or a data folder alone, e.g. while preparing them.
Never run 'biapy --config ...' directly: run checks the job first and lets job_status and stop_job follow it. A job
that finished is not necessarily good: look at inspect_results and report its limitations.
"""

server = MCPServer(name="biapy", version=__version__, instructions=INSTRUCTIONS)

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)


def _flag(name: str, value) -> List[str]:
    """``["--name", "value"]``, or nothing if ``value`` is None (or False, for a switch)."""
    if value is None or value is False:
        return []
    return [name] if value is True else [name, str(value)]


@server.tool(annotations=READ_ONLY)
def list_workflows() -> dict:
    """The workflows BiaPy can run (PROBLEM.TYPE), e.g. SEMANTIC_SEG or INSTANCE_SEG."""
    return run_subcommand("list-workflows")


@server.tool(annotations=READ_ONLY)
def inspect_image(path: str) -> dict:
    """
    Describe an image as BiaPy reads it: its stored shape, axes and type, how it is read as 2D and as 3D, and the
    suggested PROBLEM.NDIM. Use it before writing a configuration.
    """
    return run_subcommand("inspect-image", [path])


@server.tool()
def get_config_template(workflow: str, ndim: str, store: Optional[str] = None) -> dict:
    """
    The basic configuration template of a workflow ("2D" or "3D" ndim), to edit. With 'store', it is also written to
    that YAML file.
    """
    return run_subcommand("get-config-template", ["--workflow", workflow, "--ndim", ndim, *_flag("--store", store)])


@server.tool()
def create_config(
    workflow: str,
    ndim: str,
    train_raw: str,
    train_gt: Optional[str] = None,
    test_raw: Optional[str] = None,
    test_gt: Optional[str] = None,
    quick: bool = False,
    store: Optional[str] = None,
) -> dict:
    """
    Create a configuration adapted to the data, as the BiaPy wizard does: the data folders are checked and the model,
    patch size and training are planned from them and from this computer's GPUs. 'quick' gives a short training, to
    try it. With 'store', it is written to that YAML file. It can take a few minutes with large data.
    """
    args = ["--workflow", workflow, "--ndim", ndim, "--train-raw", train_raw]
    args += _flag("--train-gt", train_gt) + _flag("--test-raw", test_raw) + _flag("--test-gt", test_gt)
    args += _flag("--quick", quick) + _flag("--store", store)
    return run_subcommand("create-config", args)


@server.tool(annotations=READ_ONLY)
def validate_config(config: str, check_data_paths: bool = True) -> dict:
    """
    Check a configuration file (YAML): its values and, unless 'check_data_paths' is false, that its data exists. Not
    needed before 'plan', which already validates the configuration.
    """
    return run_subcommand("validate-config", [config, *_flag("--no-data-check", not check_data_paths)])


@server.tool(annotations=READ_ONLY)
def check_data(
    config: Optional[str] = None,
    workflow: Optional[str] = None,
    ndim: Optional[str] = None,
    raw: Optional[str] = None,
    gt: Optional[str] = None,
    percentage: Optional[float] = None,
) -> dict:
    """
    Check that the data can be used, as the BiaPy wizard does: images readable, same channels, masks with the right
    labels, each image with its target... Either of a configuration file ('config'), or of folders for a workflow
    ('workflow', 'ndim', 'raw' and optionally 'gt'). 'percentage' of each Zarr/H5 image to check (default 100).
    Not needed before 'plan', which already checks the data of a configuration.
    """
    args = [config] if config else []
    args += _flag("--workflow", workflow) + _flag("--ndim", ndim) + _flag("--raw", raw) + _flag("--gt", gt)
    args += _flag("--percentage", percentage)
    return run_subcommand("check-data", args)


@server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
def search_bmz_models(
    workflow: Optional[str] = None,
    ndim: Optional[str] = None,
    nchannels: Optional[int] = None,
    nclasses: Optional[int] = None,
    query: Optional[str] = None,
) -> dict:
    """
    Search the BioImage Model Zoo for pretrained models BiaPy can use, filtered by workflow, dimensions, channels,
    classes and words in their name or description. Needs internet and may take a minute.
    """
    args = _flag("--workflow", workflow) + _flag("--ndim", ndim) + _flag("--nchannels", nchannels)
    args += _flag("--nclasses", nclasses) + _flag("--query", query)
    return run_subcommand("search-bmz-models", args)


@server.tool(annotations=READ_ONLY)
def plan(
    config: str,
    result_dir: Optional[str] = None,
    name: Optional[str] = None,
    run_id: Optional[int] = None,
    percentage: Optional[float] = None,
) -> dict:
    """
    Describe what running a configuration would do, without running it: it validates the configuration and checks its
    data (as validate_config and check_data do), and gives the data it reads, the model, the training, this
    computer's GPUs and RAM, the resources it needs and, with 'result_dir' (and 'name', 'run_id'), where it would
    write and whether that run already exists.
    """
    args = [config, *_flag("--result_dir", result_dir), *_flag("--name", name), *_flag("--run_id", run_id)]
    return run_subcommand("plan", args + _flag("--percentage", percentage))


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False))
def run(
    config: str,
    result_dir: str,
    name: Optional[str] = None,
    run_id: Optional[int] = None,
    gpu: Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    """
    Start a job in the background and return at once, after checking it as 'plan' does: it is not started if the
    configuration or the data can't be used, or if the run exists (unless 'overwrite'). 'gpu' e.g. "0" (default: the
    CPU). Follow it with job_status and stop it with stop_job.
    """
    args = [config, "--result_dir", result_dir, *_flag("--name", name), *_flag("--run_id", run_id)]
    return run_subcommand("run", args + _flag("--gpu", gpu) + _flag("--overwrite", overwrite))


@server.tool(annotations=READ_ONLY)
def job_status(job_dir: str, run_id: Optional[int] = None) -> dict:
    """
    Whether a job is running, where it is (training epoch and estimated time, test images) and, if it failed, its
    error. 'job_dir' is '<result_dir>/<name>'.
    """
    return run_subcommand("job-status", [job_dir, *_flag("--run-id", run_id)])


@server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, openWorldHint=False))
def stop_job(job_dir: str, run_id: Optional[int] = None, timeout: Optional[float] = None) -> dict:
    """Stop a job started with 'run' (and the processes it started). 'job_dir' is '<result_dir>/<name>'."""
    return run_subcommand("stop-job", [job_dir, *_flag("--run-id", run_id), *_flag("--timeout", timeout)])


@server.tool(annotations=READ_ONLY)
def inspect_results(job_dir: str, run_id: Optional[int] = None) -> dict:
    """
    What a job left in its folder: whether it finished, its training (losses of every epoch, best epoch), test
    metrics (also by image), statistics of its predictions, checkpoints, outputs, and warnings about what is plainly
    wrong (e.g. images predicted as only background). 'job_dir' is '<result_dir>/<name>'.
    """
    return run_subcommand("inspect-results", [job_dir, *_flag("--run-id", run_id)])


#########
# Setup #
#########
def server_command() -> List[str]:
    """
    The command that starts this server: the 'biapy' of this environment, with its full path, and 'mcp'. The agents'
    clients don't have the environment (e.g. conda's) activated, so 'biapy' alone may not be found.
    """
    biapy = shutil.which("biapy", path=sysconfig.get_path("scripts"))
    if biapy:
        return [biapy, "mcp"]
    return [sys.executable, "-c", "from biapy import main; main()", "mcp"]  # e.g. BiaPy not installed with pip


def _run_cli(command: List[str]) -> bool:
    """Run an agent's command (e.g. 'claude mcp add ...'), and say whether it worked."""
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=60).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _setup_claude(command: List[str], folder: str) -> str:
    """
    Add the server to Claude Code: for the user with the 'claude' command if it is installed, or else in the
    '.mcp.json' of ``folder`` (Claude Code, also its VS Code extension, uses it when that folder is opened).
    """
    claude = shutil.which("claude")
    if claude:
        _run_cli([claude, "mcp", "remove", "--scope", "user", "biapy"])  # if it was added before
        if _run_cli([claude, "mcp", "add", "--scope", "user", "biapy", "--", *command]):
            return "Added to Claude Code for your user ('claude mcp add --scope user biapy ...')."
    path = os.path.join(os.path.abspath(folder), ".mcp.json")
    config = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            config = json.load(f)
    config.setdefault("mcpServers", {})["biapy"] = {"command": command[0], "args": command[1:]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    return f"Written to {path}: Claude Code uses it when {os.path.dirname(path)} is opened (it asks to approve it)."


def _codex_toml(command: List[str]) -> str:
    """The configuration of the server for Codex (its config.toml). JSON strings are valid TOML strings."""
    return f"[mcp_servers.biapy]\ncommand = {json.dumps(command[0])}\nargs = {json.dumps(command[1:])}\n"


def _setup_codex(command: List[str]) -> str:
    """Add the server to Codex: with the 'codex' command if it is installed, or else in ~/.codex/config.toml."""
    codex = shutil.which("codex")
    if codex:
        _run_cli([codex, "mcp", "remove", "biapy"])  # if it was added before
        if _run_cli([codex, "mcp", "add", "biapy", "--", *command]):
            return "Added to Codex ('codex mcp add biapy ...')."
    path = os.path.join(os.path.expanduser("~"), ".codex", "config.toml")
    text = ""
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
    if "[mcp_servers.biapy]" in text:
        return f"{path} already has a [mcp_servers.biapy] section: check that it is\n{_codex_toml(command)}"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(("\n" if text and not text.endswith("\n\n") else "") + _codex_toml(command))
    return f"Added to {path}."


def setup(agent: str, folder: str = ".") -> str:
    """
    Add this server to an agent: "claude" (Claude Code), "codex", or "print" (the configuration to add it to any
    other agent, e.g. Claude Desktop or Cursor). Returns what was done.
    """
    command = server_command()
    if agent == "claude":
        return _setup_claude(command, folder)
    if agent == "codex":
        return _setup_codex(command)
    mcp_json = json.dumps({"mcpServers": {"biapy": {"command": command[0], "args": command[1:]}}}, indent=2)
    return (
        "Claude Desktop (claude_desktop_config.json), Cursor (.cursor/mcp.json) and most agents:\n"
        f"{mcp_json}\n\nCodex (~/.codex/config.toml):\n{_codex_toml(command)}"
    )


def serve(argv=None):
    """
    Run the server (``biapy mcp``), through stdin/stdout as the agents' clients start it, or add it to an agent
    (``biapy mcp --setup claude``).
    """
    parser = argparse.ArgumentParser(prog="biapy mcp", description="Run BiaPy's MCP server, for AI agents.")
    parser.add_argument(
        "--setup",
        choices=["claude", "codex", "print"],
        help="Add the server to an agent instead of running it: Claude Code, Codex, or print its configuration",
    )
    parser.add_argument(
        "--folder",
        default=".",
        help="With --setup claude, without the 'claude' command: folder where to write .mcp.json (default: here)",
    )
    args = parser.parse_args(argv)
    if args.setup:
        print(setup(args.setup, args.folder))
        return
    server.run("stdio")

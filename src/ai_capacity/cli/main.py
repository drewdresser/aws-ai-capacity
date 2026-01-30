"""CLI entry point for AWS GPU Capacity Agent."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
import aioboto3
import typer
from pydantic_ai import (
    AgentRunResult,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from ai_capacity.agent.agent import capacity_agent
from ai_capacity.agent.deps import AgentDeps
from ai_capacity.agent.prompts import (
    AVAILABILITY_CHECK_PROMPT,
    CAPACITY_SEARCH_PROMPT,
    DAILY_CAPACITY_REPORT_PROMPT,
    TRAINING_PLAN_STATUS_PROMPT,
)
from ai_capacity.config import settings

app = typer.Typer(
    name="aws-ai-capacity",
    help="AWS GPU Capacity Management Agent CLI",
    no_args_is_help=True,
)
console = Console()


async def validate_aws_credentials(deps: AgentDeps) -> None:
    """Validate AWS credentials by making a simple STS call."""
    async with deps.session.client("sts") as sts:
        await sts.get_caller_identity()


def check_credentials(deps: AgentDeps) -> None:
    """Check AWS credentials and exit with helpful message if invalid."""
    try:
        asyncio.run(validate_aws_credentials(deps))
    except Exception as e:
        error_msg = str(e)
        console.print("[red bold]AWS credential error:[/red bold]")
        console.print(f"[red]{error_msg}[/red]")
        console.print()
        if "expired" in error_msg.lower() or "token" in error_msg.lower():
            console.print("[yellow]Try refreshing your SSO session:[/yellow]")
            console.print(f"  aws sso login --profile {settings.aws_profile or 'default'}")
        else:
            console.print("[yellow]Check your AWS configuration:[/yellow]")
            console.print("  - Verify AWS_PROFILE is set correctly")
            console.print("  - Ensure credentials are configured in ~/.aws/")
        raise typer.Exit(1)


def create_deps(region: str | None = None) -> AgentDeps:
    """Create agent dependencies from settings."""
    session = aioboto3.Session(
        region_name=region or settings.aws_region,
        profile_name=settings.aws_profile if settings.aws_profile else None,
    )
    return AgentDeps(
        session=session,
        region=region or settings.aws_region,
        account_id=settings.aws_account_id,
    )


async def run_agent_query(prompt: str, deps: AgentDeps, return_result: bool = False) -> str | AgentRunResult[str]:
    """Run a single agent query and return the response."""
    try:
        result = await capacity_agent.run(prompt, deps=deps)
        return result if return_result else result.output
    finally:
        await deps.close()


def print_tool_calls(result: AgentRunResult[str]) -> None:
    """Pretty print tool calls from the agent run."""
    console.print("\n[bold cyan]Agent Trajectory[/bold cyan]\n")

    tool_call_count = 0
    for message in result.all_messages():
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    tool_call_count += 1
                    console.print(f"[bold yellow]Tool Call #{tool_call_count}:[/bold yellow] {part.tool_name}")
                    if part.args:
                        args_json = json.dumps(part.args, indent=2, default=str)
                        console.print(Syntax(args_json, "json", theme="monokai", line_numbers=False))

        elif isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    # Truncate long responses
                    content = part.content
                    if isinstance(content, str) and len(content) > 500:
                        content = content[:500] + "... [truncated]"
                    elif not isinstance(content, str):
                        content = json.dumps(content, indent=2, default=str)
                        if len(content) > 500:
                            content = content[:500] + "... [truncated]"
                    console.print(f"[dim]Response:[/dim]")
                    console.print(Panel(str(content), border_style="dim"))

    console.print(f"\n[bold cyan]Total tool calls: {tool_call_count}[/bold cyan]\n")


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="Question or prompt for the agent"),
    region: str = typer.Option(
        None, "--region", "-r", help="AWS region to query"
    ),
    debug: bool = typer.Option(
        False, "--debug", "-d", help="Show tool calls made during execution"
    ),
) -> None:
    """Run a single query against the capacity agent."""
    deps = create_deps(region)
    check_credentials(deps)

    # Recreate deps since check_credentials runs the event loop
    deps = create_deps(region)
    with console.status("[bold green]Querying AWS capacity..."):
        result = asyncio.run(run_agent_query(prompt, deps, return_result=True))

    if debug:
        print_tool_calls(result)

    console.print(Panel(result.output, title="Agent Response", border_style="green"))


@app.command()
def report(
    report_type: str = typer.Argument(
        "daily",
        help="Report type: daily, availability, training-plans, search",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (prints to stdout if not specified)",
    ),
    region: str = typer.Option(None, "--region", "-r", help="AWS region"),
) -> None:
    """Generate a predefined capacity report."""
    prompts = {
        "daily": DAILY_CAPACITY_REPORT_PROMPT,
        "availability": AVAILABILITY_CHECK_PROMPT,
        "training-plans": TRAINING_PLAN_STATUS_PROMPT,
        "search": CAPACITY_SEARCH_PROMPT,
    }

    if report_type not in prompts:
        console.print(f"[red]Unknown report type: {report_type}[/red]")
        console.print(f"Available types: {', '.join(prompts.keys())}")
        raise typer.Exit(1)

    deps = create_deps(region)
    check_credentials(deps)

    deps = create_deps(region)
    with console.status(f"[bold green]Generating {report_type} report..."):
        response = asyncio.run(run_agent_query(prompts[report_type], deps))

    # Add report header
    timestamp = datetime.now().isoformat()
    report_content = f"""# AWS GPU Capacity Report
**Type**: {report_type}
**Generated**: {timestamp}
**Region**: {deps.region}

---

{response}
"""

    if output:
        output.write_text(report_content)
        console.print(f"[green]Report saved to: {output}[/green]")
    else:
        console.print(report_content)


@app.command()
def serve(
    host: str = typer.Option(
        settings.chainlit_host, "--host", "-h", help="Host to bind to"
    ),
    port: int = typer.Option(
        settings.chainlit_port, "--port", "-p", help="Port to bind to"
    ),
) -> None:
    """Start the Chainlit chat interface."""
    import subprocess
    import sys

    console.print(f"[green]Starting Chainlit UI on {host}:{port}[/green]")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "chainlit",
            "run",
            "src/ai_capacity/ui/app.py",
            "--host",
            host,
            "--port",
            str(port),
        ],
        check=True,
    )


@app.command("cron-report")
def cron_report(
    output_dir: Path = typer.Option(
        Path("./reports"),
        "--output-dir",
        "-d",
        help="Directory to save reports",
    ),
    region: str = typer.Option(None, "--region", "-r", help="AWS region"),
) -> None:
    """Generate all reports for cron job execution.

    This command generates all report types and saves them to the output directory.
    Designed to be run from a cron job for automated reporting.
    """
    deps = create_deps(region)
    check_credentials(deps)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    reports = [
        ("daily", DAILY_CAPACITY_REPORT_PROMPT),
        ("availability", AVAILABILITY_CHECK_PROMPT),
        ("training-plans", TRAINING_PLAN_STATUS_PROMPT),
    ]

    succeeded = []
    failed = []

    for report_name, prompt in reports:
        console.print(f"[blue]Generating {report_name} report...[/blue]")
        deps = create_deps(region)

        try:
            response = asyncio.run(run_agent_query(prompt, deps))

            output_file = output_dir / f"{report_name}_{timestamp}.md"
            report_content = f"""# AWS GPU Capacity Report: {report_name}
**Generated**: {datetime.now().isoformat()}
**Region**: {deps.region}

---

{response}
"""
            output_file.write_text(report_content)
            console.print(f"[green]  Saved: {output_file}[/green]")
            succeeded.append(report_name)

        except Exception as e:
            console.print(f"[red]  Error generating {report_name}: {e}[/red]")
            failed.append(report_name)

    if failed:
        console.print(f"[red]Failed: {', '.join(failed)}[/red]")
        console.print(f"[green]Succeeded: {', '.join(succeeded) if succeeded else 'none'}[/green]")
        raise typer.Exit(1)
    else:
        console.print("[green]All reports generated successfully![/green]")


@app.command()
def list_instance_types() -> None:
    """List all known GPU instance types and their specifications."""
    from ai_capacity.tools.ec2 import GPU_INSTANCE_SPECS

    console.print("\n[bold]AWS GPU Instance Types[/bold]\n")

    # Group by GPU type
    gpu_groups: dict[str, list[tuple[str, dict]]] = {}
    for instance_type, specs in sorted(GPU_INSTANCE_SPECS.items()):
        gpu_type = specs.get("gpu_type", "Unknown")
        if gpu_type not in gpu_groups:
            gpu_groups[gpu_type] = []
        gpu_groups[gpu_type].append((instance_type, specs))

    for gpu_type, instances in gpu_groups.items():
        console.print(f"\n[bold cyan]{gpu_type}[/bold cyan]")
        for instance_type, specs in instances:
            console.print(
                f"  {instance_type}: "
                f"{specs['gpus']}x GPU ({specs['gpu_memory_gb']}GB), "
                f"{specs['vcpus']} vCPUs, "
                f"{specs['memory_gb']}GB RAM"
            )


if __name__ == "__main__":
    app()

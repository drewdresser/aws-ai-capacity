# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
uv sync                              # Install dependencies
uv sync --all-extras                 # Install with dev dependencies
uv run aws-ai-capacity chat "query"      # Run a single agent query
uv run aws-ai-capacity serve             # Start Chainlit UI on localhost:8000
uv run aws-ai-capacity report daily      # Generate capacity report
uv run aws-ai-capacity list-instance-types  # List GPU specs
uv run pytest                        # Run tests
uv run ruff check .                  # Lint
```

## Architecture

This is a Pydantic AI agent that queries AWS APIs for GPU/ML capacity information, with two interfaces: a Chainlit chat UI and a Typer CLI.

### Key Components

**Agent (`src/ai_capacity/agent/`)**
- `agent.py` - Creates `capacity_agent` using Bedrock Claude, registers all tools
- `deps.py` - `AgentDeps` dataclass holds `aioboto3.Session` and provides cached AWS clients
- `prompts.py` - System prompt and predefined report prompts for cron jobs

**Tools (`src/ai_capacity/tools/`)**
- `sagemaker.py` - SageMaker Training Plans API: `search_training_plan_offerings`, `list_training_plans`, `describe_training_plan`
- `ec2.py` - EC2 capacity tools + `GPU_INSTANCE_SPECS` dict with hardware specs for all GPU instance types

**Interfaces**
- `cli/main.py` - Typer CLI with `chat`, `report`, `serve`, `cron-report` commands
- `ui/app.py` - Chainlit app with `@cl.on_message` handler that streams agent responses

### Tool Pattern

Tools are async functions that receive `RunContext[AgentDeps]` as first parameter:

```python
async def my_tool(ctx: RunContext[AgentDeps], param: str) -> dict:
    """Docstring becomes tool description for the LLM."""
    client = await ctx.deps.get_ec2_client(region="us-east-1")
    # ... call AWS API
```

Tools are registered on the agent via `capacity_agent.tool(fn)`.

### Configuration

Settings loaded from `.env` via Pydantic Settings in `config.py`:
- `AWS_PROFILE` - AWS credentials profile
- `BEDROCK_REGION` - Region for Bedrock LLM calls
- `BEDROCK_MODEL_ID` - Claude model ID
- `AWS_REGION` - Default region for capacity queries

# AWS AI Capacity - GPU Capacity Management Agent

A Pydantic AI-powered agent for managing and querying AWS GPU capacity, with a Chainlit chat interface and CLI for automated reporting.

## Features

- **SageMaker Training Plans**: Search offerings, list plans, check utilization
- **EC2 Capacity**: Query reservations, check instance availability
- **GPU Specs**: Compare instance types and their capabilities
- **Chat Interface**: Interactive Chainlit UI for ad-hoc queries
- **CLI Reports**: Automated report generation for cron jobs

## Installation

```bash
# Clone and install
cd ai-capacity
uv sync

# Copy environment template
cp .env.example .env
```

## Configuration

The agent uses **AWS Bedrock** for the LLM, so it uses your existing AWS credentials.

Required AWS permissions:
- `bedrock:InvokeModel` - For Claude model access
- `sagemaker:SearchTrainingPlanOfferings` - Query available plans
- `sagemaker:ListTrainingPlans` - List existing plans
- `sagemaker:DescribeTrainingPlan` - Get plan details
- `ec2:DescribeCapacityReservations` - Query reservations
- `ec2:DescribeInstanceTypeOfferings` - Check availability

Configure in `.env`:
```bash
AWS_REGION=us-east-1
AWS_PROFILE=default  # Optional
```

## Usage

### Chat Interface

Start the Chainlit UI:
```bash
uv run aws-ai-capacity serve
```

Then open http://localhost:8000

### CLI Commands

```bash
# Single query
uv run aws-ai-capacity chat "What p5 training plans are available?"

# Generate reports
uv run aws-ai-capacity report daily
uv run aws-ai-capacity report availability
uv run aws-ai-capacity report training-plans

# Save report to file
uv run aws-ai-capacity report daily -o capacity-report.md

# Generate all reports (for cron)
uv run aws-ai-capacity cron-report -d ./reports

# List GPU instance types
uv run aws-ai-capacity list-instance-types
```

### Cron Job Example

```bash
# Daily capacity report at 8am
0 8 * * * cd /path/to/ai-capacity && uv run aws-ai-capacity cron-report -d ./reports
```

## Example Questions

- "What p5.48xlarge training plans are available for the next week?"
- "Show me all active capacity reservations"
- "Which regions have p4d.24xlarge available?"
- "Compare the specs of p5 vs p4d instances"
- "Generate a daily capacity report"
- "Search for available H100 capacity"

## Project Structure

```
ai-capacity/
├── src/ai_capacity/
│   ├── agent/          # Pydantic AI agent definition
│   ├── tools/          # AWS API tools (SageMaker, EC2)
│   ├── cli/            # Typer CLI commands
│   ├── ui/             # Chainlit chat interface
│   └── config.py       # Settings management
├── chainlit.md         # Chat welcome message
└── .env.example        # Environment template
```

## Development

```bash
# Install with dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Lint
uv run ruff check .
```

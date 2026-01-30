"""Chainlit UI for AWS GPU Capacity Agent."""

import json

import aioboto3
import chainlit as cl
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)

from ai_capacity.agent.agent import capacity_agent
from ai_capacity.agent.deps import AgentDeps
from ai_capacity.config import settings


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize the agent session when a user connects."""
    # Create AWS session
    session = aioboto3.Session(
        region_name=settings.aws_region,
        profile_name=settings.aws_profile if settings.aws_profile else None,
    )

    # Create agent dependencies
    deps = AgentDeps(
        session=session,
        region=settings.aws_region,
        account_id=settings.aws_account_id,
    )

    # Store in session
    cl.user_session.set("deps", deps)
    cl.user_session.set("message_history", [])

    # Welcome message
    await cl.Message(
        content="""# AWS GPU Capacity Manager

I can help you with:

**SageMaker Training Plans**
- Search available training plan offerings
- List existing training plans
- Check plan utilization and availability

**EC2 Capacity**
- Query capacity reservations
- Check GPU instance availability by region/zone
- Compare GPU instance specifications

**Example Questions**
- "What p5.48xlarge training plans are available?"
- "Show me all active capacity reservations"
- "Which regions have p4d.24xlarge available?"
- "Compare the specs of p5 vs p4d instances"
- "Generate a daily capacity report"

What would you like to know about your GPU capacity?"""
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle incoming user messages."""
    deps: AgentDeps = cl.user_session.get("deps")
    message_history: list[ModelMessage] = cl.user_session.get("message_history")

    # Track tool calls for UI - maps tool_call_id to Step
    tool_steps: dict[str, cl.Step] = {}

    # Create response message for streaming text
    response_msg = cl.Message(content="")
    await response_msg.send()

    # Track which tool call IDs we've already shown
    seen_tool_calls: set[str] = set()
    accumulated_text = ""

    try:
        # Run agent with streaming
        async with capacity_agent.run_stream(
            message.content,
            deps=deps,
            message_history=message_history,
        ) as result:
            # Stream responses to capture tool calls and text as they happen
            async for model_response, is_last in result.stream_responses():
                # Process each part in the response
                for part in model_response.parts:
                    # Handle tool calls
                    if isinstance(part, ToolCallPart):
                        if part.tool_call_id and part.tool_call_id not in seen_tool_calls:
                            seen_tool_calls.add(part.tool_call_id)
                            step = cl.Step(
                                name=part.tool_name,
                                type="tool",
                            )
                            step.input = _format_tool_args(part.args)
                            tool_steps[part.tool_call_id] = step
                            await step.send()

                    # Handle text parts
                    elif isinstance(part, TextPart):
                        # Stream new text content
                        new_text = part.content
                        if new_text and len(new_text) > len(accumulated_text):
                            delta = new_text[len(accumulated_text):]
                            accumulated_text = new_text
                            await response_msg.stream_token(delta)

            # After streaming, check for tool returns in new_messages
            for msg in result.new_messages():
                if isinstance(msg, ModelRequest):
                    for part in msg.parts:
                        if isinstance(part, ToolReturnPart):
                            if part.tool_call_id in tool_steps:
                                step = tool_steps[part.tool_call_id]
                                step.output = _truncate_output(_format_tool_result(part.content))
                                await step.update()

            # Update message history with all new messages
            message_history.extend(result.new_messages())
            cl.user_session.set("message_history", message_history)

    except Exception as e:
        await response_msg.stream_token(f"\n\n**Error**: {e}")

    # Finalize response
    await response_msg.update()


@cl.on_chat_end
async def on_chat_end() -> None:
    """Clean up resources when chat ends."""
    deps: AgentDeps | None = cl.user_session.get("deps")
    if deps:
        await deps.close()


def _format_tool_args(args: dict) -> str:
    """Format tool arguments for display."""
    if not args:
        return "(no arguments)"
    lines = []
    for key, value in args.items():
        if value is not None:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines) if lines else "(no arguments)"


def _format_tool_result(content: object) -> str:
    """Format tool result for display."""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, indent=2, default=str)
    except (TypeError, ValueError):
        return str(content)


def _truncate_output(output: str, max_length: int = 2000) -> str:
    """Truncate long output for UI display."""
    if len(output) <= max_length:
        return output
    return output[:max_length] + f"\n... (truncated, {len(output)} chars total)"

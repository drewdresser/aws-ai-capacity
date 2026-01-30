"""Pydantic AI Agent for AWS GPU capacity management."""

from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

from ai_capacity.agent.deps import AgentDeps
from ai_capacity.agent.prompts import SYSTEM_PROMPT
from ai_capacity.config import settings
from ai_capacity.tools import ec2, sagemaker

# Create Bedrock provider with explicit region and profile for the LLM
# This region is for Bedrock API calls, not for capacity queries
_bedrock_provider = BedrockProvider(
    region_name=settings.bedrock_region,
    profile_name=settings.aws_profile,
)

# Create the Bedrock model with the provider
_bedrock_model = BedrockConverseModel(
    model_name=settings.bedrock_model_id,
    provider=_bedrock_provider,
)

# Create the agent with AWS Bedrock Claude model
capacity_agent: Agent[AgentDeps, str] = Agent(
    model=_bedrock_model,
    deps_type=AgentDeps,
    system_prompt=SYSTEM_PROMPT,
    retries=settings.agent_max_retries,
)

# Register SageMaker tools
capacity_agent.tool(sagemaker.search_training_plan_offerings)
capacity_agent.tool(sagemaker.list_training_plans)
capacity_agent.tool(sagemaker.describe_training_plan)

# Register EC2 tools
capacity_agent.tool(ec2.describe_capacity_reservations)
capacity_agent.tool(ec2.describe_instance_type_offerings)
capacity_agent.tool(ec2.discover_gpu_instance_types)  # Find what GPU instances exist
capacity_agent.tool(ec2.describe_instance_types_live)  # Live AWS API for accurate specs
capacity_agent.tool(ec2.get_gpu_instance_specs)
capacity_agent.tool(ec2.list_regions)
capacity_agent.tool(ec2.check_instance_availability_all_regions)

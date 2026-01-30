"""SageMaker Training Plans API tools."""

from datetime import datetime
from typing import Any

from pydantic_ai import RunContext

from ai_capacity.agent.deps import AgentDeps


async def search_training_plan_offerings(
    ctx: RunContext[AgentDeps],
    duration_hours: int,
    instance_count: int = 1,
    target_resources: list[str] | None = None,
    instance_type: str | None = None,
    start_time_after: datetime | None = None,
    end_time_before: datetime | None = None,
) -> list[dict[str, Any]]:
    """Search for available SageMaker Training Plan offerings.

    Use this tool to find what training plan capacity is available for purchase.
    Training plans provide reserved ML compute capacity at predetermined pricing.

    Args:
        ctx: Runtime context with AWS dependencies.
        duration_hours: Required duration in hours (1 to 87,600 hours).
            Common values: 24 (1 day), 168 (1 week), 720 (30 days).
        instance_count: Number of instances needed (1-256). Defaults to 1.
        target_resources: Resource types to include. Options: 'training-job',
            'hyperpod-cluster', 'endpoint'. Defaults to ['training-job'].
        instance_type: Filter by ML instance type (e.g., 'ml.p5.48xlarge',
            'ml.p4d.24xlarge', 'ml.trn1.32xlarge'). Note: Use 'ml.' prefix.
        start_time_after: Only show offerings available after this date.
        end_time_before: Only show offerings available before this date.

    Returns:
        List of available training plan offerings with pricing and availability.
        Each offering includes: offering_id, instance_type, instance_count,
        duration, upfront_fee, currency, and availability_zone.

    Example:
        Search for p5 instances for 1 week:
        >>> search_training_plan_offerings(
        ...     duration_hours=168,
        ...     instance_type='ml.p5.48xlarge',
        ...     instance_count=1
        ... )
    """
    client = await ctx.deps.get_sagemaker_client()

    # Build request parameters
    params: dict[str, Any] = {
        "DurationHours": duration_hours,
        "TargetResources": target_resources or ["training-job"],
        "InstanceCount": instance_count,
    }

    if instance_type:
        params["InstanceType"] = instance_type
    if start_time_after:
        params["StartTimeAfter"] = start_time_after
    if end_time_before:
        params["EndTimeBefore"] = end_time_before

    try:
        response = await client.search_training_plan_offerings(**params)
    except Exception as e:
        error_msg = str(e)
        return [{
            "error": error_msg,
            "hint": "Use discover_gpu_instance_types to find valid EC2 instance names, then add 'ml.' prefix for SageMaker (e.g., p5.48xlarge -> ml.p5.48xlarge). If the instance type doesn't exist on AWS, it won't work in SageMaker either.",
        }]

    offerings = []
    for offering in response.get("TrainingPlanOfferings", []):
        # Extract reserved capacity details
        reserved_capacities = offering.get("ReservedCapacityOfferings", [])
        capacity_info = reserved_capacities[0] if reserved_capacities else {}

        offerings.append({
            "offering_id": offering.get("TrainingPlanOfferingId"),
            "duration_hours": offering.get("DurationHours"),
            "duration_minutes": offering.get("DurationMinutes"),
            "upfront_fee": offering.get("UpfrontFee"),
            "currency": offering.get("CurrencyCode", "USD"),
            "target_resources": offering.get("TargetResources", []),
            "instance_type": capacity_info.get("InstanceType"),
            "instance_count": capacity_info.get("InstanceCount"),
            "availability_zone": capacity_info.get("AvailabilityZone"),
            "start_time": str(capacity_info.get("StartTime")) if capacity_info.get("StartTime") else None,
            "end_time": str(capacity_info.get("EndTime")) if capacity_info.get("EndTime") else None,
        })

    return offerings


async def list_training_plans(
    ctx: RunContext[AgentDeps],
    status_filter: str | None = None,
    sort_by: str = "TrainingPlanName",
    sort_order: str = "Ascending",
    max_results: int = 50,
) -> list[dict[str, Any]]:
    """List all SageMaker Training Plans in the account.

    Use this tool to see existing training plans and their current status.
    This shows reserved capacity that has already been purchased.

    Args:
        ctx: Runtime context with AWS dependencies.
        status_filter: Filter by status. Options: 'Pending', 'Active',
            'Scheduled', 'Expired', 'Failed'.
        sort_by: Field to sort by. Options: 'TrainingPlanName', 'StartTime', 'Status'.
        sort_order: Sort direction. Options: 'Ascending', 'Descending'.
        max_results: Maximum number of results to return (1-100).

    Returns:
        List of training plans with status, capacity, and utilization information.
        Each plan includes: plan_name, plan_arn, status, instance_type,
        total_instance_count, available_instance_count, in_use_instance_count.

    Example:
        Get all active plans:
        >>> list_training_plans(status_filter='Active')
    """
    client = await ctx.deps.get_sagemaker_client()

    params: dict[str, Any] = {
        "SortBy": sort_by,
        "SortOrder": sort_order,
        "MaxResults": min(max_results, 100),
    }

    if status_filter:
        params["Filters"] = [{"Name": "Status", "Value": status_filter}]

    try:
        response = await client.list_training_plans(**params)
    except Exception as e:
        error_msg = str(e)
        return [{
            "error": error_msg,
            "hint": "Check that you have the correct AWS permissions and region. Valid status filters are: 'Pending', 'Active', 'Scheduled', 'Expired', 'Failed'.",
        }]

    plans = []
    for plan in response.get("TrainingPlanSummaries", []):
        # Extract reserved capacity summaries
        reserved_summaries = plan.get("ReservedCapacitySummaries", [])
        capacity_info = reserved_summaries[0] if reserved_summaries else {}

        plans.append({
            "plan_arn": plan.get("TrainingPlanArn"),
            "plan_name": plan.get("TrainingPlanName"),
            "status": plan.get("Status"),
            "status_message": plan.get("StatusMessage"),
            "instance_type": capacity_info.get("InstanceType"),
            "availability_zone": capacity_info.get("AvailabilityZone"),
            "total_instance_count": plan.get("TotalInstanceCount"),
            "available_instance_count": plan.get("AvailableInstanceCount"),
            "in_use_instance_count": plan.get("InUseInstanceCount", 0),
            "start_time": str(plan.get("StartTime")) if plan.get("StartTime") else None,
            "end_time": str(plan.get("EndTime")) if plan.get("EndTime") else None,
            "duration_hours": plan.get("DurationHours"),
            "duration_minutes": plan.get("DurationMinutes"),
            "upfront_fee": plan.get("UpfrontFee"),
            "currency": plan.get("CurrencyCode", "USD"),
            "target_resources": plan.get("TargetResources", []),
        })

    return plans


async def describe_training_plan(
    ctx: RunContext[AgentDeps],
    plan_name: str,
) -> dict[str, Any]:
    """Get detailed information about a specific SageMaker Training Plan.

    Use this tool to get complete details about a training plan including
    its full configuration, current utilization, and availability.

    Args:
        ctx: Runtime context with AWS dependencies.
        plan_name: Name of the training plan to describe.

    Returns:
        Detailed training plan information including:
        - Plan configuration (instance type, count, duration)
        - Current status and availability
        - Usage statistics and remaining capacity
        - Reserved capacity summaries by availability zone

    Example:
        Get details for a specific plan:
        >>> describe_training_plan(plan_name='my-gpu-training-plan')
    """
    client = await ctx.deps.get_sagemaker_client()

    try:
        response = await client.describe_training_plan(TrainingPlanName=plan_name)
    except Exception as e:
        error_msg = str(e)
        return {
            "error": error_msg,
            "hint": f"Training plan '{plan_name}' may not exist. Use list_training_plans() first to see available plans.",
        }

    # Process reserved capacity summaries
    reserved_summaries = []
    for summary in response.get("ReservedCapacitySummaries", []):
        reserved_summaries.append({
            "reserved_capacity_arn": summary.get("ReservedCapacityArn"),
            "availability_zone": summary.get("AvailabilityZone"),
            "instance_type": summary.get("InstanceType"),
            "total_instance_count": summary.get("TotalInstanceCount"),
            "status": summary.get("Status"),
            "start_time": str(summary.get("StartTime")) if summary.get("StartTime") else None,
            "end_time": str(summary.get("EndTime")) if summary.get("EndTime") else None,
        })

    return {
        "plan_arn": response.get("TrainingPlanArn"),
        "plan_name": response.get("TrainingPlanName"),
        "status": response.get("Status"),
        "status_message": response.get("StatusMessage"),
        "start_time": str(response.get("StartTime")) if response.get("StartTime") else None,
        "end_time": str(response.get("EndTime")) if response.get("EndTime") else None,
        "duration_hours": response.get("DurationHours"),
        "duration_minutes": response.get("DurationMinutes"),
        "total_instance_count": response.get("TotalInstanceCount"),
        "available_instance_count": response.get("AvailableInstanceCount"),
        "available_spare_instance_count": response.get("AvailableSpareInstanceCount"),
        "in_use_instance_count": response.get("InUseInstanceCount", 0),
        "unhealthy_instance_count": response.get("UnhealthyInstanceCount", 0),
        "upfront_fee": response.get("UpfrontFee"),
        "currency": response.get("CurrencyCode", "USD"),
        "target_resources": response.get("TargetResources", []),
        "reserved_capacity_summaries": reserved_summaries,
    }

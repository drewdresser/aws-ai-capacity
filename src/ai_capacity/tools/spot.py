"""Spot and On-Demand capacity tools."""

from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic_ai import RunContext

from ai_capacity.agent.deps import AgentDeps
from ai_capacity.tools.ec2 import GPU_INSTANCE_SPECS, GPU_REGIONS

# Known GPU instance type prefixes for filtering
GPU_INSTANCE_PREFIXES = ("p3", "p3dn", "p4d", "p4de", "p5", "g5", "g6", "trn1", "trn1n", "inf2", "dl1")


async def describe_running_gpu_instances(
    ctx: RunContext[AgentDeps],
    instance_types: list[str] | None = None,
    region: str | None = None,
) -> list[dict[str, Any]]:
    """List running GPU instances in the account.

    Shows current GPU instance utilization including whether each instance
    is on-demand or spot. Use this to understand current consumption before
    requesting additional capacity.

    Args:
        ctx: Runtime context with AWS dependencies.
        instance_types: Filter to specific instance types. If None, returns
            all running GPU instances (p3/p4/p5/g5/g6/trn1/inf2 families).
        region: AWS region to check. Uses default region if not specified.

    Returns:
        List of running GPU instances with:
        - instance_id, instance_type, state, availability_zone
        - launch_time, lifecycle (spot or on-demand)
        - gpu_specs, tags
        - summary with counts by type and lifecycle
    """
    client = await ctx.deps.get_ec2_client(region=region)

    filters: list[dict[str, Any]] = [
        {"Name": "instance-state-name", "Values": ["running", "pending"]},
    ]

    if instance_types:
        filters.append({"Name": "instance-type", "Values": instance_types})

    try:
        paginator = client.get_paginator("describe_instances")
        instances = []
        async for page in paginator.paginate(Filters=filters):
            for reservation in page.get("Reservations", []):
                for inst in reservation.get("Instances", []):
                    inst_type = inst.get("InstanceType", "")

                    # If no specific types requested, filter to GPU families
                    if not instance_types:
                        if not any(inst_type.startswith(prefix) for prefix in GPU_INSTANCE_PREFIXES):
                            continue

                    lifecycle = inst.get("InstanceLifecycle", "on-demand")
                    gpu_specs = GPU_INSTANCE_SPECS.get(inst_type, {})
                    tags = {
                        t["Key"]: t["Value"]
                        for t in inst.get("Tags", [])
                    }

                    instances.append({
                        "instance_id": inst.get("InstanceId"),
                        "instance_type": inst_type,
                        "state": inst.get("State", {}).get("Name"),
                        "availability_zone": inst.get("Placement", {}).get("AvailabilityZone"),
                        "launch_time": str(inst.get("LaunchTime")) if inst.get("LaunchTime") else None,
                        "lifecycle": lifecycle,
                        "gpu_specs": gpu_specs if gpu_specs else None,
                        "tags": tags if tags else None,
                    })
    except Exception as e:
        return [{"error": str(e), "hint": "Check AWS permissions for ec2:DescribeInstances."}]

    # Add summary
    if instances:
        by_type: dict[str, int] = {}
        by_lifecycle: dict[str, int] = {}
        for inst in instances:
            by_type[inst["instance_type"]] = by_type.get(inst["instance_type"], 0) + 1
            by_lifecycle[inst["lifecycle"]] = by_lifecycle.get(inst["lifecycle"], 0) + 1

        instances.append({
            "summary": {
                "total_gpu_instances": len(instances),
                "by_instance_type": by_type,
                "by_lifecycle": by_lifecycle,
            }
        })

    return instances


async def get_spot_price_history(
    ctx: RunContext[AgentDeps],
    instance_types: list[str],
    hours_back: int = 24,
    region: str | None = None,
) -> list[dict[str, Any]]:
    """Get recent spot price history for GPU instance types.

    Price trends reveal capacity health: stable low prices indicate ample
    capacity, while price spikes indicate scarcity. Returns aggregated
    statistics (min/max/avg/latest) per instance type per availability zone.

    Args:
        ctx: Runtime context with AWS dependencies.
        instance_types: Instance types to check (e.g., ['p4d.24xlarge', 'g5.12xlarge']).
        hours_back: How far back to look in hours (default 24, max 168).
        region: AWS region. Uses default region if not specified.

    Returns:
        Aggregated spot price data per instance type per AZ:
        - instance_type, availability_zone
        - latest_price_usd, min_price_usd, max_price_usd, avg_price_usd
        - data_points count, oldest/newest timestamps
        - gpu_specs
    """
    client = await ctx.deps.get_ec2_client(region=region)
    hours_back = min(hours_back, 168)  # Cap at 1 week
    start_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)

    try:
        paginator = client.get_paginator("describe_spot_price_history")
        raw_prices: list[dict[str, Any]] = []

        async for page in paginator.paginate(
            InstanceTypes=instance_types,
            ProductDescriptions=["Linux/UNIX"],
            StartTime=start_time,
        ):
            for item in page.get("SpotPriceHistory", []):
                raw_prices.append(item)
    except Exception as e:
        return [{
            "error": str(e),
            "hint": "Check AWS permissions for ec2:DescribeSpotPriceHistory. Use discover_gpu_instance_types to find valid instance names.",
        }]

    # Aggregate by (instance_type, availability_zone)
    aggregated: dict[tuple[str, str], list[dict]] = {}
    for item in raw_prices:
        key = (item.get("InstanceType", ""), item.get("AvailabilityZone", ""))
        if key not in aggregated:
            aggregated[key] = []
        aggregated[key].append(item)

    results = []
    for (inst_type, az), items in sorted(aggregated.items()):
        prices = [float(item.get("SpotPrice", "0")) for item in items]
        timestamps = [item.get("Timestamp") for item in items if item.get("Timestamp")]

        gpu_specs = GPU_INSTANCE_SPECS.get(inst_type, {})

        results.append({
            "instance_type": inst_type,
            "availability_zone": az,
            "latest_price_usd": prices[0] if prices else None,
            "min_price_usd": min(prices) if prices else None,
            "max_price_usd": max(prices) if prices else None,
            "avg_price_usd": round(sum(prices) / len(prices), 4) if prices else None,
            "data_points": len(prices),
            "newest": str(max(timestamps)) if timestamps else None,
            "oldest": str(min(timestamps)) if timestamps else None,
            "gpu_specs": gpu_specs if gpu_specs else None,
        })

    if not results:
        return [{
            "message": "No spot price data found for the requested instance types.",
            "hint": "These instance types may not be available as spot instances in this region.",
            "instance_types": instance_types,
            "region": region or ctx.deps.region,
        }]

    return results


async def get_spot_placement_scores(
    ctx: RunContext[AgentDeps],
    instance_types: list[str],
    target_capacity: int = 1,
    regions: list[str] | None = None,
    single_availability_zone: bool = False,
) -> list[dict[str, Any]]:
    """Get spot placement scores indicating likelihood of getting spot capacity.

    Returns a score from 1-10 for each instance type in each region/AZ,
    where 10 means very likely to get capacity and 1 means very unlikely.
    This is the best way to assess spot availability without launching anything.

    Args:
        ctx: Runtime context with AWS dependencies.
        instance_types: Instance types to check (e.g., ['p4d.24xlarge', 'g5.12xlarge']).
        target_capacity: Number of instances needed (default 1).
        regions: Regions to check. If None, checks all common GPU regions.
        single_availability_zone: If True, requires all capacity in a single AZ
            (important for distributed training with EFA networking).

    Returns:
        List of placement scores:
        - instance_type, region, availability_zone_id (if single AZ)
        - score (1-10, higher is better)
        - gpu_specs

    Example:
        Check spot scores for p5 across regions:
        >>> get_spot_placement_scores(instance_types=['p5.48xlarge'])
    """
    target_regions = regions or GPU_REGIONS

    try:
        # The API must be called from any region; it accepts RegionNames as a param
        client = await ctx.deps.get_ec2_client(region="us-east-1")

        all_results = []
        next_token = None

        while True:
            kwargs: dict[str, Any] = {
                "InstanceTypes": instance_types,
                "TargetCapacity": target_capacity,
                "TargetCapacityUnitType": "units",
                "RegionNames": target_regions,
                "SingleAvailabilityZone": single_availability_zone,
            }
            if next_token:
                kwargs["NextToken"] = next_token

            response = await client.get_spot_placement_scores(**kwargs)

            for score_entry in response.get("SpotPlacementScores", []):
                inst_type = score_entry.get("InstanceTypes", [None])[0] if score_entry.get("InstanceTypes") else None
                region = score_entry.get("Region", "")
                gpu_specs = GPU_INSTANCE_SPECS.get(inst_type, {}) if inst_type else {}

                result: dict[str, Any] = {
                    "region": region,
                    "score": score_entry.get("Score"),
                }

                if inst_type:
                    result["instance_type"] = inst_type
                if score_entry.get("AvailabilityZoneId"):
                    result["availability_zone_id"] = score_entry["AvailabilityZoneId"]
                if gpu_specs:
                    result["gpu_specs"] = gpu_specs

                all_results.append(result)

            next_token = response.get("NextToken")
            if not next_token:
                break

    except Exception as e:
        return [{
            "error": str(e),
            "hint": "Check AWS permissions for ec2:GetSpotPlacementScores. This API may not be available in all accounts.",
        }]

    # Sort by score descending
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_results


async def _get_default_ami(ctx: RunContext[AgentDeps], region: str) -> str:
    """Resolve the latest Amazon Linux 2023 AMI ID for a region via SSM."""
    ssm = await ctx.deps.get_ssm_client(region=region)
    response = await ssm.get_parameter(
        Name="/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"
    )
    return response["Parameter"]["Value"]


async def check_on_demand_capacity(
    ctx: RunContext[AgentDeps],
    instance_type: str,
    availability_zone: str | None = None,
    region: str | None = None,
    confirm_launch: bool = False,
) -> dict[str, Any]:
    """Check on-demand capacity by launching and immediately terminating an instance.

    WARNING: This tool ACTUALLY LAUNCHES a real EC2 instance and immediately
    terminates it. This incurs a minimum billing charge (~$0.50-$30 depending
    on the instance type). Use this only when you need definitive confirmation
    of on-demand capacity availability.

    The confirm_launch parameter MUST be set to True. Before calling with
    confirm_launch=True, you MUST inform the user that:
    1. A real instance will be launched and immediately terminated
    2. There will be a minimum billing charge
    3. Ask for their explicit confirmation to proceed

    If capacity is unavailable (InsufficientInstanceCapacity), no instance is
    launched and there is no cost.

    Args:
        ctx: Runtime context with AWS dependencies.
        instance_type: Instance type to check (e.g., 'p4d.24xlarge').
        availability_zone: Specific AZ to check. If None, AWS picks the best AZ.
        region: AWS region. Uses default if not specified.
        confirm_launch: Must be True to proceed. Set to False (default) to get
            a warning message instead of launching.

    Returns:
        - capacity_available: True if launch succeeded, False if insufficient capacity
        - instance_id: ID of launched instance (if successful)
        - terminated: Whether the instance was successfully terminated
        - If terminated is False, the instance_id MUST be manually terminated
    """
    target_region = region or ctx.deps.region
    gpu_specs = GPU_INSTANCE_SPECS.get(instance_type, {})

    if not confirm_launch:
        return {
            "error": "confirm_launch must be True to proceed",
            "message": (
                "This tool will LAUNCH a real EC2 instance and immediately terminate it. "
                "This incurs a minimum billing charge. Before calling again with "
                "confirm_launch=True, you must inform the user and get their explicit "
                "confirmation."
            ),
            "instance_type": instance_type,
            "region": target_region,
            "gpu_specs": gpu_specs if gpu_specs else None,
        }

    # Resolve AMI
    try:
        ami_id = await _get_default_ami(ctx, target_region)
    except Exception as e:
        return {
            "error": f"Failed to resolve AMI: {e}",
            "hint": "Check AWS permissions for ssm:GetParameter.",
        }

    client = await ctx.deps.get_ec2_client(region=target_region)

    # Build launch params
    launch_kwargs: dict[str, Any] = {
        "InstanceType": instance_type,
        "ImageId": ami_id,
        "MinCount": 1,
        "MaxCount": 1,
    }
    if availability_zone:
        launch_kwargs["Placement"] = {"AvailabilityZone": availability_zone}

    # Attempt launch
    try:
        response = await client.run_instances(**launch_kwargs)
        instances = response.get("Instances", [])
        if not instances:
            return {
                "capacity_available": False,
                "error": "Launch returned no instances",
                "instance_type": instance_type,
                "region": target_region,
                "gpu_specs": gpu_specs if gpu_specs else None,
            }

        instance_id = instances[0]["InstanceId"]
        actual_az = instances[0].get("Placement", {}).get("AvailabilityZone", "unknown")

    except client.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "InsufficientInstanceCapacity":
            return {
                "capacity_available": False,
                "instance_type": instance_type,
                "region": target_region,
                "availability_zone": availability_zone or "any",
                "status": "InsufficientInstanceCapacity",
                "gpu_specs": gpu_specs if gpu_specs else None,
            }
        return {
            "error": str(e),
            "hint": "Check IAM permissions for ec2:RunInstances.",
            "instance_type": instance_type,
            "region": target_region,
        }
    except Exception as e:
        return {
            "error": str(e),
            "instance_type": instance_type,
            "region": target_region,
        }

    # Immediately terminate
    terminated = False
    try:
        await client.terminate_instances(InstanceIds=[instance_id])
        terminated = True
    except Exception as e:
        # Critical: instance is running but we failed to terminate
        return {
            "capacity_available": True,
            "instance_id": instance_id,
            "instance_type": instance_type,
            "region": target_region,
            "availability_zone": actual_az,
            "terminated": False,
            "termination_error": str(e),
            "WARNING": f"Instance {instance_id} was launched but FAILED TO TERMINATE. Manual cleanup required!",
            "gpu_specs": gpu_specs if gpu_specs else None,
        }

    return {
        "capacity_available": True,
        "instance_id": instance_id,
        "instance_type": instance_type,
        "region": target_region,
        "availability_zone": actual_az,
        "terminated": terminated,
        "gpu_specs": gpu_specs if gpu_specs else None,
    }

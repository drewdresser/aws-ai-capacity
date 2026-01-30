"""EC2 Capacity and Instance Availability tools."""

from typing import Any

from pydantic_ai import RunContext

from ai_capacity.agent.deps import AgentDeps

# Common regions for GPU/ML workloads
GPU_REGIONS = [
    "us-east-1",
    "us-east-2",
    "us-west-2",
    "eu-west-1",
    "eu-west-2",
    "eu-central-1",
    "ap-northeast-1",
    "ap-southeast-1",
    "ap-southeast-2",
]

# GPU instance specifications lookup
GPU_INSTANCE_SPECS: dict[str, dict[str, Any]] = {
    # P5 (H100)
    "p5.48xlarge": {
        "gpus": 8,
        "gpu_type": "NVIDIA H100",
        "gpu_memory_gb": 640,
        "vcpus": 192,
        "memory_gb": 2048,
        "network": "3200 Gbps EFA",
    },
    # P4d/P4de (A100)
    "p4d.24xlarge": {
        "gpus": 8,
        "gpu_type": "NVIDIA A100",
        "gpu_memory_gb": 320,
        "vcpus": 96,
        "memory_gb": 1152,
        "network": "400 Gbps EFA",
    },
    "p4de.24xlarge": {
        "gpus": 8,
        "gpu_type": "NVIDIA A100 (80GB)",
        "gpu_memory_gb": 640,
        "vcpus": 96,
        "memory_gb": 1152,
        "network": "400 Gbps EFA",
    },
    # P3 (V100)
    "p3.2xlarge": {
        "gpus": 1,
        "gpu_type": "NVIDIA V100",
        "gpu_memory_gb": 16,
        "vcpus": 8,
        "memory_gb": 61,
        "network": "10 Gbps",
    },
    "p3.8xlarge": {
        "gpus": 4,
        "gpu_type": "NVIDIA V100",
        "gpu_memory_gb": 64,
        "vcpus": 32,
        "memory_gb": 244,
        "network": "10 Gbps",
    },
    "p3.16xlarge": {
        "gpus": 8,
        "gpu_type": "NVIDIA V100",
        "gpu_memory_gb": 128,
        "vcpus": 64,
        "memory_gb": 488,
        "network": "25 Gbps",
    },
    "p3dn.24xlarge": {
        "gpus": 8,
        "gpu_type": "NVIDIA V100 (32GB)",
        "gpu_memory_gb": 256,
        "vcpus": 96,
        "memory_gb": 768,
        "network": "100 Gbps EFA",
    },
    # G5 (A10G)
    "g5.xlarge": {
        "gpus": 1,
        "gpu_type": "NVIDIA A10G",
        "gpu_memory_gb": 24,
        "vcpus": 4,
        "memory_gb": 16,
        "network": "10 Gbps",
    },
    "g5.2xlarge": {
        "gpus": 1,
        "gpu_type": "NVIDIA A10G",
        "gpu_memory_gb": 24,
        "vcpus": 8,
        "memory_gb": 32,
        "network": "10 Gbps",
    },
    "g5.4xlarge": {
        "gpus": 1,
        "gpu_type": "NVIDIA A10G",
        "gpu_memory_gb": 24,
        "vcpus": 16,
        "memory_gb": 64,
        "network": "25 Gbps",
    },
    "g5.8xlarge": {
        "gpus": 1,
        "gpu_type": "NVIDIA A10G",
        "gpu_memory_gb": 24,
        "vcpus": 32,
        "memory_gb": 128,
        "network": "25 Gbps",
    },
    "g5.12xlarge": {
        "gpus": 4,
        "gpu_type": "NVIDIA A10G",
        "gpu_memory_gb": 96,
        "vcpus": 48,
        "memory_gb": 192,
        "network": "40 Gbps",
    },
    "g5.24xlarge": {
        "gpus": 4,
        "gpu_type": "NVIDIA A10G",
        "gpu_memory_gb": 96,
        "vcpus": 96,
        "memory_gb": 384,
        "network": "50 Gbps",
    },
    "g5.48xlarge": {
        "gpus": 8,
        "gpu_type": "NVIDIA A10G",
        "gpu_memory_gb": 192,
        "vcpus": 192,
        "memory_gb": 768,
        "network": "100 Gbps",
    },
    # G6 (L4)
    "g6.xlarge": {
        "gpus": 1,
        "gpu_type": "NVIDIA L4",
        "gpu_memory_gb": 24,
        "vcpus": 4,
        "memory_gb": 16,
        "network": "10 Gbps",
    },
    "g6.2xlarge": {
        "gpus": 1,
        "gpu_type": "NVIDIA L4",
        "gpu_memory_gb": 24,
        "vcpus": 8,
        "memory_gb": 32,
        "network": "10 Gbps",
    },
    "g6.12xlarge": {
        "gpus": 4,
        "gpu_type": "NVIDIA L4",
        "gpu_memory_gb": 96,
        "vcpus": 48,
        "memory_gb": 192,
        "network": "40 Gbps",
    },
    "g6.48xlarge": {
        "gpus": 8,
        "gpu_type": "NVIDIA L4",
        "gpu_memory_gb": 192,
        "vcpus": 192,
        "memory_gb": 768,
        "network": "100 Gbps",
    },
    # Trainium (trn1)
    "trn1.2xlarge": {
        "gpus": 1,
        "gpu_type": "AWS Trainium",
        "gpu_memory_gb": 32,
        "vcpus": 8,
        "memory_gb": 32,
        "network": "12.5 Gbps",
    },
    "trn1.32xlarge": {
        "gpus": 16,
        "gpu_type": "AWS Trainium",
        "gpu_memory_gb": 512,
        "vcpus": 128,
        "memory_gb": 512,
        "network": "800 Gbps EFA",
    },
    "trn1n.32xlarge": {
        "gpus": 16,
        "gpu_type": "AWS Trainium",
        "gpu_memory_gb": 512,
        "vcpus": 128,
        "memory_gb": 512,
        "network": "1600 Gbps EFA",
    },
    # Inferentia (inf2)
    "inf2.xlarge": {
        "gpus": 1,
        "gpu_type": "AWS Inferentia2",
        "gpu_memory_gb": 32,
        "vcpus": 4,
        "memory_gb": 16,
        "network": "15 Gbps",
    },
    "inf2.8xlarge": {
        "gpus": 1,
        "gpu_type": "AWS Inferentia2",
        "gpu_memory_gb": 32,
        "vcpus": 32,
        "memory_gb": 128,
        "network": "25 Gbps",
    },
    "inf2.24xlarge": {
        "gpus": 6,
        "gpu_type": "AWS Inferentia2",
        "gpu_memory_gb": 192,
        "vcpus": 96,
        "memory_gb": 384,
        "network": "50 Gbps",
    },
    "inf2.48xlarge": {
        "gpus": 12,
        "gpu_type": "AWS Inferentia2",
        "gpu_memory_gb": 384,
        "vcpus": 192,
        "memory_gb": 768,
        "network": "100 Gbps",
    },
}


async def describe_capacity_reservations(
    ctx: RunContext[AgentDeps],
    instance_type: str | None = None,
    state: str | None = None,
    availability_zone: str | None = None,
) -> list[dict[str, Any]]:
    """Query EC2 Capacity Reservations for GPU instances.

    Use this tool to check existing capacity reservations that guarantee
    EC2 instance availability. Reservations ensure capacity is available
    when you need to launch instances.

    Args:
        ctx: Runtime context with AWS dependencies.
        instance_type: Filter by instance type (e.g., 'p4d.24xlarge', 'g5.xlarge').
        state: Filter by state. Options: 'active', 'expired', 'cancelled',
            'pending', 'failed'.
        availability_zone: Filter by specific AZ (e.g., 'us-east-1a').

    Returns:
        List of capacity reservations with:
        - reservation_id and state
        - instance_type and platform
        - total and available instance count
        - availability_zone
        - start and end dates
        - GPU specifications for the instance type

    Example:
        Check active p4d reservations:
        >>> describe_capacity_reservations(
        ...     instance_type='p4d.24xlarge',
        ...     state='active'
        ... )
    """
    client = await ctx.deps.get_ec2_client()

    filters = []
    if instance_type:
        filters.append({"Name": "instance-type", "Values": [instance_type]})
    if state:
        filters.append({"Name": "state", "Values": [state]})
    if availability_zone:
        filters.append({"Name": "availability-zone", "Values": [availability_zone]})

    try:
        response = await client.describe_capacity_reservations(
            Filters=filters if filters else []
        )
    except Exception as e:
        return [{
            "error": str(e),
            "hint": "Check AWS permissions and region. Valid states: 'active', 'expired', 'cancelled', 'pending', 'failed'. Use discover_gpu_instance_types to find valid instance names.",
        }]

    reservations = []
    for res in response.get("CapacityReservations", []):
        instance_type_val = res.get("InstanceType", "")
        gpu_specs = GPU_INSTANCE_SPECS.get(instance_type_val, {})

        reservations.append({
            "reservation_id": res.get("CapacityReservationId"),
            "reservation_arn": res.get("CapacityReservationArn"),
            "state": res.get("State"),
            "instance_type": instance_type_val,
            "instance_platform": res.get("InstancePlatform"),
            "availability_zone": res.get("AvailabilityZone"),
            "total_instance_count": res.get("TotalInstanceCount"),
            "available_instance_count": res.get("AvailableInstanceCount"),
            "start_date": str(res.get("StartDate")) if res.get("StartDate") else None,
            "end_date": str(res.get("EndDate")) if res.get("EndDate") else None,
            "end_date_type": res.get("EndDateType"),
            "instance_match_criteria": res.get("InstanceMatchCriteria"),
            "tenancy": res.get("Tenancy"),
            "gpu_specs": gpu_specs if gpu_specs else None,
        })

    return reservations


async def describe_instance_type_offerings(
    ctx: RunContext[AgentDeps],
    instance_types: list[str],
    location_type: str = "availability-zone",
    region: str | None = None,
) -> list[dict[str, Any]]:
    """Check GPU instance type availability by region and availability zone.

    Use this tool to discover where specific GPU instance types are available.
    This helps identify which regions and zones support the instances you need.

    Args:
        ctx: Runtime context with AWS dependencies.
        instance_types: List of instance types to check. Examples:
            ['p5.48xlarge', 'p4d.24xlarge', 'trn1.32xlarge'].
        location_type: Granularity of location. Options:
            'region', 'availability-zone', 'availability-zone-id'.
        region: Specific region to check. Uses default region if not specified.

    Returns:
        List of offerings showing where each instance type is available:
        - instance_type
        - location (region, zone, or zone ID based on location_type)
        - GPU specifications for the instance

    Example:
        Check p5 availability across zones in us-east-1:
        >>> describe_instance_type_offerings(
        ...     instance_types=['p5.48xlarge'],
        ...     location_type='availability-zone',
        ...     region='us-east-1'
        ... )
    """
    client = await ctx.deps.get_ec2_client(region=region)

    try:
        response = await client.describe_instance_type_offerings(
            LocationType=location_type,
            Filters=[{"Name": "instance-type", "Values": instance_types}],
        )
    except Exception as e:
        return [{
            "error": str(e),
            "hint": "Check instance type names using discover_gpu_instance_types first. Valid location_types: 'region', 'availability-zone', 'availability-zone-id'.",
        }]

    offerings = []
    for offering in response.get("InstanceTypeOfferings", []):
        instance_type = offering.get("InstanceType", "")
        gpu_specs = GPU_INSTANCE_SPECS.get(instance_type, {})

        offerings.append({
            "instance_type": instance_type,
            "location": offering.get("Location"),
            "location_type": offering.get("LocationType"),
            "gpu_specs": gpu_specs if gpu_specs else None,
        })

    return offerings


async def get_gpu_instance_specs(
    ctx: RunContext[AgentDeps],
    instance_types: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Get detailed GPU specifications for instance types.

    Use this tool to understand the GPU capabilities of different instance types.
    Returns detailed hardware specifications including GPU count, memory, and type.

    Args:
        ctx: Runtime context with AWS dependencies.
        instance_types: Specific instance types to query. If None, returns
            all known GPU instances.

    Returns:
        List of instance specifications:
        - instance_type: Instance type name
        - gpus: Number of GPUs/accelerators
        - gpu_type: GPU model (H100, A100, V100, Trainium, etc.)
        - gpu_memory_gb: Total GPU memory in GB
        - vcpus: Number of vCPUs
        - memory_gb: System memory in GB
        - network: Network bandwidth

    Example:
        Compare p5 and p4d specs:
        >>> get_gpu_instance_specs(
        ...     instance_types=['p5.48xlarge', 'p4d.24xlarge']
        ... )
    """
    if instance_types is None:
        # Return all known GPU instances
        return [
            {"instance_type": k, **v}
            for k, v in sorted(GPU_INSTANCE_SPECS.items())
        ]

    return [
        {"instance_type": it, **GPU_INSTANCE_SPECS[it]}
        for it in instance_types
        if it in GPU_INSTANCE_SPECS
    ]


async def discover_gpu_instance_types(
    ctx: RunContext[AgentDeps],
    region: str | None = None,
) -> list[dict[str, Any]]:
    """Discover all GPU/accelerator instance types available on AWS.

    Use this tool FIRST to find what GPU instance types exist before querying
    specific specs. This queries AWS directly to get the current list of
    instance types with GPUs or accelerators.

    Args:
        ctx: Runtime context with AWS dependencies.
        region: AWS region to query. Defaults to us-east-1.

    Returns:
        List of all GPU instance types with their specifications including:
        - instance_type: Instance type name
        - gpu_manufacturer: GPU manufacturer (NVIDIA, AMD, AWS)
        - gpu_name: GPU model name (H100, A100, Trainium, etc.)
        - gpu_count: Number of GPUs
        - gpu_memory_mb: Total GPU memory

    Example:
        Find all available GPU instances:
        >>> discover_gpu_instance_types()
    """
    client = await ctx.deps.get_ec2_client(region=region or "us-east-1")

    results = []

    try:
        paginator = client.get_paginator("describe_instance_types")

        # Filter for instances with GPUs or accelerators
        async for page in paginator.paginate(
            Filters=[
                {"Name": "processor-info.supported-architecture", "Values": ["x86_64", "arm64"]},
            ]
        ):
            for it in page.get("InstanceTypes", []):
                # Check if it has GPU info
                if "GpuInfo" not in it or not it["GpuInfo"].get("Gpus"):
                    continue

                gpus = it["GpuInfo"]["Gpus"]
                for gpu in gpus:
                    results.append({
                        "instance_type": it.get("InstanceType"),
                        "gpu_manufacturer": gpu.get("Manufacturer"),
                        "gpu_name": gpu.get("Name"),
                        "gpu_count": gpu.get("Count"),
                        "gpu_memory_mb": gpu.get("MemoryInfo", {}).get("SizeInMiB"),
                        "total_gpu_memory_mb": it["GpuInfo"].get("TotalGpuMemoryInMiB"),
                        "vcpus": it.get("VCpuInfo", {}).get("DefaultVCpus"),
                        "memory_mb": it.get("MemoryInfo", {}).get("SizeInMiB"),
                        "network_performance": it.get("NetworkInfo", {}).get("NetworkPerformance"),
                    })
    except Exception as e:
        return [{
            "error": str(e),
            "hint": "Check AWS permissions and that the region exists. Try a different region like 'us-east-1'.",
        }]

    # Sort by GPU name then instance type
    results.sort(key=lambda x: (x.get("gpu_name", ""), x.get("instance_type", "")))
    return results


async def describe_instance_types_live(
    ctx: RunContext[AgentDeps],
    instance_types: list[str],
    region: str | None = None,
) -> list[dict[str, Any]]:
    """Get LIVE instance type specifications directly from AWS API.

    Use this tool to get accurate, up-to-date GPU and instance specifications
    directly from AWS. If you're unsure what instance types exist, use
    `discover_gpu_instance_types` first.

    Args:
        ctx: Runtime context with AWS dependencies.
        instance_types: List of instance types to query (e.g., ['p5.48xlarge']).
            Use discover_gpu_instance_types first if unsure of valid names.
        region: AWS region to query. Defaults to us-east-1.

    Returns:
        List of instance specifications from AWS including:
        - instance_type: Instance type name
        - vcpus: Number of vCPUs
        - memory_mb: Memory in MB
        - gpu_info: GPU details (manufacturer, name, count, memory)
        - network_performance: Network bandwidth description

    Example:
        Get live specs for p5 instances:
        >>> describe_instance_types_live(instance_types=['p5.48xlarge'])
    """
    client = await ctx.deps.get_ec2_client(region=region or "us-east-1")

    try:
        response = await client.describe_instance_types(InstanceTypes=instance_types)
    except Exception as e:
        return [{"error": str(e), "hint": "Use discover_gpu_instance_types to find valid instance type names"}]

    results = []
    for it in response.get("InstanceTypes", []):
        gpu_info = None
        if "GpuInfo" in it and it["GpuInfo"].get("Gpus"):
            gpus = it["GpuInfo"]["Gpus"]
            gpu_info = {
                "total_gpus": sum(g.get("Count", 0) for g in gpus),
                "total_gpu_memory_mb": it["GpuInfo"].get("TotalGpuMemoryInMiB"),
                "gpus": [
                    {
                        "manufacturer": g.get("Manufacturer"),
                        "name": g.get("Name"),
                        "count": g.get("Count"),
                        "memory_mb": g.get("MemoryInfo", {}).get("SizeInMiB"),
                    }
                    for g in gpus
                ],
            }

        results.append({
            "instance_type": it.get("InstanceType"),
            "vcpus": it.get("VCpuInfo", {}).get("DefaultVCpus"),
            "memory_mb": it.get("MemoryInfo", {}).get("SizeInMiB"),
            "gpu_info": gpu_info,
            "network_performance": it.get("NetworkInfo", {}).get("NetworkPerformance"),
            "hypervisor": it.get("Hypervisor"),
            "supported_architectures": it.get("ProcessorInfo", {}).get("SupportedArchitectures"),
        })

    return results


async def list_regions(
    ctx: RunContext[AgentDeps],
) -> list[dict[str, Any]]:
    """List AWS regions commonly used for GPU/ML workloads.

    Use this tool to get a list of regions where GPU instances are typically
    available. This helps when planning multi-region capacity queries.

    Args:
        ctx: Runtime context with AWS dependencies.

    Returns:
        List of regions with their names and descriptions.

    Example:
        Get available regions for GPU workloads:
        >>> list_regions()
    """
    region_info = {
        "us-east-1": "US East (N. Virginia)",
        "us-east-2": "US East (Ohio)",
        "us-west-2": "US West (Oregon)",
        "eu-west-1": "Europe (Ireland)",
        "eu-west-2": "Europe (London)",
        "eu-central-1": "Europe (Frankfurt)",
        "ap-northeast-1": "Asia Pacific (Tokyo)",
        "ap-southeast-1": "Asia Pacific (Singapore)",
        "ap-southeast-2": "Asia Pacific (Sydney)",
    }

    return [
        {"region": region, "name": name}
        for region, name in region_info.items()
    ]


async def check_instance_availability_all_regions(
    ctx: RunContext[AgentDeps],
    instance_types: list[str],
    regions: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Check GPU instance availability across multiple regions.

    Use this tool to find where specific GPU instance types are available
    across multiple AWS regions at once.

    Args:
        ctx: Runtime context with AWS dependencies.
        instance_types: List of instance types to check. Examples:
            ['p5.48xlarge', 'p4d.24xlarge', 'trn1.32xlarge'].
        regions: List of regions to check. If None, checks all common GPU regions:
            us-east-1, us-east-2, us-west-2, eu-west-1, eu-central-1,
            ap-northeast-1, ap-southeast-1, ap-southeast-2.

    Returns:
        List of availability results by region:
        - region: AWS region code
        - instance_type: Instance type
        - available_zones: List of availability zones where the instance is available
        - gpu_specs: GPU specifications for the instance

    Example:
        Check p5 availability across all regions:
        >>> check_instance_availability_all_regions(
        ...     instance_types=['p5.48xlarge']
        ... )
    """
    target_regions = regions or GPU_REGIONS
    results = []

    for region in target_regions:
        try:
            client = await ctx.deps.get_ec2_client(region=region)

            response = await client.describe_instance_type_offerings(
                LocationType="availability-zone",
                Filters=[{"Name": "instance-type", "Values": instance_types}],
            )

            # Group by instance type
            by_type: dict[str, list[str]] = {}
            for offering in response.get("InstanceTypeOfferings", []):
                it = offering.get("InstanceType", "")
                zone = offering.get("Location", "")
                if it not in by_type:
                    by_type[it] = []
                by_type[it].append(zone)

            for instance_type in instance_types:
                zones = by_type.get(instance_type, [])
                gpu_specs = GPU_INSTANCE_SPECS.get(instance_type, {})
                results.append({
                    "region": region,
                    "instance_type": instance_type,
                    "available": len(zones) > 0,
                    "available_zones": sorted(zones),
                    "zone_count": len(zones),
                    "gpu_specs": gpu_specs if gpu_specs else None,
                })

        except Exception as e:
            # If we can't query a region, note the error but continue
            for instance_type in instance_types:
                results.append({
                    "region": region,
                    "instance_type": instance_type,
                    "available": False,
                    "available_zones": [],
                    "zone_count": 0,
                    "error": str(e),
                })

    return results

"""System prompts and report templates for the capacity agent."""

SYSTEM_PROMPT = """You are an AWS GPU and ML Capacity Management specialist.
Your role is to help users understand and manage their GPU compute capacity on AWS.

IMPORTANT: Always use your tools to look up instance specifications and availability.
Do NOT rely on your training data for GPU specs, instance types, or availability -
this information changes frequently and you may hallucinate incorrect instance names.

When asked about GPU types (like B200, H100, etc.):
1. First use `discover_gpu_instance_types` to see what GPU instances actually exist on AWS
2. Then use `describe_instance_types_live` for detailed specs on specific instances
3. Never guess instance type names - always discover them first

ERROR HANDLING: Tool responses may contain an "error" field with error messages and hints.
When you see an error:
1. Read the error message and hint carefully
2. Understand what went wrong (e.g., invalid instance type name)
3. Use the suggested corrective action (e.g., discover valid instance types first)
4. Retry the operation with corrected parameters
5. If B200 or other new GPUs aren't found, inform the user they may not be available on AWS yet

You have access to tools that query:
1. **SageMaker Training Plans** - Reserved ML capacity for training workloads
2. **EC2 Capacity Reservations** - Reserved EC2 GPU instance capacity
3. **Instance Type Availability** - Real-time GPU instance availability by region/zone

When providing information:
- Always specify the region and availability zone when relevant
- Include instance types with their GPU specifications (GPU count, memory, architecture)
- Format capacity information in clear tables when appropriate
- Highlight any capacity constraints or limitations
- Suggest alternatives when requested capacity is unavailable

## GPU Instance Families Reference

### NVIDIA H100 (Latest Generation)
- **p5.48xlarge**: 8x H100 (640GB HBM3), 192 vCPUs, 2TB RAM, 3200 Gbps EFA

### NVIDIA A100 (High Performance Training)
- **p4d.24xlarge**: 8x A100 40GB (320GB total), 96 vCPUs, 1.1TB RAM
- **p4de.24xlarge**: 8x A100 80GB (640GB total), 96 vCPUs, 1.1TB RAM

### NVIDIA V100 (Previous Generation)
- **p3.16xlarge**: 8x V100 (128GB), 64 vCPUs, 488GB RAM
- **p3dn.24xlarge**: 8x V100 32GB (256GB), 96 vCPUs, 768GB RAM

### NVIDIA A10G (Inference/Training)
- **g5.xlarge** to **g5.48xlarge**: 1-8x A10G (24-192GB)

### NVIDIA L4 (Inference)
- **g6.xlarge** to **g6.48xlarge**: 1-8x L4 (24-192GB)

### AWS Trainium (Custom ML Training)
- **trn1.32xlarge**: 16x Trainium (512GB), 128 vCPUs, 512GB RAM
- **trn1n.32xlarge**: Same with 1600 Gbps EFA networking

### AWS Inferentia2 (Inference)
- **inf2.xlarge** to **inf2.48xlarge**: 1-12x Inferentia2 (32-384GB)

## SageMaker ML Instance Prefixes
When querying SageMaker Training Plans, instance types use the `ml.` prefix:
- `ml.p5.48xlarge`, `ml.p4d.24xlarge`, `ml.trn1.32xlarge`, etc.

Always be concise but thorough in your responses."""

# Report templates for cron jobs
DAILY_CAPACITY_REPORT_PROMPT = """Generate a comprehensive daily capacity report covering:

1. **SageMaker Training Plans Status**
   - List all active training plans with utilization metrics
   - Identify plans with low available capacity (<20%)
   - Flag any plans expiring in the next 7 days

2. **EC2 Capacity Reservations Status**
   - List all active GPU capacity reservations
   - Show utilization (available vs total instances)
   - Highlight any reservations expiring soon

3. **Capacity Recommendations**
   - Identify underutilized capacity that could be released
   - Suggest capacity that may need expansion based on usage trends
   - Note any optimization opportunities

Format the report with clear sections, tables where appropriate, and actionable insights."""

AVAILABILITY_CHECK_PROMPT = """Check current GPU instance availability and provide a status report:

1. **High-Performance Training Instances**
   Query availability for these instance types:
   - p5.48xlarge (H100)
   - p4d.24xlarge (A100)
   - trn1.32xlarge (Trainium)

2. **Focus Regions**
   Check availability in:
   - us-east-1
   - us-west-2
   - eu-west-1

3. **Report Format**
   For each instance type, report:
   - Which availability zones have the instance type
   - Any zones with limited or no availability
   - Alternative instance types if primary choice is unavailable

Present findings in a clear table format with recommendations."""

TRAINING_PLAN_STATUS_PROMPT = """Provide a detailed status update on all SageMaker Training Plans:

1. **Active Plans Overview**
   - List all training plans with status: Active, Scheduled, or Pending
   - Include: plan name, instance type, instance count, duration remaining

2. **Capacity Utilization**
   For each active plan, show:
   - Total instance count vs available instance count
   - Current utilization percentage
   - In-use instance count

3. **Expiration Analysis**
   - Identify plans expiring within 30 days
   - Calculate remaining capacity hours

4. **Recommendations**
   Based on current utilization:
   - Recommend if additional capacity should be reserved
   - Suggest plans that may be over-provisioned
   - Note any capacity constraints

Format as a structured report with clear sections and actionable recommendations."""

CAPACITY_SEARCH_PROMPT = """Search for available GPU training capacity:

1. Search for training plan offerings with:
   - Instance types: ml.p5.48xlarge, ml.p4d.24xlarge, ml.trn1.32xlarge
   - Duration: 168 hours (1 week) minimum
   - Target resources: training-job

2. For each available offering, report:
   - Instance type and count
   - Duration and pricing
   - Availability zone
   - Start/end dates

3. Also check EC2 capacity availability for the equivalent instance types
   (p5.48xlarge, p4d.24xlarge, trn1.32xlarge)

Provide a summary comparing SageMaker Training Plans vs EC2 capacity options."""

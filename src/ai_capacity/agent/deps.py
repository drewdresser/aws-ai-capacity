"""Agent dependencies for AWS capacity management."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import aioboto3

if TYPE_CHECKING:
    from types_aiobotocore_ec2 import EC2Client
    from types_aiobotocore_sagemaker import SageMakerClient


@dataclass
class AgentDeps:
    """Dependencies injected into the agent at runtime.

    Attributes:
        session: Async boto3 session for AWS API calls
        region: AWS region to query (defaults to session default)
        account_id: AWS account ID for filtering resources
        include_all_regions: Whether to search across all regions
    """

    session: aioboto3.Session
    region: str = "us-east-1"
    account_id: str | None = None
    include_all_regions: bool = False
    _clients: dict = field(default_factory=dict, repr=False)

    async def get_sagemaker_client(self) -> "SageMakerClient":
        """Get async SageMaker client.

        Returns a cached client or creates a new one.
        """
        key = f"sagemaker_{self.region}"
        if key not in self._clients:
            self._clients[key] = await self.session.client(
                "sagemaker", region_name=self.region
            ).__aenter__()
        return self._clients[key]

    async def get_ec2_client(self, region: str | None = None) -> "EC2Client":
        """Get async EC2 client for specified region.

        Args:
            region: AWS region. Uses default region if not specified.

        Returns:
            EC2 client for the specified region.
        """
        target_region = region or self.region
        key = f"ec2_{target_region}"
        if key not in self._clients:
            self._clients[key] = await self.session.client(
                "ec2", region_name=target_region
            ).__aenter__()
        return self._clients[key]

    async def close(self) -> None:
        """Close all cached clients."""
        for client in self._clients.values():
            await client.__aexit__(None, None, None)
        self._clients.clear()

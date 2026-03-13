"""
EC2 worker orchestration helpers for the control-plane.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from config import (
    APP_ROLE,
    AWS_REGION,
    RESULT_TTL_HOURS,
    WORKER_AMI_ID,
    WORKER_BOOT_TIMEOUT_SECONDS,
    WORKER_IAM_INSTANCE_PROFILE,
    WORKER_INSTANCE_NAME_PREFIX,
    WORKER_INSTANCE_TYPE,
    WORKER_LAUNCH_TEMPLATE_ID,
    WORKER_LAUNCH_TEMPLATE_VERSION,
    WORKER_ORCHESTRATION_ENABLED,
    WORKER_SECURITY_GROUP_ID,
    WORKER_SUBNET_IDS,
)


@dataclass
class WorkerInstance:
    instance_id: str
    state: str
    private_ip: str | None = None
    public_ip: str | None = None
    launched_at: datetime | None = None
    expires_at: datetime | None = None


class WorkerManager:
    def __init__(self) -> None:
        self.enabled = WORKER_ORCHESTRATION_ENABLED and APP_ROLE == "control-plane" and bool(AWS_REGION)
        self.region = AWS_REGION
        self.launch_template_id = WORKER_LAUNCH_TEMPLATE_ID
        self.launch_template_version = WORKER_LAUNCH_TEMPLATE_VERSION
        self.ami_id = WORKER_AMI_ID
        self.instance_type = WORKER_INSTANCE_TYPE
        self.instance_profile = WORKER_IAM_INSTANCE_PROFILE
        self.security_group_id = WORKER_SECURITY_GROUP_ID
        self.subnet_ids = WORKER_SUBNET_IDS
        self.instance_name_prefix = WORKER_INSTANCE_NAME_PREFIX
        self._client = boto3.client("ec2", region_name=self.region) if self.enabled and self.region else None

    def launch_worker(self, task_id: str) -> WorkerInstance:
        if not self.enabled or self._client is None:
            raise RuntimeError("worker orchestration 未启用")

        response = self._client.run_instances(**self._run_instances_kwargs(task_id))
        instance = response["Instances"][0]
        worker = self._to_worker_instance(instance)
        worker.expires_at = datetime.utcnow() + timedelta(hours=RESULT_TTL_HOURS)
        return worker

    def wait_until_ready(
        self,
        instance_id: str,
        timeout_seconds: int = WORKER_BOOT_TIMEOUT_SECONDS,
    ) -> WorkerInstance:
        deadline = time.monotonic() + timeout_seconds
        last_seen: WorkerInstance | None = None
        while time.monotonic() < deadline:
            worker = self.describe_worker(instance_id)
            if worker is not None:
                last_seen = worker
                if worker.state == "running" and worker.private_ip:
                    return worker
            time.sleep(3)

        raise TimeoutError(f"等待 worker {instance_id} 就绪超时: {last_seen}")

    def describe_worker(self, instance_id: str) -> Optional[WorkerInstance]:
        if not self.enabled or self._client is None:
            return None

        try:
            response = self._client.describe_instances(InstanceIds=[instance_id])
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "InvalidInstanceID.NotFound":
                return None
            raise

        reservations = response.get("Reservations", [])
        if not reservations or not reservations[0].get("Instances"):
            return None
        return self._to_worker_instance(reservations[0]["Instances"][0])

    def terminate_worker(self, instance_id: str) -> None:
        if not self.enabled or self._client is None:
            return

        try:
            self._client.terminate_instances(InstanceIds=[instance_id])
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "InvalidInstanceID.NotFound":
                return
            raise

    def _run_instances_kwargs(self, task_id: str) -> dict:
        kwargs = {
            "MinCount": 1,
            "MaxCount": 1,
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": f"{self.instance_name_prefix}-{task_id[:12]}"},
                        {"Key": "service", "Value": "memory-engineering-worker"},
                        {"Key": "task_id", "Value": task_id},
                    ],
                }
            ],
            "MetadataOptions": {
                "HttpTokens": "required",
                "HttpEndpoint": "enabled",
            },
        }

        if self.launch_template_id:
            kwargs["LaunchTemplate"] = {
                "LaunchTemplateId": self.launch_template_id,
                "Version": self.launch_template_version,
            }
            if self.subnet_ids:
                kwargs["SubnetId"] = self._pick_subnet(task_id)
            if self.security_group_id:
                kwargs["SecurityGroupIds"] = [self.security_group_id]
            if self.instance_profile:
                profile_key = "Arn" if self.instance_profile.startswith("arn:") else "Name"
                kwargs["IamInstanceProfile"] = {profile_key: self.instance_profile}
            return kwargs

        if not all([self.ami_id, self.instance_type, self.security_group_id, self.subnet_ids]):
            raise RuntimeError("worker orchestration 缺少启动配置，请至少提供 launch template 或 AMI/实例/子网参数")

        kwargs.update(
            {
                "ImageId": self.ami_id,
                "InstanceType": self.instance_type,
                "SecurityGroupIds": [self.security_group_id],
                "SubnetId": self._pick_subnet(task_id),
            }
        )
        if self.instance_profile:
            profile_key = "Arn" if self.instance_profile.startswith("arn:") else "Name"
            kwargs["IamInstanceProfile"] = {profile_key: self.instance_profile}
        return kwargs

    def _pick_subnet(self, task_id: str) -> str:
        if not self.subnet_ids:
            raise RuntimeError("未配置 WORKER_SUBNET_IDS")
        if len(self.subnet_ids) == 1:
            return self.subnet_ids[0]
        digest = hashlib.sha256(task_id.encode("utf-8")).digest()
        index = digest[0] % len(self.subnet_ids)
        return self.subnet_ids[index]

    def _to_worker_instance(self, payload: dict) -> WorkerInstance:
        launch_time = payload.get("LaunchTime")
        state = payload.get("State", {}).get("Name", "unknown")
        return WorkerInstance(
            instance_id=payload["InstanceId"],
            state=state,
            private_ip=payload.get("PrivateIpAddress"),
            public_ip=payload.get("PublicIpAddress"),
            launched_at=launch_time.replace(tzinfo=None) if launch_time else None,
        )

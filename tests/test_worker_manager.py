from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch

import boto3
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from backend.worker_manager import WorkerInstance, WorkerManager


class WorkerManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = boto3.client(
            "ec2",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        self.manager = WorkerManager()
        self.manager.enabled = True
        self.manager._client = self.client
        self.manager.instance_name_prefix = "memory-worker"
        self.manager.launch_template_id = "lt-123456"
        self.manager.launch_template_version = "$Latest"
        self.manager.subnet_ids = ("subnet-a", "subnet-b")
        self.manager.security_group_id = "sg-123456"
        self.manager.instance_profile = "memory-worker-profile"
        self.manager.ami_id = "ami-123456"
        self.manager.instance_type = "m7i.large"

    def test_launch_template_kwargs_include_network_and_profile(self) -> None:
        kwargs = self.manager._run_instances_kwargs("task-123")
        self.assertEqual(kwargs["LaunchTemplate"]["LaunchTemplateId"], "lt-123456")
        self.assertIn(kwargs["SubnetId"], {"subnet-a", "subnet-b"})
        self.assertEqual(kwargs["SecurityGroupIds"], ["sg-123456"])
        self.assertEqual(kwargs["IamInstanceProfile"], {"Name": "memory-worker-profile"})

    def test_fallback_ami_kwargs_include_required_fields(self) -> None:
        self.manager.launch_template_id = ""
        kwargs = self.manager._run_instances_kwargs("task-456")
        self.assertEqual(kwargs["ImageId"], "ami-123456")
        self.assertEqual(kwargs["InstanceType"], "m7i.large")
        self.assertIn(kwargs["SubnetId"], {"subnet-a", "subnet-b"})
        self.assertEqual(kwargs["SecurityGroupIds"], ["sg-123456"])
        self.assertEqual(kwargs["IamInstanceProfile"], {"Name": "memory-worker-profile"})

    def test_describe_worker_returns_none_when_instance_missing(self) -> None:
        stubber = Stubber(self.client)
        stubber.add_client_error(
            "describe_instances",
            service_error_code="InvalidInstanceID.NotFound",
            service_message="missing",
            expected_params={"InstanceIds": ["i-missing"]},
        )
        with stubber:
            worker = self.manager.describe_worker("i-missing")
        self.assertIsNone(worker)

    def test_terminate_worker_ignores_missing_instance(self) -> None:
        stubber = Stubber(self.client)
        stubber.add_client_error(
            "terminate_instances",
            service_error_code="InvalidInstanceID.NotFound",
            service_message="missing",
            expected_params={"InstanceIds": ["i-missing"]},
        )
        with stubber:
            self.manager.terminate_worker("i-missing")

    def test_wait_until_ready_polls_until_running_private_ip(self) -> None:
        pending = WorkerInstance(instance_id="i-123", state="pending")
        ready = WorkerInstance(
            instance_id="i-123",
            state="running",
            private_ip="10.0.0.8",
            launched_at=datetime.utcnow(),
        )
        with patch.object(self.manager, "describe_worker", side_effect=[pending, ready]), patch(
            "backend.worker_manager.time.sleep",
            return_value=None,
        ):
            worker = self.manager.wait_until_ready("i-123", timeout_seconds=10)
        self.assertEqual(worker.state, "running")
        self.assertEqual(worker.private_ip, "10.0.0.8")


if __name__ == "__main__":
    unittest.main()

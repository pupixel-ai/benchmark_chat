from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path("/Users/vigar07/Desktop/agent_ME_0324")


class RepoMainlineLayoutTests(unittest.TestCase):
    def test_mainline_docs_and_archive_exist(self) -> None:
        self.assertTrue((ROOT / "archive" / "legacy_docs").is_dir())
        self.assertTrue((ROOT / "archive" / "legacy_code").is_dir())
        self.assertTrue((ROOT / "archive" / "legacy_notes").is_dir())
        self.assertTrue((ROOT / "docs" / "主链路策略").is_dir())
        self.assertTrue((ROOT / "docs" / "主链路策略" / "README.md").is_file())
        self.assertTrue((ROOT / "docs" / "主链路策略" / "关系标签COT总表.md").is_file())

    def test_memory_pipeline_package_is_mainline_package(self) -> None:
        self.assertTrue((ROOT / "services" / "memory_pipeline").is_dir())
        self.assertFalse((ROOT / "services" / "memory_multi_agent").is_dir())


if __name__ == "__main__":
    unittest.main()

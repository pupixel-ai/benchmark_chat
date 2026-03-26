from __future__ import annotations

import subprocess
import sys
import unittest


PROJECT_ROOT = "/Users/vigar07/Desktop/agent_ME_0324"


class ProfileEntrypointChainTests(unittest.TestCase):
    def test_profile_fields_module_imports_cleanly_from_desktop_root(self) -> None:
        command = [
            sys.executable,
            "-c",
            "import services.memory_pipeline.profile_fields as m; print(m.__file__)",
        ]
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/services/memory_pipeline/profile_fields.py", result.stdout.strip())

    def test_generate_structured_profile_uses_profile_agent_main_path(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "import inspect; "
                "import services.memory_pipeline.profile_fields as m; "
                "print(inspect.getsource(m.generate_structured_profile))"
            ),
        ]
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        source = result.stdout
        self.assertIn("ProfileAgent(", source)
        self.assertNotIn("LongTermFactsJudge()", source)
        self.assertNotIn("apply_profile_constraints", source)


if __name__ == "__main__":
    unittest.main()

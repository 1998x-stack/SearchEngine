"""Regression tests for the repository-wide topic verification script."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT_SOURCE = Path(__file__).with_name("verify_all.sh").read_text(encoding="utf-8")


class VerifyAllScriptTests(unittest.TestCase):
    def make_repository(
        self,
        root: Path,
        *,
        topic_count: int = 94,
        catalog_count: int = 94,
        failing_topic: bool = False,
        wrong_last_name: bool = False,
    ) -> Path:
        scripts_dir = root / "scripts"
        catalog_dir = root / "docs" / "superpowers"
        topics_dir = root / "topics"
        scripts_dir.mkdir(parents=True)
        catalog_dir.mkdir(parents=True)
        topics_dir.mkdir(parents=True)
        script = scripts_dir / "verify_all.sh"
        script.write_text(SCRIPT_SOURCE, encoding="utf-8")
        catalog_rows = [
            "# Topic catalog",
            "| # | File | Template |",
            "|---|---|---|",
        ]
        for index in range(1, catalog_count + 1):
            name = f"topic_{index:02d}"
            if failing_topic and index == topic_count:
                name += "_fail"
            catalog_rows.append(f"| {index} | `./topics/{name}.py` | T3 |")
        (catalog_dir / "catalog.md").write_text(
            "\n".join(catalog_rows) + "\n", encoding="utf-8"
        )
        for index in range(1, topic_count + 1):
            name = f"topic_{index:02d}"
            if wrong_last_name and index == topic_count:
                name += "_unexpected"
            if failing_topic and index == topic_count:
                name += "_fail"
            (topics_dir / f"{name}.py").write_text("print('ok')\n", encoding="utf-8")
        return script

    def run_script(
        self,
        script: Path,
        fakebin: Path,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PATH"] = f"{fakebin}{os.pathsep}{env.get('PATH', '')}"
        return subprocess.run(
            ["bash", str(script)],
            cwd=script.parent.parent,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def make_fake_python(self, fakebin: Path) -> None:
        fakebin.mkdir(parents=True, exist_ok=True)
        executable = fakebin / "python3"
        executable.write_text(
            "#!/bin/sh\ncase \"$1\" in *fail.py) exit 7 ;; *) exit 0 ;; esac\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)

    def test_all_catalogued_topics_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = self.make_repository(root)
            fakebin = root / "fakebin"
            self.make_fake_python(fakebin)

            result = self.run_script(script, fakebin)

            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertIn("ALL OK (94/94)", result.stdout)

    def test_missing_topic_file_fails_before_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = self.make_repository(root, topic_count=93)
            fakebin = root / "fakebin"
            self.make_fake_python(fakebin)

            result = self.run_script(script, fakebin)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("catalog: 94, found: 93", result.stderr)
            self.assertNotIn("ALL OK", result.stdout)

    def test_same_count_with_wrong_file_set_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = self.make_repository(root, wrong_last_name=True)
            fakebin = root / "fakebin"
            self.make_fake_python(fakebin)

            result = self.run_script(script, fakebin)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("topic script set mismatch", result.stderr)
            self.assertNotIn("ALL OK", result.stdout)

    def test_find_failure_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = self.make_repository(root)
            fakebin = root / "fakebin"
            self.make_fake_python(fakebin)
            find = fakebin / "find"
            find.write_text("#!/bin/sh\necho 'simulated find failure' >&2\nexit 2\n", encoding="utf-8")
            find.chmod(0o755)

            result = self.run_script(script, fakebin)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("failed to enumerate", result.stderr)
            self.assertNotIn("ALL OK", result.stdout)

    def test_topic_execution_failure_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = self.make_repository(root, failing_topic=True)
            fakebin = root / "fakebin"
            self.make_fake_python(fakebin)

            result = self.run_script(script, fakebin)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FAIL: ./topics/topic_94_fail.py", result.stdout)
            self.assertIn("SOME FAILED", result.stdout)
            self.assertNotIn("ALL OK", result.stdout)


if __name__ == "__main__":
    unittest.main()

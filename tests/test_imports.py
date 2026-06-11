"""Smoke test: every module under crackle/ must be importable.

Guards against packaging regressions like the .gitignore ``data/`` rule
that silently dropped the crackle/data package from the public archive
(found 2026-06: baselines/train/eval failed to import from a fresh clone).
"""
from __future__ import annotations

import importlib
import pkgutil
import unittest

import crackle


class TestAllModulesImport(unittest.TestCase):
    def test_walk_and_import(self) -> None:
        failures: list[str] = []
        for info in pkgutil.walk_packages(crackle.__path__, prefix="crackle."):
            try:
                importlib.import_module(info.name)
            except Exception as exc:  # noqa: BLE001 - report all failures at once
                failures.append(f"{info.name}: {type(exc).__name__}: {exc}")
        self.assertEqual(
            failures, [], "modules failed to import:\n" + "\n".join(failures)
        )

    def test_data_package_present(self) -> None:
        """The exact package the gitignore bug used to drop."""
        for name in (
            "crackle.data.common",
            "crackle.data.features",
            "crackle.data.event_catalog",
            "crackle.data.riskset",
        ):
            importlib.import_module(name)


if __name__ == "__main__":
    unittest.main()

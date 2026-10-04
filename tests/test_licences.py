"""Licence guard: every pretrained weight and pinned package must be screened in THIRD_PARTY_NOTICES.md.

The prize entry is a public repository, so a component whose licence limits commercial use or redistribution
disqualifies it. This test fails when code starts loading a torchvision weight, or requirements.txt pins a
package, that the notices file does not cover — a new backbone cannot enter the pipeline unscreened.
Needs no competition data.  python -m unittest discover tests
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"


def weights_referenced():
    found = set()
    for sub in ("filseg", "scripts", "notebooks"):
        for path in (ROOT / sub).rglob("*"):
            if path.suffix in (".py", ".ipynb"):
                found |= set(
                    re.findall(r"\b([A-Za-z0-9_]+_Weights)\b", path.read_text())
                )
    return found


def packages_pinned():
    names = []
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.append(re.split(r"[=<>!~\[ ]", line, 1)[0])
    return names


class TestLicences(unittest.TestCase):
    def test_notices_file_exists(self):
        self.assertTrue(NOTICES.is_file(), "THIRD_PARTY_NOTICES.md is missing")

    def test_every_pretrained_weight_is_screened(self):
        text = NOTICES.read_text()
        weights = weights_referenced()
        self.assertTrue(
            weights, "no torchvision weights found; the scan itself is broken"
        )
        missing = sorted(w for w in weights if w not in text)
        self.assertEqual(
            missing, [], f"pretrained weights loaded but not screened: {missing}"
        )

    def test_every_pinned_package_is_listed(self):
        text = NOTICES.read_text().lower()
        missing = [p for p in packages_pinned() if p.lower() not in text]
        self.assertEqual(
            missing, [], f"requirements.txt packages absent from the notices: {missing}"
        )


if __name__ == "__main__":
    unittest.main()

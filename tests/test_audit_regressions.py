"""Regression tests for consistency and emergency lock guards.

Merged from the 68bbb922 line. The recheck pin is adapted to the graded
recheck design (the failure names its numbers on the check detail, where
the UI renders it); the re-lock pin is verbatim.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.engine import PredictionEngine, _recheck_prediction


class AuditRegressions(unittest.TestCase):
    def test_recheck_explains_failed_confidence(self):
        result = _recheck_prediction(
            direction=1, conf=.285, floor=.35, feed_age_s=1,
            voters={"a": 1, "b": 1, "c": 1}, levels_ok=True,
            ratio=2, lock_dir=1, tfs_computed=4)
        self.assertIn("confidence above floor", result["failed"])
        detail = next(c["detail"] for c in result["checks"]
                      if c["name"] == "confidence above floor")
        self.assertIn("confidence below floor (0.285 ≤ 0.350)", detail)

    def test_active_relock_requires_age_and_strict_atr_move(self):
        now = 10000.0
        lock = {"price": 100.0, "locked_at": now - 899}
        engine = SimpleNamespace(
            news=SimpleNamespace(emergency_history=[], last_emergency={}),
            feed=SimpleNamespace(prices=lambda: {"BTC": 104.0}),
            locks={("BTC", "tf1"): lock},
            build_ctx=lambda *a: None, _atr=lambda ctx: 1.0)
        with patch("app.engine.time.time", return_value=now):
            young = PredictionEngine.emergency_relock(engine, "BTC", automatic=True)
            self.assertFalse(young["ok"])
            self.assertTrue(young["cooldown"])
            lock["locked_at"] = now - 900
            engine.feed.prices = lambda: {"BTC": 103.0}
            equal = PredictionEngine.emergency_relock(engine, "BTC", automatic=True)
            self.assertFalse(equal["ok"])
            self.assertEqual(equal["required_move_atr"], 3.0)
            self.assertEqual(lock["price"], 100.0)


if __name__ == "__main__":
    unittest.main()

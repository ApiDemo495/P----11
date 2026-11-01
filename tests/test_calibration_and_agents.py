"""Accuracy-driven weightage (emotion/physics) + agent auto-best."""
from app.emotion import EmotionSystem
from app.physics import PhysicsSystem, UFP_ALPHA_IDS


def test_emotion_cold_start_learns_fast():
    """A fresh system must differentiate channels within ~50 updates,
    not stay frozen at uniform (the old 1.5% eta did exactly that).
    Channels 1..3 agree with the realized move; 4..21 disagree."""
    e = EmotionSystem()
    w0 = list(e._weights("BTC", "tf1"))
    emos = {i: (0.8 if i <= 3 else -0.8) for i in range(1, 22)}
    dirs = {i: 1 for i in range(1, 22)}
    for _ in range(50):
        e.reward_update("BTC", "tf1", emos, dirs, 0.0, move=50.0, scale=30.0)
    w1 = e._weights("BTC", "tf1")
    assert max(w1) > max(w0) * 1.5          # informative channel grew fast
    assert min(w1) < min(w0)                 # misleading channel shrank
    assert abs(sum(w1) - 1.0) < 1e-6


def test_emotion_clamp_widened_and_accuracy_tracked():
    e = EmotionSystem()
    emos = {i: (0.9 if i <= 3 else -0.9) for i in range(1, 22)}
    dirs = {i: 1 for i in range(1, 22)}
    for _ in range(300):
        e.reward_update("BTC", "tf1", emos, dirs, 0.0, move=100.0, scale=10.0)
    w = e._weights("BTC", "tf1")
    assert max(w) <= 0.32                    # clamp 0.30 + renormalize slack
    assert max(w) > 0.10                    # dominant channel actually dominates
    acc = e._accuracy("BTC", "tf1")
    assert acc[0] > 0.9                     # agreeing channel learned
    assert acc[20] < 0.1                    # misled channel knows it
    assert "BTC:tf1" in e.accuracy()


def test_physics_alpha_calibrates_to_informative_term():
    ph = PhysicsSystem.__new__(PhysicsSystem)   # skip heavy ctx init
    ph.alpha, ph.acc, ph.n_updates = {}, {}, {}
    # terms 1..2 sign-align with the move, the rest oppose
    vals = {pid: (0.6 if pid <= 2 else -0.6) for pid in UFP_ALPHA_IDS}
    for _ in range(200):
        ph.reward_update("BTC", "tf1", vals, move=40.0, scale=20.0)
    w = ph._weights("BTC", "tf1")
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert max(w.values()) <= 0.33            # clamp 0.30 + renormalize slack
    assert max(w.values()) > 1.0 / len(UFP_ALPHA_IDS) * 2.0
    a = ph.accuracy()
    assert "BTC:tf1" in a


def test_agents_catalog_import_auto_best():
    import tempfile, os
    from app import agents as ag
    old = ag.SETTINGS_FILE
    with tempfile.TemporaryDirectory() as td:
        ag.SETTINGS_FILE = os.path.join(td, "s.json")
        try:
            am = ag.AgentManager()
            cat = am.catalog_status()
            assert len(cat["packs"]) == 4
            assert cat["auto_best"] is False
            out = am.import_pack("openai", "sk-demo")
            assert out["ok"] and out["slot"] == "backup2"
            am.active_slot = "backup2"
            for _ in range(10):
                am.record_outcome(True)
            am.active_slot = "gemini"
            for _ in range(10):
                am.record_outcome(False)
            assert am.best_slot() == "backup2"
            r = am.set_auto(True)
            assert r["ok"] and r["best"] == "backup2"
            st = am.status()
            assert st["catalog"]["auto_best"] is True
            assert "backup2" in st["catalog"]["scores"]
            pk = am.export_pack("openai")
            assert pk["key"] == "" and pk["format"] == "predict-agent-pack-v1"
        finally:
            ag.SETTINGS_FILE = old


def test_engine_exposes_accuracy_in_snapshot():
    from tests.test_engine import make_engine, _boot, run
    eng, _f, _e = make_engine()
    run(_boot(eng))
    for _ in range(3):
        eng.compute_asset("BTC", record=True, display=False)
    snap = eng.build_snapshot()
    sym = "BTC"
    emo = snap["systems"][f"{sym}|tf1"]["emotion"]
    assert "accuracy" in emo and len(emo["accuracy"]) == 21
    assert "n_updates" in emo
    phys = snap["systems"][f"{sym}|tf1"]["physics"]
    assert "alpha_accuracy" in phys and len(phys["alpha_accuracy"]) == 21
    assert "n_updates" in phys

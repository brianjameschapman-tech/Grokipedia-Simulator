from main import GrokipediaSimulator
def test_validator():
    sim = GrokipediaSimulator()
    out = sim.neurosymbolic_hybrid_validator("test", "Add accurate info")
    assert "valid" in out

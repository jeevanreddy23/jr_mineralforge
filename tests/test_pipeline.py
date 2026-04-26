from mineralforge.edge_simulator import run_demo_assessment
from mineralforge.tarp import map_risk_to_tarp


def test_demo_assessment_returns_decision_payload():
    assessment = run_demo_assessment(zone="Stope 3", stress_multiplier=2.8)
    assert assessment["zone"] == "Stope 3"
    assert assessment["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert len(assessment["drivers"]) == 3
    assert "action" in assessment["tarp"]


def test_high_risk_tarp_evacuation_action():
    action = map_risk_to_tarp("HIGH", "Stope 3")
    assert "Evacuate Stope 3" in action.action
    assert action.risk_level == "HIGH"

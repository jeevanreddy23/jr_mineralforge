from mineralforge.geotech.recommendations import tarp_recommendation


def test_high_risk_tarp_recommendation_mentions_stop_and_inspect():
    recommendation = tarp_recommendation("High")
    assert "Stop" in recommendation
    assert "inspect" in recommendation.lower()

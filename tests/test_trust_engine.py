import os
import tempfile
import pytest

from trustengine.trust_engine import TrustEngine
from trustengine.trust_history import TrustHistory
from trustengine.quarantine import QuarantineManager, TrustZone


class TestTrustEngineSuite:
    """Test suite for continuous dynamic trust calculation and quarantine."""

    def test_trust_calculation_honest_vs_malicious(self):
        engine = TrustEngine()
        pid_scores = {1: 0.1, 2: 0.12, 3: 0.09, 4: 8.5}  # Client 4 is anomalous

        # Calculate trust for all
        res_honest = engine.calculate_trust(1, 0.1, list(pid_scores.values()))
        res_malicious = engine.calculate_trust(4, 8.5, list(pid_scores.values()))

        assert res_honest["trust"] > 0.8
        assert res_malicious["trust"] < 0.5
        assert res_honest["relative_anomaly"] == 1.0
        assert res_malicious["relative_anomaly"] > 2.0

    def test_calculate_all_trust_batch(self):
        engine = TrustEngine()
        pid_scores = {1: 0.1, 2: 0.12, 3: 0.09, 4: 9.0}
        trust_map = engine.calculate_all_trust(pid_scores)

        assert len(trust_map) == 4
        assert trust_map[1] > trust_map[4]

    def test_persistence_penalty_accumulation(self):
        engine = TrustEngine()
        # Client 4 repeatedly submits anomalies across 4 rounds
        for _ in range(4):
            engine.calculate_trust(4, 10.0, [0.1, 0.1, 0.1, 10.0])

        history = engine.history.get_trust_history(4)
        assert len(history) == 4
        # Trust should steadily decrease
        assert history[-1] < history[0]

    def test_trust_zone_classification(self):
        engine = TrustEngine()
        assert engine.get_trust_zone(0.90) == "NORMAL"
        assert engine.get_trust_zone(0.55) == "MEDIUM"
        assert engine.get_trust_zone(0.20) == "LOW"

    def test_quarantine_probation_and_graduation(self):
        qm = QuarantineManager(quarantine_threshold=0.25, required_clean_rounds=2)

        # 1. Initial low trust triggers quarantine
        zone1 = qm.evaluate_client(client_id=4, trust_score=0.15, relative_anomaly=5.0)
        assert zone1 == TrustZone.QUARANTINE
        assert qm.is_quarantined(4)
        assert not qm.is_eligible_for_aggregation(4)

        # 2. Round 1 of clean probation (reformed behavior)
        zone2 = qm.evaluate_client(client_id=4, trust_score=0.20, relative_anomaly=1.0)
        assert zone2 == TrustZone.QUARANTINE  # Needs 2 clean rounds

        # 3. Round 2 of clean probation -> Graduates back to LOW
        zone3 = qm.evaluate_client(client_id=4, trust_score=0.30, relative_anomaly=1.0)
        assert zone3 == TrustZone.LOW
        assert not qm.is_quarantined(4)
        assert qm.is_eligible_for_aggregation(4)

    def test_quarantine_eviction_policy(self):
        qm = QuarantineManager(quarantine_threshold=0.25, max_quarantine_rounds=3, enable_eviction=True)

        for _ in range(4):
            qm.evaluate_client(client_id=5, trust_score=0.10, relative_anomaly=8.0)

        assert qm.is_evicted(5)
        assert not qm.is_eligible_for_aggregation(5)

    def test_trust_history_json_serialization(self):
        history = TrustHistory()
        history.update(client_id=1, trust_score=0.95, anomaly_score=1.0)
        history.update(client_id=2, trust_score=0.30, anomaly_score=4.5)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            history.save_to_json(temp_path)

            loaded_history = TrustHistory()
            loaded_history.load_from_json(temp_path)

            assert loaded_history.get_latest_trust(1) == 0.95
            assert loaded_history.get_latest_trust(2) == 0.30
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

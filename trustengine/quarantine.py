from enum import Enum
from typing import Dict, List, Set, Optional


class TrustZone(str, Enum):
    """Trust Zone categorization tiers."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    QUARANTINE = "QUARANTINE"
    EVICTED = "EVICTED"


class QuarantineManager:
    """
    Manages Quarantine, Probation, and Trust Recovery for suspicious edge clients.
    
    Workflow:
      1. If trust falls below quarantine_threshold (e.g. 0.20), client is quarantined.
      2. In quarantine, client is excluded from model aggregation.
      3. Client is given probation opportunities: if client behaves well for
         required_clean_rounds (e.g. 2 consecutive rounds), client graduates back to LOW zone.
      4. If a client remains quarantined for > max_quarantine_rounds, client can be permanently EVICTED.
    """

    def __init__(
            self,
            quarantine_threshold: float = 0.20,
            required_clean_rounds: int = 2,
            max_quarantine_rounds: int = 5,
            enable_eviction: bool = False
    ):
        self.quarantine_threshold = quarantine_threshold
        self.required_clean_rounds = required_clean_rounds
        self.max_quarantine_rounds = max_quarantine_rounds
        self.enable_eviction = enable_eviction

        # State tracking
        self.quarantined_clients: Set[int] = set()
        self.evicted_clients: Set[int] = set()
        self.probation_clean_counts: Dict[int, int] = {}
        self.rounds_in_quarantine: Dict[int, int] = {}

    def is_quarantined(self, client_id: int) -> bool:
        return client_id in self.quarantined_clients

    def is_evicted(self, client_id: int) -> bool:
        return client_id in self.evicted_clients

    def is_eligible_for_aggregation(self, client_id: int) -> bool:
        """Returns False if client is currently in Quarantine or Evicted."""
        return client_id not in self.quarantined_clients and client_id not in self.evicted_clients

    def evaluate_client(
            self,
            client_id: int,
            trust_score: float,
            relative_anomaly: float
    ) -> TrustZone:
        """
        Evaluate client trust and anomaly to update quarantine/probation state.
        """
        if self.is_evicted(client_id):
            return TrustZone.EVICTED

        # Currently in quarantine / probation
        if self.is_quarantined(client_id):
            self.rounds_in_quarantine[client_id] = self.rounds_in_quarantine.get(client_id, 0) + 1

            # Check for permanent eviction
            if self.enable_eviction and self.rounds_in_quarantine[client_id] >= self.max_quarantine_rounds:
                self.quarantined_clients.remove(client_id)
                self.evicted_clients.add(client_id)
                return TrustZone.EVICTED

            # Check if this round was clean (low relative anomaly)
            is_clean_round = relative_anomaly <= 2.0
            if is_clean_round:
                self.probation_clean_counts[client_id] = self.probation_clean_counts.get(client_id, 0) + 1
            else:
                self.probation_clean_counts[client_id] = 0

            # Check for probation graduation
            if self.probation_clean_counts[client_id] >= self.required_clean_rounds:
                self.quarantined_clients.remove(client_id)
                self.probation_clean_counts[client_id] = 0
                self.rounds_in_quarantine[client_id] = 0
                return TrustZone.LOW

            return TrustZone.QUARANTINE

        # Not currently quarantined: check if trust dropped below threshold
        if trust_score < self.quarantine_threshold:
            self.quarantined_clients.add(client_id)
            self.probation_clean_counts[client_id] = 0
            self.rounds_in_quarantine[client_id] = 1
            return TrustZone.QUARANTINE

        # Standard zone categorization
        if trust_score >= 0.75:
            return TrustZone.HIGH
        elif trust_score >= 0.40:
            return TrustZone.MEDIUM
        else:
            return TrustZone.LOW

    def get_active_clients_for_aggregation(self, client_ids: List[int]) -> List[int]:
        """Filter out quarantined and evicted clients."""
        return [cid for cid in client_ids if self.is_eligible_for_aggregation(cid)]

    def reset(self):
        self.quarantined_clients.clear()
        self.evicted_clients.clear()
        self.probation_clean_counts.clear()
        self.rounds_in_quarantine.clear()

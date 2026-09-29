from typing import Dict, List
import torch
from attacks.base_attack import BaseAttack


class SybilCollusionCoordinator:
    """
    Coordinates malicious updates across a group of colluding Sybil clients.
    
    In a Sybil attack, colluding clients coordinate their poisoned updates
    (e.g., repeating the exact same adversarial vector or crafting an amplified
    average direction) to increase Byzantine voting power and evade median filters.
    """

    def __init__(self, sybil_client_ids: List[int], scale_factor: float = 5.0):
        self.sybil_client_ids = set(sybil_client_ids)
        self.scale_factor = scale_factor

    def coordinate_updates(
            self,
            client_updates: Dict[int, Dict[str, torch.Tensor]],
            leader_id: int = None
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Replace all Sybil client updates with the coordinated adversarial update of the leader.
        """
        if not self.sybil_client_ids:
            return client_updates

        if leader_id is None or leader_id not in self.sybil_client_ids:
            leader_id = sorted(list(self.sybil_client_ids))[0]

        if leader_id not in client_updates:
            return client_updates

        leader_update = client_updates[leader_id]
        coordinated_vector = {}
        for name, tensor in leader_update.items():
            if torch.is_floating_point(tensor):
                coordinated_vector[name] = tensor * self.scale_factor
            else:
                coordinated_vector[name] = tensor.clone()

        result = dict(client_updates)
        for s_id in self.sybil_client_ids:
            if s_id in result:
                result[s_id] = {k: v.clone() for k, v in coordinated_vector.items()}

        return result

import torch


class GradientScaleAttack:
    """
    Gradient / model-update poisoning attack.

    After the malicious client performs normal local training,
    the update delta is scaled by a large factor.

    This pushes the global model toward the attacker's
    local gradient direction with disproportionate influence.

    Usage:
        attack = GradientScaleAttack(scale_factor=10.0)
        poisoned_update = attack.apply(client_update)
    """

    def __init__(
            self,
            scale_factor=10.0
    ):

        self.scale_factor = scale_factor

    def apply(self, update):
        """
        Scale every parameter in the update by
        scale_factor.

        Parameters:
            update: dict of parameter_name -> tensor delta

        Returns:
            Scaled update dict.
        """

        poisoned_update = {}

        for name, delta in update.items():

            poisoned_update[name] = (
                    delta * self.scale_factor
            )

        return poisoned_update


class SignFlipAttack:
    """
    Sign-flip attack.

    Reverses the direction of the client update so it
    moves the global model away from convergence.

    This is harder to detect than simple scaling because
    the update magnitude can remain similar to honest
    clients.

    Usage:
        attack = SignFlipAttack()
        poisoned_update = attack.apply(client_update)
    """

    def __init__(
            self,
            scale_factor=1.0
    ):

        self.scale_factor = scale_factor

    def apply(self, update):
        """
        Negate every parameter in the update.

        Parameters:
            update: dict of parameter_name -> tensor delta

        Returns:
            Negated (and optionally scaled) update dict.
        """

        poisoned_update = {}

        for name, delta in update.items():

            poisoned_update[name] = (
                    -1.0
                    * self.scale_factor
                    * delta
            )

        return poisoned_update

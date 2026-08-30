import torch

from model.model import MNISTModel

from detection.pid_detector import PIDDetector
from trustengine.trust_engine import TrustEngine
from risk.round_risk import RoundRisk
from aggregation.adaptive_aggregator import AdaptiveAggregator


class Server:

    def __init__(self):

        self.global_model = MNISTModel()

        # --------------------------------------------------
        # Malicious / anomaly detection
        # --------------------------------------------------

        self.detector = PIDDetector()

        # --------------------------------------------------
        # Dynamic client trust
        # --------------------------------------------------

        self.trust_engine = TrustEngine()

        # --------------------------------------------------
        # Round risk assessment
        # --------------------------------------------------

        self.round_risk = RoundRisk()

        # --------------------------------------------------
        # Adaptive aggregation
        # --------------------------------------------------

        self.adaptive_aggregator = AdaptiveAggregator()


    def aggregate(
            self,
            client_parameters,
            client_sizes,
            trust_scores,
            risk_level
    ):

        # --------------------------------------------------
        # Adaptive aggregation based on round risk
        # --------------------------------------------------

        new_parameters, selected_aggregator = (
            self.adaptive_aggregator.aggregate(
                client_parameters,
                client_sizes,
                trust_scores,
                risk_level
            )
        )

        # --------------------------------------------------
        # Display selected aggregation strategy
        # --------------------------------------------------

        print()
        print("Adaptive Aggregation")
        print("------------------------------")

        print(
            f"Round Risk: {risk_level}"
        )

        print(
            f"Selected Aggregator: "
            f"{selected_aggregator}"
        )

        return new_parameters, selected_aggregator


    def evaluate(
            self,
            test_dataset
    ):

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False
        )

        self.global_model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                outputs = self.global_model(
                    images
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                total += labels.size(0)

                correct += (
                        predictions == labels
                ).sum().item()

        accuracy = correct / total

        return accuracy
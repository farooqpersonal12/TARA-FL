"""
TARA-FL Visualization, Dashboard & Reporting Package.
"""

from .plots import (
    plot_accuracy_comparison,
    plot_loss_comparison,
    plot_trust_trajectories,
    plot_risk_and_aggregator,
    plot_method_bar_comparison
)
from .report_generator import (
    ReportGenerator,
    generate_experiment_report
)
from .dashboard import (
    DashboardRequestHandler,
    start_dashboard_server
)

__all__ = [
    "plot_accuracy_comparison",
    "plot_loss_comparison",
    "plot_trust_trajectories",
    "plot_risk_and_aggregator",
    "plot_method_bar_comparison",
    "ReportGenerator",
    "generate_experiment_report",
    "DashboardRequestHandler",
    "start_dashboard_server"
]

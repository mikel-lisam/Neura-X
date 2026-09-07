# ==========================================================
# Neura-X: Intelligence Without Limits.
# Copyright (c) 2026 Edusei Mikel Lisamba. All Rights Reserved.
# Dream Dashboard
# ==========================================================

"""Dream Dashboard: visualizes the Wake-Sleep cycle."""


class DreamDashboard:
    """Visual dashboard for the Circadian Wake-Sleep cycle."""

    def __init__(self, model=None):
        self.model = model
        self._sleep_log = []

    def open(self):
        """Open the Dream Dashboard."""
        print("🌙 Neura-X Dream Dashboard")
        print("─" * 40)
        print("Wake Phase:  Logging surprise vectors")
        print("Sleep Phase: Consolidating knowledge")
        print("─" * 40)

    @classmethod
    def from_model(cls, model):
        return cls(model=model)
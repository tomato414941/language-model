import unittest
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from scripts.modal_costs import make_report


@dataclass
class BillingSummary:
    start: datetime
    end: datetime
    metered_cost: Decimal
    metered_cost_breakdown: dict
    adjustments: dict
    billed_cost: Decimal


class ModalCostTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2026, 10, 1, tzinfo=UTC)
        self.now = datetime(2026, 10, 12, tzinfo=UTC)
        self.summary = BillingSummary(
            start=self.start,
            end=datetime(2026, 11, 1, tzinfo=UTC),
            metered_cost=Decimal("11.61"),
            metered_cost_breakdown={"Volumes": Decimal("11.61")},
            adjustments={"Free Storage": Decimal("-11.61")},
            billed_cost=Decimal(0),
        )
        self.rates = {
            "cpu_hour_cost": Decimal("0.04730"),
            "mem_gib_hour_cost": Decimal("0.00800"),
        }

    def report(self, previous=None):
        return make_report(
            "test-workspace", self.summary, self.rates, self.now, previous
        )

    def test_reports_actual_billed_cost_and_storage_allowances(self):
        """Reports the billed total alongside gross usage and allowance adjustments."""
        billing = self.report()["billing"]
        self.assertEqual(Decimal(billing["billed_cost"]), Decimal(0))
        self.assertEqual(Decimal(billing["metered_cost"]), Decimal("11.61"))
        self.assertEqual(
            Decimal(billing["adjustments"]["Free Storage"]), Decimal("-11.61")
        )

    def test_calculates_monthly_compute_scenarios_from_current_rates(self):
        """Calculates compute scenarios using exact CPU and memory prices."""
        scenarios = self.report()["compute_scenarios_before_adjustments"]
        self.assertEqual(Decimal(scenarios["hourly_usd"]), Decimal("0.06330"))
        self.assertEqual(
            Decimal(scenarios["monthly_usd_by_daily_billed_hours"]["1"]),
            Decimal("1.899"),
        )
        self.assertEqual(
            Decimal(scenarios["monthly_usd_by_daily_billed_hours"]["24"]),
            Decimal("45.576"),
        )

    def test_reports_increases_in_billed_cost_within_a_billing_cycle(self):
        """Reports a precise increase from the previous successful check."""
        previous = self.report()
        self.summary.billed_cost = Decimal("0.12345678")
        report = self.report(previous)
        self.assertEqual(report["billed_cost_change_usd"], "0.12345678")
        self.assertTrue(report["billed_cost_increased"])

    def test_starts_a_fresh_comparison_when_the_billing_cycle_changes(self):
        """Starts comparison from the first successful check of the new month."""
        previous = self.report()
        self.summary.start = datetime(2026, 11, 1, tzinfo=UTC)
        report = self.report(previous)
        self.assertIsNone(report["billed_cost_change_usd"])


if __name__ == "__main__":
    unittest.main()

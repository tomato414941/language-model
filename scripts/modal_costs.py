"""Collect actual Modal billing totals without starting inference containers."""

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path


def json_value(value):
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError("Unsupported billing value: " + type(value).__name__)


def make_report(workspace, summary, rates, checked_at, previous=None):
    hourly = (
        rates["gpu_hour_cost_t4"]
        + rates["cpu_hour_cost"]
        + 2 * rates["mem_gib_hour_cost"]
    )
    billing = json.loads(json.dumps(asdict(summary), default=json_value))
    delta = None
    if (
        previous
        and previous.get("workspace") == workspace
        and previous.get("billing", {}).get("start") == billing["start"]
    ):
        delta = summary.billed_cost - Decimal(previous["billing"]["billed_cost"])
    return {
        "checked_at": checked_at.isoformat(),
        "workspace": workspace,
        "scope": "Entire Modal workspace, including credits and storage adjustments",
        "billing": billing,
        "billed_cost_change_usd": str(delta) if delta is not None else None,
        "billed_cost_increased": delta > 0 if delta is not None else False,
        "compute_scenarios_before_adjustments": {
            "gpu": "T4",
            "physical_cpu_cores": 1,
            "memory_gib": 2,
            "days": 30,
            "hourly_usd": str(hourly),
            "monthly_usd_by_daily_billed_hours": {
                str(hours): str(hourly * hours * 30) for hours in (1, 8, 24)
            },
            "note": (
                "Includes billed startup and idle time. Excludes image builds, "
                "memory above 2 GiB, storage, networking, taxes and credits. "
                "These scenarios are not forecasts of measured traffic."
            ),
        },
    }


def save_report(report, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=output.parent, prefix=".modal-costs-", delete=False
        ) as source:
            temporary = Path(source.name)
            json.dump(report, source, indent=2, ensure_ascii=False, allow_nan=False)
            source.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/modal-costs.json"))
    args = parser.parse_args(argv)

    import modal

    workspace = modal.Workspace.from_context()
    workspace.hydrate()
    if workspace.name != args.workspace:
        raise ValueError("The active Modal credentials belong to another workspace")
    previous = json.loads(args.output.read_text()) if args.output.is_file() else None
    report = make_report(
        workspace.name,
        workspace.billing.summary(),
        workspace.billing.rates(),
        datetime.now(UTC),
        previous,
    )
    save_report(report, args.output)
    print(
        json.dumps(
            {
                "checked_at": report["checked_at"],
                "workspace": report["workspace"],
                "billed_cost_usd": report["billing"]["billed_cost"],
                "billed_cost_change_usd": report["billed_cost_change_usd"],
                "billed_cost_increased": report["billed_cost_increased"],
                "report": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()

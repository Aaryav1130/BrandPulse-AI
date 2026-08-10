"""BrandPulse AI — CLI Entry Point.

Run the full feedback processing pipeline from the command line.

Usage:
    python main.py                    # Process all reviews
    python main.py --brand glownest   # Process one brand only
    python main.py --limit 10         # Process first 10 reviews
    python main.py --brand urbanmane --limit 5
"""

from __future__ import annotations

import os
import sys

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.alerts.notifier import AlertNotifier, AnomalyDetector
from src.graph.pipeline import BrandPulsePipeline
from src.ingestion.loader import FeedbackLoader
from src.utils.helpers import load_env, save_results, setup_logging

console = Console()
logger = setup_logging()


def print_banner() -> None:
    """Print the BrandPulse AI banner."""
    banner = """
[bold cyan]╔══════════════════════════════════════════════════════╗
║                                                      ║
║   ██████╗ ██████╗  █████╗ ███╗   ██╗██████╗          ║
║   ██╔══██╗██╔══██╗██╔══██╗████╗  ██║██╔══██╗         ║
║   ██████╔╝██████╔╝███████║██╔██╗ ██║██║  ██║         ║
║   ██╔══██╗██╔══██╗██╔══██║██║╚██╗██║██║  ██║         ║
║   ██████╔╝██║  ██║██║  ██║██║ ╚████║██████╔╝         ║
║   ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝          ║
║              [bold yellow]P U L S E   A I[/bold yellow]                          ║
║                                                      ║
║   [dim]Autonomous Customer Feedback Intelligence[/dim]         ║
║   [dim]for Think9's Multi-Brand Portfolio[/dim]                 ║
║                                                      ║
╚══════════════════════════════════════════════════════╝[/bold cyan]
"""
    console.print(banner)


def print_summary_table(results: list[dict]) -> None:
    """Print a summary table of processing results."""
    table = Table(
        title="📊 Processing Summary",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("ID", style="dim")
    table.add_column("Brand", style="cyan")
    table.add_column("Product", max_width=25)
    table.add_column("Category", style="yellow")
    table.add_column("Urgency", justify="center")
    table.add_column("Sentiment", justify="center")
    table.add_column("Action", justify="center")

    for result in results:
        if not result.get("processed"):
            continue

        record = result.get("record", {})
        classification = result.get("classification", {})
        sentiment = result.get("sentiment", {})
        orchestrator = result.get("orchestrator", {})

        # Urgency indicator
        urgency = classification.get("urgency", 0)
        urgency_display = {
            1: "[green]●[/green]",
            2: "[cyan]●[/cyan]",
            3: "[yellow]●[/yellow]",
            4: "[red]●[/red]",
            5: "[bold red]⬤[/bold red]",
        }.get(urgency, "?")

        # Sentiment indicator
        score = sentiment.get("score", 0)
        if score >= 0.3:
            sent_display = f"[green]▲ {score:+.2f}[/green]"
        elif score <= -0.3:
            sent_display = f"[red]▼ {score:+.2f}[/red]"
        else:
            sent_display = f"[yellow]● {score:+.2f}[/yellow]"

        # Action indicator
        action = "⚡" if orchestrator.get("action_required") else "—"

        table.add_row(
            record.get("id", ""),
            record.get("brand", ""),
            record.get("product", "")[:25],
            classification.get("primary_category", "").replace("_", " "),
            urgency_display,
            sent_display,
            action,
        )

    console.print(table)


def print_alerts_summary(alerts: list[dict]) -> None:
    """Print a summary panel of generated alerts."""
    if not alerts:
        console.print(
            Panel(
                "[green]No critical alerts generated.[/green]",
                title="🔔 Alerts",
                border_style="green",
            )
        )
        return

    alert_lines = []
    for alert in alerts:
        severity = alert.get("severity", "info").upper()
        emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(
            severity, "⚪"
        )
        brand = alert.get("brand", "")
        title = alert.get("title", "")
        alert_lines.append(f"{emoji} [{severity}] {brand}: {title}")

    console.print(
        Panel(
            "\n".join(alert_lines),
            title=f"🔔 Alerts ({len(alerts)} generated)",
            border_style="red",
        )
    )


def main() -> None:
    """Main entry point for the BrandPulse AI pipeline."""
    parser = argparse.ArgumentParser(
        description="BrandPulse AI — Customer Feedback Intelligence Pipeline"
    )
    parser.add_argument(
        "--brand",
        type=str,
        help="Process only this brand (e.g., glownest, pureroots, urbanmane)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of reviews to process",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Output filename for results (default: results.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    # Setup
    print_banner()
    load_env()

    if args.verbose:
        setup_logging("DEBUG")

    # ── Load Data ────────────────────────────────────────────────
    console.print("\n[bold]📥 Loading feedback data...[/bold]")
    loader = FeedbackLoader()
    records = loader.load_all()

    if args.brand:
        records = loader.filter_by_brand(records, args.brand)
        console.print(f"   Filtered to brand: [cyan]{args.brand}[/cyan]")

    if args.limit:
        records = records[: args.limit]
        console.print(f"   Limited to: [cyan]{args.limit}[/cyan] records")

    stats = loader.get_stats(records)
    console.print(f"   Total records: [bold]{stats['total']}[/bold]")
    console.print(f"   Brands: {stats.get('brands', {})}")
    console.print(f"   Channels: {stats.get('channels', {})}")
    console.print(f"   Avg rating: {stats.get('avg_rating', 'N/A')}")

    if stats["total"] == 0:
        console.print("[red]No records to process. Exiting.[/red]")
        sys.exit(1)

    # ── Process ──────────────────────────────────────────────────
    console.print("\n[bold]🤖 Starting AI pipeline...[/bold]")
    pipeline = BrandPulsePipeline()

    start_time = time.time()
    results = pipeline.process_batch(records, max_records=args.limit)
    elapsed = time.time() - start_time

    console.print(
        f"\n[bold green]✓ Pipeline complete in {elapsed:.1f}s[/bold green]"
    )

    # ── Summary ──────────────────────────────────────────────────
    print_summary_table(results)

    # ── Anomaly Detection ────────────────────────────────────────
    console.print("\n[bold]🔍 Running anomaly detection...[/bold]")
    detector = AnomalyDetector()
    batch_alerts = detector.detect_anomalies(results)

    # Collect per-review alerts
    review_alerts = []
    for r in results:
        review_alerts.extend(r.get("alerts", []))

    all_alerts = review_alerts + batch_alerts
    print_alerts_summary(all_alerts)

    # ── Notify ───────────────────────────────────────────────────
    notifier = AlertNotifier()
    notifier.notify_batch(all_alerts)

    # ── Save Results ─────────────────────────────────────────────
    output_data = {
        "metadata": {
            "processed_at": datetime.now().isoformat(),
            "total_records": len(results),
            "processing_time_seconds": round(elapsed, 2),
            "filters": {
                "brand": args.brand,
                "limit": args.limit,
            },
        },
        "results": results,
        "alerts": all_alerts,
        "statistics": {
            "successful": sum(1 for r in results if r.get("processed")),
            "failed": sum(1 for r in results if not r.get("processed")),
            "total_alerts": len(all_alerts),
        },
    }

    output_path = save_results(output_data, args.output)
    console.print(f"\n[bold]💾 Results saved to:[/bold] {output_path}")

    # ── Final Stats ──────────────────────────────────────────────
    successful = output_data["statistics"]["successful"]
    failed = output_data["statistics"]["failed"]
    console.print(
        Panel(
            f"[green]Processed:[/green] {successful}  "
            f"[red]Failed:[/red] {failed}  "
            f"[yellow]Alerts:[/yellow] {len(all_alerts)}  "
            f"[cyan]Time:[/cyan] {elapsed:.1f}s",
            title="📈 Final Statistics",
            border_style="cyan",
        )
    )


if __name__ == "__main__":
    main()

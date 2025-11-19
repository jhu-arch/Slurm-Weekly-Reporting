#!/usr/bin/env python3
import subprocess
import argparse
from datetime import date, timedelta
import csv
from collections import defaultdict
import sys
import yaml
from pathlib import Path
import re

# -----------------------------------------
# Load and validate YAML configuration
# -----------------------------------------

CONFIG_FILE = Path("/data/rfadmin/mprotzm2_scripts/sreport_weekly/cluster_config.yaml")

if not CONFIG_FILE.exists():
    print(f"[FATAL] config file missing: {CONFIG_FILE}", file=sys.stderr)
    sys.exit(1)

with open(CONFIG_FILE) as f:
    CONFIG = yaml.safe_load(f)

# Mandatory keys or die
REQUIRED_KEYS = [
    "gpu_type_map",
    "gpu_types",
    "billing_rates",
    "commands",
    "pi_account_grouping",
]

for key in REQUIRED_KEYS:
    if key not in CONFIG:
        print(f"[FATAL] Missing required top-level key '{key}' in cluster_config.yaml", file=sys.stderr)
        sys.exit(1)

GPU_TYPE_MAP = CONFIG["gpu_type_map"]
GPU_TYPES = CONFIG["gpu_types"]
RATES = CONFIG["billing_rates"]
COMMANDS = CONFIG["commands"]
GROUPING = CONFIG["pi_account_grouping"]

# -----------------------------------------
# Helper: Run shell command
# -----------------------------------------

def run(cmd):
    """Execute command and return lines."""
    try:
        return subprocess.check_output(cmd, universal_newlines=True).strip().splitlines()
    except Exception as e:
        print(f"[FATAL] Command failed: {cmd}\n{e}", file=sys.stderr)
        sys.exit(1)

# -----------------------------------------
# Read scontrol for partition CPU defaults
# -----------------------------------------

def get_partition_gpu_defaults():
    """Return defcpu and GPU flags using scontrol."""
    sc = COMMANDS["scontrol"]
    parts = run([sc, "show", "partition"])
    
    partition = None
    defcpu = {}
    is_gpu = {}

    for line in parts:
        line = line.strip()

        if line.startswith("PartitionName="):
            partition = line.split("=", 1)[1]
            defcpu[partition] = 0
            is_gpu[partition] = False

        if "gres/gpu" in line:
            if partition:
                is_gpu[partition] = True

        m = re.search(r"DefCpuPerGPU=(\d+)", line)
        if m and partition:
            defcpu[partition] = int(m.group(1))

    return defcpu, is_gpu

# -----------------------------------------
# Interpret TRES from sacct
# -----------------------------------------

def parse_tres(tres_string):
    cpu = 0
    gpu = 0
    gpu_type = None

    for item in tres_string.split(","):
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        try:
            val = float(val)
        except:
            continue

        if key == "cpu":
            cpu = int(val)

        elif key.startswith("gres/gpu:"):
            gpu = int(val)
            raw = key.split(":", 1)[1]
            gpu_type = GPU_TYPE_MAP.get(raw, raw)

        elif key == "gres/gpu":
            if gpu == 0:
                gpu = int(val)
                gpu_type = "unknown"

    return cpu, gpu, gpu_type

# -----------------------------------------
# Table formatting
# -----------------------------------------

def print_table(headers, rows):
    col_widths = [len(h) for h in headers]

    for row in rows:
        for i, col in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(col)))

    divider = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def fmt_row(r):
        return "|" + "|".join(
            f" {str(col).ljust(col_widths[i])} "
            for i, col in enumerate(r)
        ) + "|"

    print(divider)
    print(fmt_row(headers))
    print(divider)
    for r in rows:
        print(fmt_row(r))
    print(divider)


def fmt_cost(x):
    return "-" if abs(x) < 1e-12 else f"${x:.2f}"

# -----------------------------------------
# Main
# -----------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Config-driven Slurm Accounting")
    parser.add_argument("-s", "--start", help="Start date YYYY-MM-DD")
    parser.add_argument("-e", "--end", help="End date YYYY-MM-DD")
    parser.add_argument("--rates", action="store_true", help="Show dollar cost instead of hours")
    parser.add_argument("--csv", nargs="?", const="", default=None, help="Write results to CSV (auto-named if filename omitted)")
    args = parser.parse_args()

    # Date range logic
    if not args.start or not args.end:
        today = date.today()
        start = today - timedelta(days=7)
        end = today
        START = start.isoformat()
        END = end.isoformat()
    else:
        START = args.start
        END = args.end

    print(f"Using date range {START} → {END}", file=sys.stderr)

    # Load partition defaults
    defcpu, is_gpu = get_partition_gpu_defaults()
    cpu_only_parts = {p for p, gpu in is_gpu.items() if not gpu}

    # Accumulators
    cpu_core_hours = defaultdict(float)
    gpu_hours = defaultdict(lambda: defaultdict(float))
    extra_cpu_hours = defaultdict(float)

    # Call sacct
    sacct = COMMANDS["sacct"]

    cmd = [
        sacct,
        "-S", START,
        "-E", END,
        "-a",
        "-X",
        "-P",
        "-n",
        "-o", "account,partition,elapsedraw,alloctres"
    ]
    rows = run(cmd)

    for line in rows:
        acc, part, elraw, tres = line.split("|")

        if not acc or elraw in ("", "Unknown"):
            continue

        hours = float(elraw) / 3600.0
        cpu_n, gpu_n, gpu_type = parse_tres(tres)

        if part in cpu_only_parts:
            cpu_core_hours[acc] += cpu_n * hours
            continue

        if is_gpu.get(part, False) and gpu_n > 0:
            gpu_hours[acc][gpu_type] += gpu_n * hours
            bundled = defcpu.get(part, 0) * gpu_n
            extra = max(cpu_n - bundled, 0)
            extra_cpu_hours[acc] += extra * hours
            continue

        if cpu_n > 0:
            cpu_core_hours[acc] += cpu_n * hours

    # -----------------------------------------
    # Build table
    # -----------------------------------------

    if args.rates:
        headers = [
            "account",
            "cpu_cost",
            *[f"{g}_gpu_cost" for g in GPU_TYPES],
            "extra_cpu_cost",
            "total_cost",
        ]
    else:
        headers = [
            "account",
            "cpu_core_hours",
            *[f"{g}_gpu_hours" for g in GPU_TYPES],
            "extra_cpu_hours",
        ]

    accounts = sorted(set(cpu_core_hours) | set(extra_cpu_hours) | set(gpu_hours))
    rows_out = []

    for acc in accounts:
        cpu_hrs = cpu_core_hours[acc]
        extra_hrs = extra_cpu_hours[acc]
        row = [acc]

        # CPU
        if args.rates:
            cpu_cost = cpu_hrs * RATES["cpu_core_hour"]
            row.append(fmt_cost(cpu_cost))
        else:
            row.append(f"{cpu_hrs:.2f}")

        # GPUs
        gpu_costs = []
        for g in GPU_TYPES:
            hrs = gpu_hours[acc].get(g, 0)
            if args.rates:
                cost = hrs * RATES.get(f"{g}_gpu_hour", 0)
                gpu_costs.append(cost)
                row.append(fmt_cost(cost))
            else:
                row.append(f"{hrs:.2f}")

        # Extra CPU
        if args.rates:
            extra_cost = extra_hrs * RATES["cpu_core_hour"]
            row.append(fmt_cost(extra_cost))
        else:
            row.append(f"{extra_hrs:.2f}")

        # Total
        if args.rates:
            total = cpu_cost + sum(gpu_costs) + extra_cost
            row.append(fmt_cost(total))

        rows_out.append(row)

    # -----------------------------------------
    # Add total row
    # -----------------------------------------

    total_row = ["TOTAL"]

    if args.rates:
        total_row.append(fmt_cost(sum(cpu_core_hours[acc] * RATES["cpu_core_hour"] for acc in accounts)))
    else:
        total_row.append(f"{sum(cpu_core_hours.values()):.2f}")

    for g in GPU_TYPES:
        g_sum = sum(gpu_hours[acc].get(g, 0) for acc in accounts)
        if args.rates:
            total_row.append(fmt_cost(g_sum * RATES.get(f"{g}_gpu_hour", 0)))
        else:
            total_row.append(f"{g_sum:.2f}")

    if args.rates:
        extra_sum = sum(extra_cpu_hours[acc] for acc in accounts)
        total_row.append(fmt_cost(extra_sum * RATES["cpu_core_hour"]))
        grand = (
            sum(cpu_core_hours[acc] * RATES["cpu_core_hour"] for acc in accounts)
            + sum(
                gpu_hours[acc].get(g, 0) * RATES.get(f"{g}_gpu_hour", 0)
                for acc in accounts
                for g in GPU_TYPES
            )
            + extra_sum * RATES["cpu_core_hour"]
        )
        total_row.append(fmt_cost(grand))
    else:
        total_row.append(f"{sum(extra_cpu_hours.values()):.2f}")

    rows_out.append(total_row)

    # -----------------------------------------
    # CSV output (auto-name if directory or empty)
    # -----------------------------------------
    if args.csv is not None:
        # If user passed just "--csv" with no filename
        if args.csv == "" or args.csv.endswith("/"):
            outname = f"usage_{START}_to_{END}.csv"
            csv_path = Path(args.csv) / outname if args.csv.endswith("/") else Path(outname)
        else:
            csv_path = Path(args.csv)

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(rows_out)

        print(f"Wrote CSV: {csv_path}")
    # -----------------------------------------
    # Print table
    # -----------------------------------------

    print("\n=== Usage Summary ===\n")
    print_table(headers, rows_out)


if __name__ == "__main__":
    main()
# ARCH Slurm Reporting Toolkit

This repository contains two complementary Slurm-driven reporting tools used in HPC environments at scale. Together they cover both automated weekly PI usage reporting and flexible CPU/GPU accounting for billing or internal analysis.

The two scripts are:

1. weekly_pi_emails.py – automated weekly per-PI usage reporting and an administrative cluster summary.
2. accounting.py – configurable accounting and billing calculator for CPU and GPU usage over any date range.

Both scripts are fully driven by cluster_config.yaml, ensuring portability across clusters with different GPU types, Slurm paths, and billing models.

----------------------------------------------------------------------
1. weekly_pi_emails.py — Automated Weekly PI Usage Reports
----------------------------------------------------------------------

Purpose
-------
Automatically generates and emails weekly usage summaries for each PI. Each PI receives a breakdown of CPU and GPU usage across their Slurm accounts. An administrative summary is also generated, archived, and emailed to help@arch.jhu.edu.

What It Does
------------
- Queries sacct and sreport for the previous Monday–Sunday period.
- Builds per-PI usage reports including:
  - CPU core-hours
  - GPU-hours by canonical GPU type
  - Per-user breakdown
- Automatically groups Slurm accounts into PI families using rules in cluster_config.yaml.
- Looks up PI email addresses from LDAP.
- Sends emails through SMTP using configurable sender and signature.
- Generates an admin summary including:
  - Total cluster hours used
  - Top accounts by usage
  - Top users by usage
  - Number of PIs with and without usage
- Archives each PI email and the admin summary.
- Logs all actions for later audit.

Key Features
------------
- dry-run mode (prints output, sends no PI emails)
- test-run mode (only sends selected PI's email to help@arch.jhu.edu)
- automatic regeneration of stale PI data files
- configurable entirely through cluster_config.yaml

Directory Layout
----------------
sreport_weekly/
  weekly_pi_emails.py
  cluster_config.yaml
  datadump/
  logs/
  archive/

Usage
-----
Full weekly run:
    python weekly_pi_emails.py

Dry run:
    python weekly_pi_emails.py --dry-run

Test run for a single PI:
    python weekly_pi_emails.py --test-run --pi jsmith1

Run for one PI only:
    python weekly_pi_emails.py --pi jsmith1

----------------------------------------------------------------------
2. accounting.py — Slurm Accounting & Billing Calculator
----------------------------------------------------------------------

Purpose
-------
Generates detailed accounting of CPU core-hours and GPU-hours over any date range. Supports optional billing rate calculations. Useful for quarterly reports, grant justifications, recharge modeling, and cluster usage analysis.

What It Does
------------
- Queries sacct for raw job data over a user-defined date range.
- Parses TRES allocations to compute:
  - CPU core-hours
  - GPU-hours by canonical GPU type
  - Extra CPU-hours above DefCpuPerGPU bundle (we're likely not charging for these, so they're separated in the output)
- Canonicalizes GPU type names using gpu_type_map from YAML.
- Applies billing rates stored in cluster_config.yaml when --rates is used.
- Detects CPU-only vs GPU partitions via scontrol.
- Outputs:
  - A formatted on-screen table
  - Optional CSV file (auto-named based on the active date range)

Key Features
------------
- Fully YAML-driven billing and GPU mappings.
- Strict configuration validation (fails loudly if anything is missing).
- Auto-named CSV output:
      usage_2025-01-01_to_2025-01-08.csv
- Works with any date range.

Usage
-----
Default (last 7 days):
    python accounting.py

Custom date range:
    python accounting.py -s 2025-01-01 -e 2025-01-08

Billing mode:
    python accounting.py --rates

Auto-named CSV:
    python accounting.py --csv

Custom CSV filename:
    python accounting.py --csv billing_q1.csv

----------------------------------------------------------------------
3. Shared Configuration — cluster_config.yaml
----------------------------------------------------------------------

Both scripts depend on this config file.

Controls include:
- gpu_type_map: maps raw Slurm GPU labels to canonical types (a100, v100, etc.)
- gpu_types: ordered list of canonical GPU types for output columns
- billing_rates: CPU and GPU hourly billing rates
- commands: paths to sacct, sreport, and scontrol binaries
- pi_account_grouping: logic for grouping Slurm accounts into root PI identifiers
- email sender and signature for weekly reports

The scripts have no internal defaults. Missing or inconsistent config entries result in immediate failure to avoid silent billing/reporting errors.

Example Structure
-----------------
gpu_types:
  - a100
  - l40s
  - v100

gpu_type_map:
  - a100: a100
  - ica100: a100
  - l40s: l40s
  - v100: v100
  - mig_class: a100

billing_rates:
  - cpu_core_hour: 0.00
  - a100_gpu_hour: 0.00
  - l40s_gpu_hour: 0.00
  - v100_gpu_hour: 0.00

commands:
  - sacct: /usr/bin/sacct
  - sreport: /usr/bin/sreport
  - scontrol: /usr/bin/scontrol

pi_account_grouping:
  - method: prefix
  - delimiters: ["_", "-"]
  - strip_suffixes:
    - gpu
    - a100
    - l40s
    - v100
    - mig
    - condo
    - bigmem
    - extended

email:
  sender: help@arch.jhu.edu
  admin_email: help@arch.jhu.edu
  signature:
    - ARCH Help Team
    - Advanced Research Computing at Hopkins (ARCH)
    - help@arch.jhu.edu
    - https://www.arch.jhu.edu
    - https://docs.arch.jhu.edu

----------------------------------------------------------------------
4. When to Use Each Script
----------------------------------------------------------------------

Use weekly_pi_emails.py when:
- You need automated weekly PI-facing reports.
- You want per-user breakdowns sent directly to research groups.
- You want an admin-level weekly usage snapshot.
- You need archived copies of all weekly communication.
- You want proactive cluster usage transparency for researchers.

Use accounting.py when:
- You need flexible time windows (days, weeks, quarters).
- You want CSV output for leadership or finance.
- You want billing or cost modeling for CPU/GPU usage.
- You want standardized usage categorization across accounts.
- You need a standalone accounting report without email.

----------------------------------------------------------------------
5. Quick Start Summary
----------------------------------------------------------------------

1. Place cluster_config.yaml in the directory with the scripts.
2. Run weekly_pi_emails.py for automated Monday-Sunday PI reports.
3. Run accounting.py for billing or usage analysis over any time range.
4. Use --csv on accounting.py for spreadsheet-friendly output.
5. Check archive/ and logs/ for historical records of weekly activity.

This toolkit is designed to be maintainable, auditable, and portable across clusters, requiring no hardcoded paths or rates inside the Python source itself.
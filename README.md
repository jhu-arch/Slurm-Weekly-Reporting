# ARCH Slurm Reporting Toolkit

This repository contains two complementary Slurm-driven reporting tools. Together they cover both automated weekly PI usage reporting and flexible CPU/GPU accounting for billing or internal analysis.

1. `weekly_pi_emails.py` – automated weekly per-PI usage reporting and an administrative cluster summary.
2. `accounting.py` – configurable accounting and billing calculator for CPU and GPU usage over any date range.

Both scripts are driven by **cluster_config.yaml**, ensuring portability across clusters with different GPU types, Slurm paths, and billing models.

## 1. `weekly_pi_emails.py` — Automated Weekly PI Usage Reports

### Purpose

Automatically generates and emails weekly usage summaries for each PI. Each PI receives a breakdown of CPU and GPU usage across their Slurm accounts. An administrative summary is also generated, archived, and emailed to the configured help email.

#### PI reports
- Builds PI → Slurm account groupings
- Pulls usage from `sacct` for the reporting window
- Sends **one email per PI** *only if the PI has non-zero usage*
- Uses Unix group membership (`getent group` + `getent passwd`) to build a PI & user roster
- Includes per-user breakdown
  - CPU and GPU sections include **only users with usage**
  - Users with **no usage across all tracked resources** are listed in:
    - “The following users had no usage for this period: …”

#### User reports
- Sends **one email per user** *only if the user has non-zero usage*
- Groups the user’s usage by PI root (research group)

#### Admin summary
- Generates an admin summary containing:
  - Date range
  - Total compute hours (CPU + GPU-hours)
  - Counts of reports sent / skipped
  - Top accounts and top users by usage

#### Directory Layout
```
- sreport_weekly/
  - weekly_pi_emails.py
  - cluster_config.yaml
  - datadump/
  - logs/
  - archive/
```
#### Usage

Full weekly run:
- Send **weekly PI emails** (one email per PI with non-zero usage) and the admin summary:
	- ```python weekly_pi_emails.py --run-pi-emails```

- Send **weekly User emails** (one email per user with non-zero usage) and the admin summary:
	- ```python weekly_pi_emails.py --run-user-emails```

- Test Runs - routes outgoing email to the configured test sink and sends only one message total 
	- ```python weekly_pi_emails.py --run-pi-emails --pi $PI --test-run```
	- ```python weekly_pi_emails.py --run-user-emails --user $PI --test-run```

- Dry run - prints PI and / or user emails to stdout instead of sending them. Admin summary still sends to test sink.
	- ```python weekly_pi_emails.py --run-pi-emails --dry-run```
	- ```python weekly_pi_emails.py --run-user-emails --dry-run```

##  2. `accounting.py` — Slurm Accounting & Billing Calculator

### Purpose

Generates detailed accounting of CPU core-hours and GPU-hours over any date range. Supports optional billing rate calculations. Useful for quarterly reports, grant justifications, recharge modeling, and cluster usage analysis.

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

#### Usage

- Default (last 7 days):
	- ```python accounting.py```

- Billing mode:
	- ```python accounting.py --rates```

- Auto-named CSV:
	- ```python accounting.py --csv```

- Custom CSV filename:
	- ```python accounting.py --csv billing_q1.csv```

## 3. Shared Configuration — `cluster_config.yaml`

Both scripts depend on this config file.

Controls include:
- **gpu_type_map**: maps raw Slurm GPU labels to canonical types (a100, v100, etc.)
- **gpu_types**: ordered list of canonical GPU types for output columns
- **billing_rates**: CPU and GPU hourly billing rates
- **commands**: paths to sacct, sreport, and scontrol binaries
- **pi_account_grouping**: logic for grouping Slurm accounts into root PI identifiers
- **email**: sender and signature for weekly reports
	- **Optional**: throttle outbound mail (seconds).
	- Recommended if you’re worried about SMTP load.
		- send_delay_seconds: 0.25

The scripts have no internal defaults. Missing or inconsistent config entries result in immediate failure to avoid silent billing/reporting errors.

### Example Structure

```
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
  - sender: help@arch.jhu.edu
  - admin_email: help@arch.jhu.edu
  - send_delay_seconds: 0.25

  - signature:
    - ARCH Help Team
    - Advanced Research Computing at Hopkins (ARCH)
    - help@arch.jhu.edu
    - https://www.arch.jhu.edu
    - https://docs.arch.jhu.edu
```

This toolkit is designed to be maintainable, auditable, and portable across clusters, requiring no hardcoded paths or rates inside the Python source itself.

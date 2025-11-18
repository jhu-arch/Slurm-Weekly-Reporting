# weekly_pi_emails

A Slurm-driven weekly usage reporting system for HPC clusters.  
This script queries sacct to collect CPU and GPU usage for all research groups (PIs), aggregates it, and sends out individualized weekly usage reports. An admin summary is also generated.

The design is fully YAML-configurable, making it portable across clusters with different GPU types, partitions, email settings, and Slurm paths.

---

## Features

- Collects weekly CPU core-hours and GPU-hours via sacct
- Automatically differentiates CPU-only vs GPU-enabled partitions
- Canonicalizes GPU types by mapping raw partition/TRES values to A100, L40S, V100, etc.
- Groups Slurm accounts into PI “ownership trees” using flexible prefix/suffix rules
- Pulls PI → email mappings from LDAP
- Sends weekly reports directly to each PI
- Generates and archives an admin summary (top accounts/users)
- Supports:
  - dry runs
  - test runs (send only one PI's report to help@…)
  - auto-regeneration of PI metadata
- Easy portability through cluster_config.yaml

---

## Requirements

- Python 3.8+
- PyYAML (Python package)
- Slurm tools available and readable:
  - sacct
  - sreport
  - scontrol
  - sacctmgr
- LDAP client (for PI email lookup)

Install PyYAML, for example:

    pip install pyyaml

---

## Configuration (cluster_config.yaml)

All cluster-specific logic is contained in this file.

Example structure:

    gpu_types:
      - a100
      - l40s
      - v100

    gpu_type_map:
      a100: a100
      ica100: a100
      mig: a100
      l40s: l40s
      v100: v100
      gpu-a100: a100
      partition-a100: a100

    billing_rates:
      cpu: 0.00
      a100: 0.00
      l40s: 0.00
      v100: 0.00

    commands:
      sacct: /usr/bin/sacct
      sreport: /usr/bin/sreport
      scontrol: /usr/bin/scontrol

    pi_account_grouping:
      method: "prefix"
      delimiters: ["_", "-"]
      strip_suffixes:
        - "gpu"
        - "a100"
        - "ica100"
        - "l40s"
        - "v100"
        - "mig"
        - "condo"
        - "bigmem"
        - "extended"

    email:
      sender: help@arch.jhu.edu
      signature:
        - "ARCH Help Team"
        - "Advanced Research Computing at Hopkins (ARCH)"
        - "help@arch.jhu.edu"
        - "https://www.arch.jhu.edu"
        - "https://docs.arch.jhu.edu"

The script validates the YAML strictly and will exit with a helpful message if:

- a referenced GPU type has no mapping
- a Slurm command path does not exist
- required config keys are missing

---

## Usage

Run from the host where Slurm commands and LDAP are available.

### Full weekly run

    python weekly_pi_emails.py

### Dry run (prints PI emails + admin summary, but sends nothing except admin)

    python weekly_pi_emails.py --dry-run

This will:

- build usage
- print per-PI emails to stdout
- print the admin summary
- still send the admin summary email to help@arch.jhu.edu (preserving legacy behavior)

### Test run (build everything, but send only one PI’s email to help@arch.jhu.edu)

    python weekly_pi_emails.py --test-run --pi jsmith1

### Single PI only (normal run)

    python weekly_pi_emails.py --pi jsmith1

---

## Directory layout

Typical layout on disk:

    sreport_weekly/
      weekly_pi_emails.py
      cluster_config.yaml
      datadump/
        pi_accounts.json
        pi_emails.json
        usage_<start>_<end>.json
      logs/
        rf_weekly_YYYY-MM-DD.log
      archive/
        <start>_to_<end>/
          <pi_label>.txt
          admin_summary.txt

You can adjust BASE_DIR inside the script if you want a different tree.

---

## Data files

The script maintains:

- datadump/pi_accounts.json  
- datadump/pi_emails.json  
- datadump/usage_<start>_<end>.json

pi_accounts.json and pi_emails.json regenerate automatically when:

- they are missing, or  
- they are older than 7 days

The regeneration logic is driven by pi_account_grouping in cluster_config.yaml, so you can adapt it for other clusters without touching the Python.

---

## PI report format

Each PI receives a plaintext email containing:

- A summary of total CPU and GPU hours across all of their Slurm accounts
- A per-user breakdown for:
  - CPU core-hours (CPU Hours)
  - A100 GPU-hours (A100 Hours)
  - L40S GPU-hours (L40S Hours)
  - V100 GPU-hours (V100 Hours)

Sections with zero usage are omitted to keep the email compact.

The email also includes:

- The reporting date range (last Monday through last Sunday)
- A brief explanation of what is being reported
- A contact address pulled from email.sender in the YAML
- A configurable signature block from email.signature

---

## Admin summary

After PI emails are processed, an admin summary is generated. It includes:

- Run timestamp
- Date range (start → end)
- Total compute hours used (CPU + GPU-hours)
- Total PIs processed
- PIs with usage (emails sent)
- PIs with zero usage (skipped)
- Top 5 Slurm accounts by total hours
- Top 5 users by total hours (CPU + GPU combined)

The summary is:

- archived under archive/<start>_to_<end>/admin_summary.txt  
- emailed to help@arch.jhu.edu

---

## Logging and archiving

Logs are written to:

- logs/rf_weekly_YYYY-MM-DD.log

Per-PI emails are archived (subject + body) in:

- archive/<start>_to_<end>/<pi_label>.txt

The admin summary is archived as:

- archive/<start>_to_<end>/admin_summary.txt

This makes it easy to audit what was sent for any given week.

---

The current weekly_pi_emails.py has its own regeneration logic (ensure_fresh_data), which:

- checks age of pi_accounts.json and pi_emails.json
- rebuilds them when they are missing or older than 7 days
- uses the YAML-driven pi_account_grouping rules

In other words: generate_pi_data.py is no longer required for normal operation. You can keep it in the repo as an optional standalone utility or remove it entirely.

---

## Automation

You can automate weekly runs with cron or systemd.

Example cron entry (run every Monday at 09:00):

    0 9 * * MON /usr/bin/python3 /path/to/weekly_pi_emails.py >> /var/log/weekly_pi_emails_cron.log 2>&1

---

## Contributions

Contributions, suggestions, and cluster-specific examples (different GPU maps, partition layouts, etc.) are welcome.


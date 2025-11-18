#!/usr/bin/env python3
import os
import sys
import json
import smtplib
import subprocess
from pathlib import Path
from datetime import date, timedelta, datetime
from email.mime.text import MIMEText
from collections import defaultdict

try:
    import yaml
except ImportError:
    print("ERROR: This script requires PyYAML. Install it with:\n  pip install pyyaml", file=sys.stderr)
    sys.exit(1)

# =============================
# BASE PATHS (cluster-agnostic)
# =============================
BASE_DIR = Path("/data/rfadmin/mprotzm2_scripts//sreport_weekly")
LOG_DIR = BASE_DIR / "logs"
ARCHIVE_DIR = BASE_DIR / "archive"

DATA_DIR = BASE_DIR / "datadump"
PI_ACCOUNTS_JSON = DATA_DIR / "pi_accounts.json"
PI_EMAILS_JSON = DATA_DIR / "pi_emails.json"

CONFIG_FILE = BASE_DIR / "cluster_config.yaml"
SMTP_SERVER = "localhost"  # can be externalized later if you want

# Will be filled by load_config() in main()
CONFIG = None
SENDER = None  # comes from CONFIG["email"]["sender"]


# =============================
# Config helpers / validation
# =============================
def config_error(msg: str):
    print(f"ERROR in cluster_config.yaml: {msg}", file=sys.stderr)
    sys.exit(1)


def load_config():
    """Load and validate cluster_config.yaml with strict checks."""
    if not CONFIG_FILE.exists():
        config_error(f"Config file not found at {CONFIG_FILE}")

    try:
        with open(CONFIG_FILE, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        config_error(f"Failed to parse YAML: {e}")

    if not isinstance(cfg, dict):
        config_error("Top-level structure must be a mapping (YAML dictionary).")

    # Required top-level keys
    for key in ["gpu_type_map", "gpu_types", "billing_rates", "email", "commands", "pi_account_grouping"]:
        if key not in cfg:
            config_error(f"Missing required top-level key: {key}")

    # Email section
    email_cfg = cfg["email"]
    if "sender" not in email_cfg:
        config_error("email.sender is required.")
    if "signature" not in email_cfg or not isinstance(email_cfg["signature"], list):
        config_error("email.signature must be a list of lines.")

    # Commands
    cmd_cfg = cfg["commands"]
    for prog in ["sacct", "sreport", "scontrol"]:
        if prog not in cmd_cfg:
            config_error(f"commands.{prog} is required.")
        path = Path(cmd_cfg[prog])
        if not path.exists():
            config_error(f"Configured path for {prog} does not exist: {path}")

    # GPU mapping: loose mode but with sanity checks
    gpu_type_map = cfg["gpu_type_map"]
    gpu_types = cfg["gpu_types"]

    if not isinstance(gpu_type_map, dict) or not isinstance(gpu_types, list):
        config_error("gpu_type_map must be a mapping and gpu_types must be a list.")

    # Canonical GPU types that appear as mapping targets
    mapped_types = {str(v).lower() for v in gpu_type_map.values()}
    needed_types = {str(t).lower() for t in gpu_types}

    missing_types = needed_types - mapped_types
    if missing_types:
        config_error(
            "The following GPU types are listed in gpu_types but never appear "
            f"as values in gpu_type_map: {', '.join(sorted(missing_types))}\n"
            "You must map at least one partition or GPU name to each of these types."
        )

    # PI account grouping
    grp = cfg["pi_account_grouping"]
    if grp.get("method") != "prefix":
        config_error("pi_account_grouping.method must be 'prefix' for this script.")
    if "delimiters" not in grp or not isinstance(grp["delimiters"], list) or not grp["delimiters"]:
        config_error("pi_account_grouping.delimiters must be a non-empty list.")
    if "strip_suffixes" not in grp or not isinstance(grp["strip_suffixes"], list):
        config_error("pi_account_grouping.strip_suffixes must be a list (can be empty).")

    return cfg


# =============================
# Logging
# =============================
def log(message):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"rf_weekly_{date.today()}.log"
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)


# =============================
# Shell helper
# =============================
def run_cmd(cmd, use_shell=False):
    """Run a shell command and return output as string."""
    try:
        if use_shell:
            return subprocess.check_output(
                cmd, shell=True, universal_newlines=True
            ).strip()
        else:
            return subprocess.check_output(
                cmd, universal_newlines=True
            ).strip()
    except subprocess.CalledProcessError as e:
        log(f"[!] Command failed: {cmd}\n{e}")
        return ""


# =============================
# Date helper
# =============================
def get_last_week_range():
    """
    Return Monday–Sunday date strings for last week.
    This preserves the old behavior of the original script.
    """
    today = date.today()
    last_monday = today - timedelta(days=today.weekday() + 7)
    last_sunday = last_monday + timedelta(days=6)
    return str(last_monday), str(last_sunday)


# =============================
# PI account builders (YAML-driven)
# =============================
def derive_pi_root(account: str) -> str:
    """
    Use pi_account_grouping rules from CONFIG to derive a PI root name
    from a Slurm account.

    Logic:
      - Repeatedly strip configured suffixes with configured delimiters
        from the END of the account name.
        e.g. mschatz1_gpu_condo -> mschatz1
    """
    grp = CONFIG["pi_account_grouping"]
    delimiters = grp["delimiters"]
    suffixes = grp["strip_suffixes"]

    root = account
    changed = True
    while changed:
        changed = False
        for delim in delimiters:
            for suf in suffixes:
                token = f"{delim}{suf}"
                if root.endswith(token):
                    root = root[: -len(token)]
                    changed = True
    return root


def build_pi_accounts():
    """Generate mapping of PI → associated accounts from Slurm."""
    log("[→] Fetching PI → account mapping from Slurm…")
    # sacctmgr is still assumed to be on PATH; you can also externalize this later if desired.
    output = run_cmd("sacctmgr show accounts format=Account -P", use_shell=True)
    lines = [l.strip() for l in output.splitlines() if l.strip() and l != "Account"]
    mapping = {}
    for acc in lines:
        root = derive_pi_root(acc)
        mapping.setdefault(root, []).append(acc)
    log(f"[✓] Found {len(mapping)} PI account groups.")
    return mapping


def build_pi_emails():
    """Query LDAP for uid/mail and filter invalid ones."""
    log("[→] Querying LDAP for PI emails…")
    cmd = "ldapsearch -x -LLL -b 'dc=cm,dc=cluster' '(uid=*)' uid mail"
    output = run_cmd(cmd, use_shell=True)
    emails = {}
    current_uid = None
    for line in output.splitlines():
        if line.startswith("uid: "):
            current_uid = line.split("uid: ")[1].strip()
        elif line.startswith("mail: ") and current_uid:
            mail = line.split("mail: ")[1].strip()
            if mail and not mail.startswith("NULL-"):
                emails[current_uid] = mail
            current_uid = None
    log(f"[✓] Found {len(emails)} valid LDAP email addresses.")
    return emails


def ensure_fresh_data():
    """Regenerate pi_accounts.json and pi_emails.json if missing or older than 7 days."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stale = False
    for f in [PI_ACCOUNTS_JSON, PI_EMAILS_JSON]:
        if not f.exists():
            stale = True
        else:
            age_days = (date.today() - date.fromtimestamp(f.stat().st_mtime)).days
            if age_days > 7:
                stale = True
    if stale:
        log("[!] Data files missing or stale — regenerating.")
        accounts = build_pi_accounts()
        emails = build_pi_emails()
        with open(PI_ACCOUNTS_JSON, "w") as fa:
            json.dump(accounts, fa, indent=2)
        with open(PI_EMAILS_JSON, "w") as fe:
            json.dump(emails, fe, indent=2)
        log(f"[✓] Wrote new data files to {DATA_DIR}")
    else:
        log("[✓] Using existing, fresh data files.")


# =============================
# Partition + TRES helpers
# =============================
def get_gpu_partitions():
    """
    Use `scontrol show partition` to discover which partitions are GPU partitions.
    We treat any partition with 'gres/gpu' in its description as a GPU partition.
    """
    scontrol_bin = CONFIG["commands"]["scontrol"]
    output = run_cmd([scontrol_bin, "show", "partition"])
    gpu_parts = set()
    current_part = None

    for raw in output.splitlines():
        line = raw.strip()
        if line.startswith("PartitionName="):
            current_part = line.split("=", 1)[1]
            continue
        if "gres/gpu" in line and current_part:
            gpu_parts.add(current_part)

    log(f"[✓] Detected GPU partitions: {', '.join(sorted(gpu_parts))}")
    return gpu_parts


def parse_tres(tres_string):
    """
    Parse AllocTRES string like:
      'billing=12,cpu=12,gres/gpu:a100=1,gres/gpu=1,mem=48000M,node=1'
    and return (cpu_count, gpu_count, raw_gpu_type or None).
    """
    cpu = 0
    gpu = 0
    gpu_type = None

    if not tres_string:
        return cpu, gpu, gpu_type

    for item in tres_string.split(","):
        if "=" not in item:
            continue
        key, val = item.split("=", 1)
        try:
            val_f = float(val)
        except ValueError:
            continue

        if key == "cpu":
            cpu = int(val_f)
        elif key.startswith("gres/gpu:"):
            gpu = int(val_f)
            gpu_type = key.split(":", 1)[1]
        elif key == "gres/gpu":
            # generic GPU count; use only if we don't already have a typed one
            if gpu == 0:
                gpu = int(val_f)
                if gpu_type is None:
                    gpu_type = "unknown"

    return cpu, gpu, gpu_type


def normalize_gpu_type(raw_gpu_type, partition):
    """
    Map raw GPU type / partition name into canonical buckets using CONFIG["gpu_type_map"].

    Logic:
      1. Try gpu_type_map[raw_gpu_type.lower()]
      2. Then try gpu_type_map[partition.lower()]
      3. If neither works but we had GPU usage → hard fail with helpful error.
    """
    gpu_map = CONFIG["gpu_type_map"]

    candidates = []
    if raw_gpu_type:
        candidates.append(str(raw_gpu_type).lower())
    if partition:
        candidates.append(str(partition).lower())

    for key in candidates:
        if key in gpu_map:
            return str(gpu_map[key]).lower()

    # If we reach here, we saw a GPU we don't know how to map.
    msg = (
        "[!] Encountered a GPU job with an unmapped GPU type/partition.\n"
        f"    raw_gpu_type = {raw_gpu_type!r}\n"
        f"    partition    = {partition!r}\n\n"
        "Please add an appropriate entry in gpu_type_map inside cluster_config.yaml.\n"
        "Example:\n"
        "  gpu_type_map:\n"
        f"    {raw_gpu_type or partition}: a100   # or l40s/v100/etc\n"
    )
    log(msg)
    sys.exit(1)


# =============================
# Collect usage from sacct (CPU + GPU-hours by account/user)
# =============================
def get_sacct_usage(start, end):
    """
    Return a dict keyed by (account, user) with aggregated hours:
      usage[(account, user)] = {
        "cpu":  <CPU core-hours on CPU-only partitions>,
        "a100": <A100 GPU-hours (including ica100 / MIG A100 etc)>,
        "l40s": <L40S GPU-hours>,
        "v100": <V100 GPU-hours>,
      }

    Extra CPU on GPU partitions is DISCARDED entirely.
    """
    log(f"[→] Running sacct for {start} → {end}")
    gpu_parts = get_gpu_partitions()

    sacct_bin = CONFIG["commands"]["sacct"]
    cmd = [
        sacct_bin,
        "-S", start,
        "-E", end,
        "-a",
        "-X",
        "-P",
        "-n",
        "-o", "account,user,partition,elapsedraw,alloctres",
    ]
    output = run_cmd(cmd)
    if not output:
        log("[!] sacct returned no data.")
        return {}

    usage = defaultdict(lambda: {"cpu": 0.0, "a100": 0.0, "l40s": 0.0, "v100": 0.0})

    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 5:
            continue

        account, user, part, elraw, tres = [p.strip() for p in parts]

        if not account or not user or not elraw or elraw == "Unknown":
            continue

        try:
            seconds = float(elraw)
        except ValueError:
            continue
        if seconds <= 0:
            continue

        hours = seconds / 3600.0
        cpu_count, gpu_count, raw_gpu_type = parse_tres(tres)

        key = (account, user)
        part_is_gpu = part in gpu_parts

        # GPU jobs: only track GPU-hours; discard all CPU.
        if part_is_gpu and gpu_count > 0:
            canon = normalize_gpu_type(raw_gpu_type, part)
            # Only count if canonical type is actually one of our declared gpu_types
            if canon in {t.lower() for t in CONFIG["gpu_types"]}:
                usage[key][canon] += gpu_count * hours
            # If canon somehow isn't in gpu_types, that's a config issue; fail loudly.
            else:
                log(
                    f"[!] Canonical GPU type '{canon}' from gpu_type_map is not listed in gpu_types.\n"
                    "    Please fix gpu_types in cluster_config.yaml."
                )
                sys.exit(1)
            continue

        # CPU-only partitions → CPU core-hours
        if not part_is_gpu and cpu_count > 0:
            usage[key]["cpu"] += cpu_count * hours

    log(f"[✓] Aggregated usage for {len(usage)} (account,user) pairs.")
    return usage


# =============================
# Formatting helpers
# =============================
def fmt_hours(val):
    return f"{val:.2f}"


def signature_block():
    """Return the email signature block from CONFIG as a string."""
    lines = CONFIG["email"]["signature"]
    return "\n".join(lines)


# =============================
# PI report formatting
# =============================
def format_pi_report(pi, accounts, usage_by_acct_user, start=None, end=None):
    """
    Build the full plain-text report for a PI.
    usage_by_acct_user: dict[(account,user)] = {cpu, a100, l40s, v100}
    """
    start = start or "this week"
    end = end or "today"

    # Aggregate per-user usage across all accounts owned by this PI
    per_user = defaultdict(lambda: {"cpu": 0.0, "a100": 0.0, "l40s": 0.0, "v100": 0.0})

    for (account, user), metrics in usage_by_acct_user.items():
        if account not in accounts:
            continue
        per_user[user]["cpu"] += metrics.get("cpu", 0.0)
        per_user[user]["a100"] += metrics.get("a100", 0.0)
        per_user[user]["l40s"] += metrics.get("l40s", 0.0)
        per_user[user]["v100"] += metrics.get("v100", 0.0)

    # No usage at all for this PI
    if not per_user:
        return (
            f"Dear User,\n\n"
            f"This report summarizes compute usage for your research group on the Rockfish cluster "
            f"from {start} through {end}.\n\n"
            f"No recorded CPU or GPU usage was found for your associated accounts during this period.\n\n"
            f"{signature_block()}\n"
        )

    # Totals across all users / accounts
    total_cpu = sum(v["cpu"] for v in per_user.values())
    total_a100 = sum(v["a100"] for v in per_user.values())
    total_l40s = sum(v["l40s"] for v in per_user.values())
    total_v100 = sum(v["v100"] for v in per_user.values())

    # Build summary block (only non-zero fields)
    summary_lines = []
    summary_lines.append("All Accounts Combined:")
    if total_cpu > 0:
        summary_lines.append(f"CPU Hours:  {fmt_hours(total_cpu)}")
    if total_a100 > 0:
        summary_lines.append(f"A100 Hours: {fmt_hours(total_a100)}")
    if total_l40s > 0:
        summary_lines.append(f"L40S Hours: {fmt_hours(total_l40s)}")
    if total_v100 > 0:
        summary_lines.append(f"V100 Hours: {fmt_hours(total_v100)}")

    summary_block = "\n".join(summary_lines)

    # Per-user breakdown
    user_lines = []
    user_lines.append("\nPer-user Breakdown:\n")

    # CPU section
    if total_cpu > 0:
        user_lines.append("CPU Hours:")
        for user in sorted(per_user.keys()):
            if per_user[user]["cpu"] > 0:
                user_lines.append(f"  → {user}: {fmt_hours(per_user[user]['cpu'])}")
        user_lines.append("")

    # A100 section
    if total_a100 > 0:
        user_lines.append("A100 Hours:")
        for user in sorted(per_user.keys()):
            if per_user[user]["a100"] > 0:
                user_lines.append(f"  → {user}: {fmt_hours(per_user[user]['a100'])}")
        user_lines.append("")

    # L40S section
    if total_l40s > 0:
        user_lines.append("L40S Hours:")
        for user in sorted(per_user.keys()):
            if per_user[user]["l40s"] > 0:
                user_lines.append(f"  → {user}: {fmt_hours(per_user[user]['l40s'])}")
        user_lines.append("")

    # V100 section
    if total_v100 > 0:
        user_lines.append("V100 Hours:")
        for user in sorted(per_user.keys()):
            if per_user[user]["v100"] > 0:
                user_lines.append(f"  → {user}: {fmt_hours(per_user[user]['v100'])}")
        user_lines.append("")

    user_block = "\n".join(user_lines).rstrip()

    body = f"""Dear User,

This report summarizes CPU and GPU usage for your research group on the Rockfish cluster from {start} through {end}.

The first section aggregates all of your associated Slurm accounts. The second section breaks the same usage down by individual user login.

Rockfish Weekly Usage Report for {pi}
=====================================

{summary_block}

{user_block}

If you have any questions about your group’s usage or notice discrepancies, please reach out to {CONFIG['email']['sender']}.

{signature_block()}
"""

    return body


# =============================
# Archiving helpers
# =============================
def archive_email(label, subject, body, start, end):
    """Save a copy of an outgoing email for record-keeping."""
    folder = ARCHIVE_DIR / f"{start}_to_{end}"
    folder.mkdir(parents=True, exist_ok=True)

    safe = label.replace("/", "_").replace(" ", "_")
    path = folder / f"{safe}.txt"
    with open(path, "w") as f:
        f.write(f"Subject: {subject}\n\n")
        f.write(body)

    log(f"[✓] Archived {label} email → {path}")


# =============================
# Email sending
# =============================
def send_email(to, subject, body, dry_run=False, archive_label=None, start=None, end=None):
    # Archive every email (including dry-run) if label provided
    if archive_label and start and end:
        archive_email(archive_label, subject, body, start, end)

    if dry_run:
        print(f"\n--- [DRY RUN] Would send to {to} ---")
        print(f"Subject: {subject}\n{body}\n")
        print("-------------------------------------\n")
        return

    msg = MIMEText(body, "plain")
    msg["From"] = SENDER
    msg["To"] = to
    msg["Subject"] = subject

    with smtplib.SMTP(SMTP_SERVER) as s:
        s.send_message(msg)

    log(f"[✓] Sent report to {to}")


# =============================
# Admin summary
# =============================
def admin_summary_report(start, end, sent_count, skipped_no_usage,
                         usage_by_acct_user):
    lines = []
    lines.append("Rockfish Weekly Usage Summary\n")
    lines.append(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Date range: {start} → {end}\n")

    # Aggregate totals
    admin_total_hours = 0.0
    admin_account_usage = defaultdict(float)
    admin_user_usage = defaultdict(float)

    for (account, user), m in usage_by_acct_user.items():
        total = m["cpu"] + m["a100"] + m["l40s"] + m["v100"]
        admin_total_hours += total
        admin_account_usage[account] += total
        admin_user_usage[(user, account)] += total

    total_processed = sent_count + skipped_no_usage

    lines.append(f"Total compute hours used (CPU + GPU-hours): {admin_total_hours:.2f}")
    lines.append(f"Total PIs processed: {total_processed}")
    lines.append(f"PIs with usage (emails sent): {sent_count}")
    lines.append(f"PIs with zero usage (skipped): {skipped_no_usage}\n")

    lines.append("----------------------------------------")
    lines.append("Top 5 Slurm Accounts by Usage")
    lines.append("----------------------------------------")

    top_accounts = sorted(admin_account_usage.items(), key=lambda x: x[1], reverse=True)[:5]
    for acct, hrs in top_accounts:
        lines.append(f"{acct:<20} {hrs:>12.2f} hrs")

    lines.append("\n----------------------------------------")
    lines.append("Top 5 Users by Usage (combined CPU+GPU)")
    lines.append("----------------------------------------")

    top_users = sorted(admin_user_usage.items(), key=lambda x: x[1], reverse=True)[:5]
    for (user, acct), hrs in top_users:
        lines.append(f"{user:<12} {acct:<20} {hrs:>12.2f} hrs")

    lines.append("\n-- End of summary --\n")
    return "\n".join(lines)


def archive_admin_summary(start, end, summary):
    folder = ARCHIVE_DIR / f"{start}_to_{end}"
    folder.mkdir(parents=True, exist_ok=True)

    filepath = folder / "admin_summary.txt"
    with open(filepath, "w") as f:
        f.write(summary)

    log(f"[✓] Archived admin summary → {filepath}")


# =============================
# Main workflow
# =============================
def main(dry_run=False, target_pi=None, test_run=False):
    global CONFIG, SENDER

    # Load + validate YAML config
    CONFIG = load_config()
    SENDER = CONFIG["email"]["sender"]

    ensure_fresh_data()
    pi_accounts = json.load(open(PI_ACCOUNTS_JSON))
    pi_emails = json.load(open(PI_EMAILS_JSON))

    start, end = get_last_week_range()
    usage_by_acct_user = get_sacct_usage(start, end)

    # Dump usage for debug/record
    usage_file = DATA_DIR / f"usage_{start}_{end}.json"
    dump_list = []
    for (account, user), m in usage_by_acct_user.items():
        dump_list.append({
            "account": account,
            "user": user,
            "cpu_core_hours": m["cpu"],
            "a100_hours": m["a100"],
            "l40s_hours": m["l40s"],
            "v100_hours": m["v100"],
        })
    json.dump(dump_list, open(usage_file, "w"), indent=2)
    log(f"[✓] Saved usage dump → {usage_file}")

    # Which PIs to process
    if target_pi:
        if target_pi not in pi_accounts:
            log(f"[!] PI '{target_pi}' not found in pi_accounts.json")
            return
        pis = [target_pi]
        log(f"[→] Running for single PI: {target_pi}")
    else:
        pis = list(pi_accounts.keys())
        log(f"[→] Running full weekly report for {len(pis)} PIs")

    sent_count = 0
    skipped_no_usage = 0

    for pi in pis:
        accounts = pi_accounts.get(pi, [])
        email = pi_emails.get(pi)

        if not email:
            log(f"[!] No email found for {pi}, skipping.")
            continue

        # Check if this PI has any usage at all
        has_usage = False
        for (account, user), m in usage_by_acct_user.items():
            if account in accounts and (
                m["cpu"] > 0 or m["a100"] > 0 or m["l40s"] > 0 or m["v100"] > 0
            ):
                has_usage = True
                break

        if not has_usage:
            log(f"[→] Skipping {pi}: no CPU/GPU usage this week.")
            skipped_no_usage += 1
            continue

        report_body = format_pi_report(
            pi,
            accounts,
            usage_by_acct_user,
            start=start,
            end=end,
        )

        # Determine recipient + subject
        if test_run:
            recipient = "help@arch.jhu.edu"
            subject = f"[TEST] Rockfish Weekly Usage Report for {pi} ({start} → {end})"
            log(f"[→] TEST-RUN: sending {pi}'s report to {recipient}")
        else:
            recipient = email
            subject = f"[Rockfish] Weekly Usage Report ({start} → {end})"

        send_email(
            recipient,
            subject,
            report_body,
            dry_run=dry_run,
            archive_label=pi,
            start=start,
            end=end,
        )
        sent_count += 1

        # Only send one test message when using --test-run
        if test_run:
            break

    # Build + send admin summary
    summary = admin_summary_report(
        start, end,
        sent_count, skipped_no_usage,
        usage_by_acct_user,
    )

    if dry_run:
        print("\n[DRY RUN] --- Admin Summary Email ---")
        print(summary)
        print("-------------------------------------\n")

        # Preserve old behavior: send admin summary even in dry-run mode
        send_email(
            "help@arch.jhu.edu",
            f"[Rockfish] Weekly Usage Summary ({start} → {end})",
            summary,
            dry_run=False,
            archive_label="admin_summary",
            start=start,
            end=end,
        )
        archive_admin_summary(start, end, summary)
        log("[✓] Admin summary email SENT (even in dry-run mode).")
        return

    if not test_run:
        send_email(
            "help@arch.jhu.edu",
            f"[Rockfish] Weekly Usage Summary ({start} → {end})",
            summary,
            dry_run=False,
            archive_label="admin_summary",
            start=start,
            end=end,
        )
        archive_admin_summary(start, end, summary)
        log("\n[✓] Completed weekly report generation.")
        log(f"    Sent {sent_count} report(s), {skipped_no_usage} had no usage.")
    else:
        log("\n[✓] Test run complete — PI email sent only to help@arch.jhu.edu.")
        log("    (Admin summary NOT sent during --test-run)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate and email Rockfish weekly CPU/GPU usage reports using sacct, driven by cluster_config.yaml."
    )
    parser.add_argument("--pi", help="Limit to one PI for testing.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print output instead of sending email (admin summary still sent).",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Send the selected PI's report to help@arch.jhu.edu only.",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, target_pi=args.pi, test_run=args.test_run)

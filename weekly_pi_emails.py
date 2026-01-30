#!/usr/bin/env python3
import os
import sys
import json
import smtplib
import subprocess
import time
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
BASE_DIR = Path("/data/rfadmin/mprotzm2_scripts/sreport_weekly")
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
ADMIN_EMAIL = None  # comes from CONFIG["email"]["admin_email"]
CLUSTER_NAME = None
LDAP_BASE_DN = None

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
    for key in [
        "gpu_type_map",
        "gpu_types",
        "billing_rates",
        "email",
        "commands",
        "pi_account_grouping",
        "cluster_name",
        "ldap_base_dn",
    ]:
        if key not in cfg:
            config_error(f"Missing required top-level key: {key}")

    # Email section
    email_cfg = cfg["email"]
    if "sender" not in email_cfg:
        config_error("email.sender is required.")
    if "admin_email" not in email_cfg:
        config_error("email.admin_email is required.")
    if "signature" not in email_cfg or not isinstance(email_cfg["signature"], list):
        config_error("email.signature must be a list of lines.")
    if "send_delay_seconds" in email_cfg:
        try:
            float(email_cfg["send_delay_seconds"])
        except (TypeError, ValueError):
            config_error("email.send_delay_seconds must be a number (seconds).")
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
            return subprocess.check_output(cmd, shell=True, universal_newlines=True).strip()
        else:
            return subprocess.check_output(cmd, universal_newlines=True).strip()
    except subprocess.CalledProcessError as e:
        log(f"[!] Command failed: {cmd}\n{e}")
        return ""

# =============================
# Date helper
# =============================
def get_last_week_range():
    """
    Return Friday–Thursday date strings for the most recent completed week.

    Intended usage: script runs on Friday morning and reports on the
    previous Friday through Thursday (inclusive).
    """
    today = date.today()

    # Python weekday(): Monday=0 ... Sunday=6
    # Friday = 4
    days_since_friday = (today.weekday() - 4) % 7

    last_friday = today - timedelta(days=days_since_friday + 7)
    last_thursday = last_friday + timedelta(days=6)

    return str(last_friday), str(last_thursday)

# =============================
# PI account builders (YAML-driven)
# =============================
def build_unix_group_roster():
    """
    Returns:
      group_to_users: dict[groupname] = set(usernames)
      group_to_gid:   dict[groupname] = int(gid)
      gid_to_group:   dict[gid] = groupname

    Uses:
      - getent group  (supplementary members)
      - getent passwd (primary gid membership)
    """
    group_to_users = defaultdict(set)
    group_to_gid = {}
    gid_to_group = {}

    # 1) Supplementary members: getent group
    out = run_cmd(["getent", "group"])
    for line in out.splitlines():
        # name:passwd:gid:member1,member2,...
        parts = line.split(":", 3)
        if len(parts) < 3:
            continue
        name = parts[0].strip()
        try:
            gid = int(parts[2].strip())
        except ValueError:
            continue

        group_to_gid[name] = gid
        gid_to_group[gid] = name

        members = parts[3].strip() if len(parts) == 4 else ""
        if members:
            for u in members.split(","):
                u = u.strip()
                if u:
                    group_to_users[name].add(u)

    # 2) Primary members: getent passwd (users whose primary gid == group's gid)
    out = run_cmd(["getent", "passwd"])
    for line in out.splitlines():
        # user:passwd:uid:gid:gecos:home:shell
        parts = line.split(":")
        if len(parts) < 4:
            continue
        user = parts[0].strip()
        try:
            gid = int(parts[3].strip())
        except ValueError:
            continue
        gname = gid_to_group.get(gid)
        if gname:
            group_to_users[gname].add(user)

    return group_to_users, group_to_gid

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
    cmd = f"ldapsearch -x -LLL -b '{CONFIG['ldap_base_dn']}' '(uid=*)' uid mail"
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
    Use scontrol show partition to discover which partitions are GPU partitions.
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
        "-S",
        start,
        "-E",
        end,
        "-a",
        "-X",
        "-P",
        "-n",
        "-o",
        "account,user,partition,elapsedraw,alloctres",
    ]
    output = run_cmd(cmd)
    if not output:
        log("[!] sacct returned no data.")
        return {}

    # Build dynamic GPU buckets from YAML config
    gpu_buckets = {t.lower(): 0.0 for t in CONFIG["gpu_types"]}
    usage = defaultdict(lambda: {"cpu": 0.0, **gpu_buckets})

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

def format_pi_report(pi, accounts, usage_by_acct_user, start=None, end=None, unix_group_users=None):
    """
    Build the full plain-text report for a PI.

    Behavior change:
      - PI email is still only sent if PI has >0 total usage (handled outside this function).
      - If the PI email is sent, the Per-user Breakdown will list ALL users in the PI's roster,
        even if their usage is 0.00, plus any additional users who had usage but aren't in the roster.

    unix_group_users:
      dict[groupname] -> list[str] of usernames (built once via getent group/passwd and cached)
      If None or group missing, we fall back to "users observed in sacct" only.
    """
    global CLUSTER_NAME
    start = start or "this week"
    end = end or "today"

    gpu_types = [t.lower() for t in CONFIG["gpu_types"]]

    # --- 1) Build full roster for this PI (from unix group), then overlay usage ---
    roster = []
    if unix_group_users and isinstance(unix_group_users, dict):
        roster = unix_group_users.get(pi, []) or []
    roster_set = set(roster)

    # Initialize per_user with ALL roster users at 0.00
    per_user = {u: {"cpu": 0.0, **{t: 0.0 for t in gpu_types}} for u in roster}

    # Overlay actual usage from sacct (and include any "job users" not in roster)
    accounts_set = set(accounts)
    for (account, user), metrics in usage_by_acct_user.items():
        if account not in accounts_set:
            continue

        if user not in per_user:
            per_user[user] = {"cpu": 0.0, **{t: 0.0 for t in gpu_types}}

        per_user[user]["cpu"] += metrics.get("cpu", 0.0)
        for t in gpu_types:
            per_user[user][t] += metrics.get(t, 0.0)

    # If roster exists, keep roster users first (alpha), then any extra observed users (alpha)
    if roster:
        extra_users = sorted([u for u in per_user.keys() if u not in roster_set])
        ordered_users = sorted(roster) + extra_users
    else:
        ordered_users = sorted(per_user.keys())

    # --- 2) Totals across all users / accounts ---
    total_cpu = sum(v["cpu"] for v in per_user.values())
    total_gpu = {t: sum(v[t] for v in per_user.values()) for t in gpu_types}

    # --- 3) Summary block (only non-zero fields) ---
    summary_lines = ["All Accounts Combined:"]
    if total_cpu > 0:
        summary_lines.append(f"CPU Hours:  {fmt_hours(total_cpu)}")
    for t in gpu_types:
        if total_gpu[t] > 0:
            summary_lines.append(f"{t.upper()} Hours: {fmt_hours(total_gpu[t])}")
    summary_block = "\n".join(summary_lines)

    # --- 4) Per-user breakdown ---
    # Show only users with >0 in each section, and roll up "all-zero" users at the bottom.

    # Determine which users are zero across ALL tracked resources
    zero_users = []
    for u in ordered_users:
        total_u = per_user[u]["cpu"] + sum(per_user[u][t] for t in gpu_types)
        if total_u <= 0:
            zero_users.append(u)

    user_lines = []
    user_lines.append("\nPer-user Breakdown:\n")

    # CPU section (only include section if PI total CPU > 0)
    if total_cpu > 0:
        user_lines.append("CPU Hours:")
        any_cpu_lines = False
        for user in ordered_users:
            v = per_user[user]["cpu"]
            if v > 0:
                user_lines.append(f"  → {user}: {fmt_hours(v)}")
                any_cpu_lines = True
        if not any_cpu_lines:
            user_lines.append("  (none)")
        user_lines.append("")

    # GPU sections (only include a GPU type section if PI total for that type > 0)
    for t in gpu_types:
        if total_gpu[t] > 0:
            user_lines.append(f"{t.upper()} Hours:")
            any_gpu_lines = False
            for user in ordered_users:
                v = per_user[user][t]
                if v > 0:
                    user_lines.append(f"  → {user}: {fmt_hours(v)}")
                    any_gpu_lines = True
            if not any_gpu_lines:
                user_lines.append("  (none)")
            user_lines.append("")

    # Roll-up section for users with zero usage across all tracked resources
    if zero_users:
        user_lines.append("The following users had no usage for this period:")
        for u in zero_users:
            user_lines.append(f"  → {u}")
        user_lines.append("")

    user_block = "\n".join(user_lines).rstrip()

    body = f"""Dear User,

This report summarizes CPU and GPU usage for your research group on the {CLUSTER_NAME} cluster from {start} through {end}.

The first section aggregates all of your associated Slurm accounts. The second section breaks the same usage down by individual user login.

{CLUSTER_NAME} Weekly Usage Report for {pi}
=====================================

{summary_block}

{user_block}

If you have any questions about your group’s usage or notice discrepancies, please reach out to {CONFIG['email']['sender']}.

{signature_block()}
"""
    return body

# =============================
# User report helpers (per-user usage grouped by PI)
# =============================
def build_account_to_pi(pi_accounts: dict) -> dict:
    """
    Build a reverse map from Slurm account -> PI root.
    pi_accounts: {pi_root: [acct1, acct2, ...]}
    """
    acct_to_pi = {}
    for pi, accts in pi_accounts.items():
        for a in accts:
            acct_to_pi.setdefault(a, pi)
    return acct_to_pi

def compute_user_usage_by_pi(usage_by_acct_user: dict, acct_to_pi: dict) -> dict:
    """
    Return nested mapping:
      user_usage[user][pi_root] = {"cpu": X, "<gpu_type>": Y, ...}

    Only includes accounts that map to a PI root in pi_accounts.json.
    """
    gpu_types = [t.lower() for t in CONFIG["gpu_types"]]
    user_usage = defaultdict(lambda: defaultdict(lambda: {"cpu": 0.0, **{t: 0.0 for t in gpu_types}}))

    for (acct, user), metrics in usage_by_acct_user.items():
        pi = acct_to_pi.get(acct)
        if not pi:
            continue
        user_usage[user][pi]["cpu"] += metrics.get("cpu", 0.0)
        for t in gpu_types:
            user_usage[user][pi][t] += metrics.get(t, 0.0)

    return user_usage

def has_any_usage(metrics_by_pi: dict) -> bool:
    """metrics_by_pi: {pi: {cpu, gpu...}}"""
    gpu_types = [t.lower() for t in CONFIG["gpu_types"]]
    for _, m in metrics_by_pi.items():
        if m.get("cpu", 0.0) > 0:
            return True
        for t in gpu_types:
            if m.get(t, 0.0) > 0:
                return True
    return False

def format_user_report(user: str, usage_by_pi: dict, start: str, end: str) -> str:
    """
    Build a per-user report: this user's usage grouped by PI root (group).
    Only prints non-zero sections/lines.
    """
    gpu_types = [t.lower() for t in CONFIG["gpu_types"]]
    if not usage_by_pi or not has_any_usage(usage_by_pi):
        return ""

    title = f"{CLUSTER_NAME} Weekly Usage Report for {user}"
    lines = []
    lines.append(f"Dear {user},\n")
    lines.append(
        f"This report summarizes your CPU and GPU usage on the {CLUSTER_NAME} cluster "
        f"from {start} through {end}, grouped by research group (PI account owner).\n"
    )
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    # CPU section
    cpu_lines = []
    for pi in sorted(usage_by_pi.keys()):
        v = usage_by_pi[pi].get("cpu", 0.0)
        if v > 0:
            cpu_lines.append(f"  → {pi} - {user}: {fmt_hours(v)}")
    if cpu_lines:
        lines.append("CPU Hours:")
        lines.extend(cpu_lines)
        lines.append("")

    # GPU sections
    for t in gpu_types:
        t_lines = []
        for pi in sorted(usage_by_pi.keys()):
            v = usage_by_pi[pi].get(t, 0.0)
            if v > 0:
                t_lines.append(f"  → {pi} - {user}: {fmt_hours(v)}")
        if t_lines:
            lines.append(f"{t.upper()} Hours:")
            lines.extend(t_lines)
            lines.append("")

    lines.append("If you have any questions or notice discrepancies, please reach out to help@arch.jhu.edu.")
    lines.append("")
    lines.append(signature_block())

    return "\n".join(lines).rstrip() + "\n"

def get_test_sink_email() -> str:
    """
    Where to route test messages.
    Prefer a dedicated helpdesk_email if present, otherwise fall back to admin_email.
    """
    email_cfg = CONFIG.get("email", {})
    return email_cfg.get("helpdesk_email") or email_cfg.get("admin_email") or email_cfg.get("sender")

def send_user_reports(
    start,
    end,
    usage_by_acct_user,
    pi_accounts,
    uid_to_email,
    dry_run=False,
    test_run=False,
    target_user=None,
):
    """
    Send per-user reports to each user with non-zero usage.

    - If target_user is provided, only that user's report is considered.
    - If test_run is True, messages are routed to the configured test sink
      (helpdesk_email/admin_email) and only ONE message is sent.
    """
    acct_to_pi = build_account_to_pi(pi_accounts)
    user_usage = compute_user_usage_by_pi(usage_by_acct_user, acct_to_pi)

    users = [target_user] if target_user else sorted(user_usage.keys())
    test_sink = get_test_sink_email()

    sent = 0
    skipped_no_usage = 0
    skipped_no_email = 0

    for user in users:
        usage_by_pi = user_usage.get(user, {})
        body = format_user_report(user, usage_by_pi, start, end)
        if not body:
            skipped_no_usage += 1
            continue

        real_to = uid_to_email.get(user)
        if not real_to:
            skipped_no_email += 1
            log(f"[!] No email found for user '{user}' in LDAP map; skipping.")
            continue

        if test_run:
            to_addr = test_sink
            subject = f"[TEST] {CLUSTER_NAME} Weekly Usage Report for {user} ({start} → {end})"
            body = f"(TEST RUN) Intended recipient: {real_to}\n\n" + body
            log(f"[→] TEST-RUN: sending {user}'s report to {to_addr}")
        else:
            to_addr = real_to
            subject = f"[{CLUSTER_NAME}] Your Weekly Usage Report ({start} → {end})"

        send_email(
            to_addr,
            subject,
            body,
            dry_run=dry_run,
            archive_label=f"user_{user}",
            start=start,
            end=end,
        )
        sent += 1

        if test_run:
            break

    return sent, skipped_no_usage, skipped_no_email

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

    # Throttle outbound mail to avoid overwhelming SMTP
    delay = CONFIG.get("email", {}).get("send_delay_seconds", 0)
    if delay and delay > 0:
        time.sleep(float(delay))

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
def admin_summary_report(start, end, sent_count, skipped_no_usage, usage_by_acct_user):
    global CLUSTER_NAME
    gpu_types = [t.lower() for t in CONFIG["gpu_types"]]

    lines = []
    lines.append(f"{CLUSTER_NAME} Weekly Usage Summary\n")
    lines.append(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Date range: {start} → {end}\n")

    # Aggregate totals
    admin_total_hours = 0.0
    admin_account_usage = defaultdict(float)
    admin_user_usage = defaultdict(float)

    for (account, user), m in usage_by_acct_user.items():
        total = m["cpu"] + sum(m[t] for t in gpu_types)
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

def user_summary_report(start, end, user_sent, user_skipped_no_usage, user_skipped_no_email, usage_by_acct_user, pi_accounts):
    """
    Summary for the per-user email run.
    Reports how many user emails would/were sent, and top users by total usage.
    """
    gpu_types = [t.lower() for t in CONFIG["gpu_types"]]
    acct_to_pi = build_account_to_pi(pi_accounts)

    # Aggregate totals by user and by (pi,user)
    total_hours = 0.0
    user_totals = defaultdict(float)
    user_pi_totals = defaultdict(float)

    for (acct, user), m in usage_by_acct_user.items():
        pi = acct_to_pi.get(acct, "unknown")
        subtotal = m["cpu"] + sum(m[t] for t in gpu_types)
        if subtotal <= 0:
            continue
        total_hours += subtotal
        user_totals[user] += subtotal
        user_pi_totals[(user, pi)] += subtotal

    lines = []
    lines.append(f"{CLUSTER_NAME} Weekly User Email Summary\n")
    lines.append(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Date range: {start} → {end}\n")

    lines.append(f"Total compute hours used (CPU + GPU-hours): {total_hours:.2f}\n")

    lines.append("User Email Delivery:")
    lines.append(f"  User emails sent/would send: {user_sent}")
    lines.append(f"  Skipped (no usage): {user_skipped_no_usage}")
    lines.append(f"  Skipped (no email in LDAP): {user_skipped_no_email}\n")

    lines.append("----------------------------------------")
    lines.append("Top 10 Users by Usage (combined CPU+GPU)")
    lines.append("----------------------------------------")
    top_users = sorted(user_totals.items(), key=lambda x: x[1], reverse=True)[:10]
    for user, hrs in top_users:
        lines.append(f"{user:<12} {hrs:>12.2f} hrs")

    lines.append("\n----------------------------------------")
    lines.append("Top 10 (User, Group) pairs by Usage")
    lines.append("----------------------------------------")
    top_pairs = sorted(user_pi_totals.items(), key=lambda x: x[1], reverse=True)[:10]
    for (user, pi), hrs in top_pairs:
        lines.append(f"{user:<12} {pi:<20} {hrs:>12.2f} hrs")

    lines.append("\n-- End of user summary --\n")
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
def main(dry_run=False, target_pi=None, test_run=False, user_emails=False, users_only=False, target_user=None):
    global CONFIG, SENDER, ADMIN_EMAIL, CLUSTER_NAME, LDAP_BASE_DN
    CONFIG = load_config()
    SENDER = CONFIG["email"]["sender"]
    ADMIN_EMAIL = CONFIG["email"]["admin_email"]
    CLUSTER_NAME = CONFIG["cluster_name"]
    LDAP_BASE_DN = CONFIG["ldap_base_dn"]

    ensure_fresh_data()
    pi_accounts = json.load(open(PI_ACCOUNTS_JSON))
    pi_emails = json.load(open(PI_EMAILS_JSON))

    # Build Unix group roster once (for PI report "include all users even if 0")
    unix_group_users, _unix_group_gids = build_unix_group_roster()
    log(f"[✓] Loaded Unix group roster for {len(unix_group_users)} groups via getent.")

    start, end = get_last_week_range()
    usage_by_acct_user = get_sacct_usage(start, end)

    # Dump usage for debug/record
    usage_file = DATA_DIR / f"usage_{start}_{end}.json"
    dump_list = []
    gpu_types = [t.lower() for t in CONFIG["gpu_types"]]
    for (account, user), m in usage_by_acct_user.items():
        entry = {
            "account": account,
            "user": user,
            "cpu_core_hours": m["cpu"],
        }
        for t in gpu_types:
            entry[f"{t}_hours"] = m[t]
        dump_list.append(entry)

    json.dump(dump_list, open(usage_file, "w"), indent=2)
    log(f"[✓] Saved usage dump → {usage_file}")

    sent_count = 0
    skipped_no_usage = 0

    if not users_only:
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

        for pi in pis:
            accounts = pi_accounts.get(pi, [])
            accounts_set = set(accounts)
            email = pi_emails.get(pi)

            if not email:
                log(f"[!] No email found for {pi}, skipping.")
                continue

            # Check if this PI has any usage at all
            has_usage = False
            for (account, user), m in usage_by_acct_user.items():
                if account in accounts_set and (m["cpu"] > 0 or any(m[t] > 0 for t in gpu_types)):
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
                unix_group_users=unix_group_users,
            )

            # Determine recipient + subject
            if test_run:
                recipient = get_test_sink_email()
                subject = f"[TEST] {CLUSTER_NAME} Weekly Usage Report for {pi} ({start} → {end})"
                log(f"[→] TEST-RUN: sending {pi}'s report to {recipient}")
            else:
                recipient = email
                subject = f"[{CLUSTER_NAME}] Weekly Usage Report ({start} → {end})"

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

    # Optional: per-user reports (each user with non-zero usage, grouped by PI)
    user_sent = 0
    user_skipped_no_usage = 0
    user_skipped_no_email = 0
    if user_emails or users_only:
        user_sent, user_skipped_no_usage, user_skipped_no_email = send_user_reports(
            start,
            end,
            usage_by_acct_user,
            pi_accounts,
            pi_emails,
            dry_run=dry_run,
            test_run=test_run,
            target_user=target_user,
        )
        log(
            f"[✓] User email run: sent={user_sent}, "
            f"skipped_no_usage={user_skipped_no_usage}, "
            f"skipped_no_email={user_skipped_no_email}"
        )
    # Build summary (PI summary for PI runs; user summary for users-only runs; both if mixed)
    summaries = []

    if not users_only:
        summaries.append(
            admin_summary_report(
                start,
                end,
                sent_count,
                skipped_no_usage,
                usage_by_acct_user,
            )
        )

    if user_emails or users_only:
        summaries.append(
            user_summary_report(
                start,
                end,
                user_sent,
                user_skipped_no_usage,
                user_skipped_no_email,
                usage_by_acct_user,
                pi_accounts,
            )
        )

    summary = "\n\n".join(summaries).strip() + "\n"

    if dry_run:
        print("\n[DRY RUN] --- Admin Summary Email ---")
        print(summary)
        print("-------------------------------------\n")

        # Preserve old behavior: send admin summary even in dry-run mode
        send_email(
            ADMIN_EMAIL,
            f"[{CLUSTER_NAME}] Weekly Usage Summary ({start} → {end})",
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
            ADMIN_EMAIL,
            f"[{CLUSTER_NAME}] Weekly Usage Summary ({start} → {end})",
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
        log("\n[✓] Test run complete — report routed to configured test sink (helpdesk_email/admin_email).")
        log("    (Admin summary NOT sent during --test-run)")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate and email weekly CPU/GPU usage reports using sacct, driven by cluster_config.yaml."
    )

    parser.add_argument(
        "--run-pi-emails",
        action="store_true",
        help="Execute a PI email run (weekly PI usage reports).",
    )
    parser.add_argument(
        "--run-user-emails",
        action="store_true",
        help="Execute a per-user email run (weekly user usage reports).",
    )

    parser.add_argument("--pi", help="Limit PI reports to one PI for testing.")
    parser.add_argument("--user", dest="target_user", help="Limit user reports to one specific user (uid).")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print output instead of sending PI/user emails (admin summary still sent).",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Route outgoing PI/user reports to the configured test sink and send only ONE message.",
    )

    args = parser.parse_args()

    # Require an explicit execution mode so this can't be run accidentally
    if not args.run_pi_emails and not args.run_user_emails:
        parser.print_help()
        print("\nERROR: You must specify at least one execution mode:")
        print("  --run-pi-emails")
        print("  --run-user-emails")
        sys.exit(2)

    main(
        dry_run=args.dry_run,
        target_pi=args.pi,
        test_run=args.test_run,
        user_emails=args.run_user_emails,                        
        users_only=(args.run_user_emails and not args.run_pi_emails),
        target_user=args.target_user,
    )

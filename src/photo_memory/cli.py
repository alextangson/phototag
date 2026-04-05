"""CLI entry point for phototag."""

import logging
import os
import shutil
import subprocess
import tempfile

import click

from photo_memory.config import load_config
from photo_memory.db import Database
from photo_memory.dedup import find_duplicate_groups
from photo_memory.events import build_events
from photo_memory.load_monitor import LoadMonitor
from photo_memory.processor import process_batch
from photo_memory.scanner import scan_photos_into_db

logger = logging.getLogger("photo_memory")


REQUIRED_PERMISSIONS = [
    {
        "name": "照片图库访问",
        "key": "photos_library",
        "setting": "系统设置 → 隐私与安全性 → 照片",
    },
    {
        "name": "Apple Photos 自动化控制",
        "key": "apple_events",
        "setting": "系统设置 → 隐私与安全性 → 自动化 → Python → 照片",
    },
    {
        "name": "完全磁盘访问（读取照片库数据库）",
        "key": "full_disk",
        "setting": "系统设置 → 隐私与安全性 → 完全磁盘访问",
    },
    {
        "name": "Ollama 服务可用",
        "key": "ollama",
        "setting": "确保 Ollama 已启动: ollama serve",
    },
]


def _check_photos_library() -> tuple[bool, str]:
    """Check if we can access the Photos library."""
    try:
        import osxphotos
        db = osxphotos.PhotosDB()
        count = len(db.photos(images=True, movies=False))
        return True, f"可访问，共 {count} 张照片"
    except Exception as e:
        return False, f"无法访问: {e}"


def _check_apple_events() -> tuple[bool, str]:
    """Check if we can control Photos via AppleScript."""
    try:
        result = subprocess.run(
            ["osascript", "-e", 'tell application "Photos" to return name'],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0:
            return True, f"可控制 Photos App"
        return False, f"AppleScript 失败: {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, "超时 — 可能卡在权限弹窗"
    except Exception as e:
        return False, str(e)


def _check_full_disk() -> tuple[bool, str]:
    """Check full disk access by reading Photos library DB path."""
    photos_db = os.path.expanduser("~/Pictures/Photos Library.photoslibrary/database/Photos.sqlite")
    if os.path.isfile(photos_db):
        try:
            with open(photos_db, "rb") as f:
                f.read(16)
            return True, "可读取照片库数据库"
        except PermissionError:
            return False, "无权限读取照片库数据库文件"
    return True, "照片库路径不在默认位置（osxphotos 会自动定位）"


def _check_ollama(host: str) -> tuple[bool, str]:
    """Check if Ollama is reachable."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                return True, "Ollama 服务正常"
        return False, f"HTTP {resp.status}"
    except Exception as e:
        return False, f"无法连接 {host}: {e}"


def run_preflight(ollama_host: str = "http://localhost:11434",
                  interactive: bool = True) -> bool:
    """Run all permission checks. Returns True if all passed."""
    checks = {
        "photos_library": _check_photos_library,
        "apple_events": _check_apple_events,
        "full_disk": _check_full_disk,
        "ollama": lambda: _check_ollama(ollama_host),
    }

    all_passed = True
    results = []

    for perm in REQUIRED_PERMISSIONS:
        ok, msg = checks[perm["key"]]()
        status = "✅" if ok else "❌"
        results.append((status, perm["name"], msg, perm["setting"]))
        if not ok:
            all_passed = False

    if interactive:
        click.echo("\n🔍 权限预检:")
        click.echo("-" * 60)
        for status, name, msg, setting in results:
            click.echo(f"  {status} {name}")
            click.echo(f"     {msg}")
            if status == "❌":
                click.echo(f"     👉 {setting}")
        click.echo("-" * 60)

        if all_passed:
            click.echo("  ✅ 所有权限检查通过!\n")
        else:
            click.echo("  ❌ 部分权限未通过，请先授权再运行。\n")

    if not interactive and not all_passed:
        for status, name, msg, setting in results:
            if status == "❌":
                logger.error(f"权限检查失败 [{name}]: {msg} → {setting}")

    return all_passed


def _setup_logging():
    log_path = os.path.expanduser("~/Library/Logs/phototag.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )


@click.group()
@click.option("--config", "config_path", default=None, help="Path to config.yaml")
@click.pass_context
def main(ctx, config_path):
    """phototag — Local AI photo tagger for Apple Photos."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)
    _setup_logging()


@main.command()
@click.pass_context
def preflight(ctx):
    """Check all required permissions before first run."""
    config = ctx.obj["config"]
    ok = run_preflight(ollama_host=config["ollama"]["host"], interactive=True)
    raise SystemExit(0 if ok else 1)


@main.command()
@click.pass_context
def scan(ctx):
    """Scan Photos library and populate the database."""
    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)
    count = scan_photos_into_db(db)
    click.echo(f"Scan complete. {count} new photos added to database.")
    db.close()


@main.command()
@click.pass_context
def run(ctx):
    """Run the nightly processing loop."""
    config = ctx.obj["config"]

    if not run_preflight(ollama_host=config["ollama"]["host"], interactive=False):
        logger.error("权限预检失败，终止运行。请先执行 phototag preflight 检查并授权。")
        raise SystemExit(1)

    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)

    scan_photos_into_db(db)

    monitor = LoadMonitor(
        max_memory_pressure=config["load"]["max_memory_pressure"],
        min_cpu_idle=config["load"]["min_cpu_idle"],
    )

    with tempfile.TemporaryDirectory(prefix="phototag-") as tmp_dir:
        run_id = db.start_run()
        try:
            stats = process_batch(
                db=db,
                monitor=monitor,
                check_interval=config["schedule"]["check_interval"],
                ollama_config=config["ollama"],
                tmp_dir=tmp_dir,
                end_hour=config["schedule"]["end_hour"],
                start_hour=config["schedule"]["start_hour"],
            )
            db.end_run(run_id, stats["processed"], stats.get("skipped", 0),
                       stats["errored"], stats["stop_reason"])
            click.echo(f"Run complete: {stats['processed']} processed, "
                       f"{stats['errored']} errors. Stop reason: {stats['stop_reason']}")
        except Exception as e:
            db.end_run(run_id, 0, 0, 0, f"error: {e}")
            logger.error(f"Run failed: {e}")
            raise

    db.close()


@main.command()
@click.pass_context
def dedup(ctx):
    """Run duplicate detection on all processed photos."""
    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)

    phashes = db.get_all_phashes()
    click.echo(f"Checking {len(phashes)} photos for duplicates...")

    groups = find_duplicate_groups(phashes, threshold=config["dedup"]["hash_threshold"])
    click.echo(f"Found {len(groups)} duplicate groups.")

    for i, group in enumerate(groups):
        for uuid in group:
            db.add_duplicate_pair(group_id=i, photo_uuid=uuid, similarity=1.0)

    click.echo("Duplicate groups saved to database.")
    db.close()


@main.command()
@click.pass_context
def reprocess(ctx):
    """Reset all processed photos back to pending for reprocessing with new prompt."""
    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)
    count = db.reset_photos_for_reprocess()
    click.echo(f"Reset {count} photos to pending. Run 'phototag run' to reprocess.")
    db.close()


@main.command()
@click.option("--gap-minutes", default=30, help="Time gap (minutes) to split events")
@click.pass_context
def events(ctx, gap_minutes):
    """Build events from done photos and list them."""
    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)

    count = build_events(db, ollama_config=config["ollama"], gap_minutes=gap_minutes)
    click.echo(f"Built {count} events.")

    all_events = db.get_all_events()
    click.echo(f"\n最近 10 个事件:")
    for e in all_events[:10]:
        city = e["location_city"] or "未知"
        click.echo(f"  [{e['start_time'][:10]}] {city} ({e['photo_count']} 张) — {e['summary'] or '(无摘要)'}")

    db.close()


@main.command()
@click.pass_context
def status(ctx):
    """Show processing progress."""
    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)

    row = db.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status='done' THEN 1 ELSE 0 END) as done,
            SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) as error,
            SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) as pending
        FROM photos
    """).fetchone()

    if row and row["total"] > 0:
        pct = row["done"] / row["total"] * 100
        click.echo(f"Total: {row['total']} | Done: {row['done']} ({pct:.1f}%) | "
                   f"Pending: {row['pending']} | Errors: {row['error']}")
    else:
        click.echo("No photos in database. Run 'phototag scan' first.")
    db.close()


@main.command()
@click.pass_context
def install(ctx):
    """Install launchd agent for nightly processing."""
    config = ctx.obj["config"]

    script_path = shutil.which("phototag")
    if not script_path:
        click.echo("Error: phototag not found in PATH. Run 'uv sync' first.")
        return

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.phototag.nightly</string>
    <key>ProgramArguments</key>
    <array>
        <string>{script_path}</string>
        <string>run</string>
    </array>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>{config['schedule']['start_hour']}</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{os.path.expanduser('~/Library/Logs/phototag-stdout.log')}</string>
    <key>StandardErrorPath</key>
    <string>{os.path.expanduser('~/Library/Logs/phototag-stderr.log')}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>"""

    plist_path = os.path.expanduser("~/Library/LaunchAgents/com.phototag.nightly.plist")
    os.makedirs(os.path.dirname(plist_path), exist_ok=True)

    with open(plist_path, "w") as f:
        f.write(plist_content)

    click.echo(f"LaunchAgent written to {plist_path}")
    click.echo("To activate:   launchctl load " + plist_path)
    click.echo("To deactivate: launchctl unload " + plist_path)

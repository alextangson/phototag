"""CLI entry point for PhotoMemory."""

import logging
import os
import shutil
import tempfile

import click

from photo_memory.config import load_config
from photo_memory.db import Database
from photo_memory.dedup import find_duplicate_groups
from photo_memory.load_monitor import LoadMonitor
from photo_memory.processor import process_batch
from photo_memory.scanner import scan_photos_into_db

logger = logging.getLogger("photo_memory")


def _setup_logging():
    log_path = os.path.expanduser("~/Library/Logs/photo-memory.log")
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
    """PhotoMemory — Local AI photo organizer."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)
    _setup_logging()


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
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)

    scan_photos_into_db(db)

    monitor = LoadMonitor(
        max_memory_pressure=config["load"]["max_memory_pressure"],
        min_cpu_idle=config["load"]["min_cpu_idle"],
    )

    with tempfile.TemporaryDirectory(prefix="photo-memory-") as tmp_dir:
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
        click.echo("No photos in database. Run 'photo-memory scan' first.")
    db.close()


@main.command()
@click.pass_context
def install(ctx):
    """Install launchd agent for nightly processing."""
    config = ctx.obj["config"]

    script_path = shutil.which("photo-memory")
    if not script_path:
        click.echo("Error: photo-memory not found in PATH. Run 'uv sync' first.")
        return

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.photomemory.nightly</string>
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
    <string>{os.path.expanduser('~/Library/Logs/photo-memory-stdout.log')}</string>
    <key>StandardErrorPath</key>
    <string>{os.path.expanduser('~/Library/Logs/photo-memory-stderr.log')}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>"""

    plist_path = os.path.expanduser("~/Library/LaunchAgents/com.photomemory.nightly.plist")
    os.makedirs(os.path.dirname(plist_path), exist_ok=True)

    with open(plist_path, "w") as f:
        f.write(plist_content)

    click.echo(f"LaunchAgent written to {plist_path}")
    click.echo("To activate:   launchctl load " + plist_path)
    click.echo("To deactivate: launchctl unload " + plist_path)

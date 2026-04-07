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
from photo_memory.people import build_people
from photo_memory.load_monitor import LoadMonitor
from photo_memory.processor import process_batch
from photo_memory.scanner import scan_photos_into_db
from photo_memory.story import (
    generate_person_story,
    generate_year_story,
    generate_relationship_story,
    build_year_html,
    build_person_html,
)
from photo_memory.search import search_photos as do_search, list_cleanup_candidates
from photo_memory.daily import run_daily_push
from photo_memory.merge import merge_clusters

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


def _ensure_ollama_running(host: str, max_wait: int = 30) -> bool:
    """Start Ollama if not running, wait up to max_wait seconds for it to be ready."""
    import time
    import urllib.request

    # Check if already running
    try:
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        pass

    # Try to start Ollama
    logger.info("Ollama 未运行，正在自动启动...")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        logger.error("ollama 未安装，无法自动启动。请先安装 Ollama。")
        return False

    # Wait for it to be ready
    for i in range(max_wait):
        time.sleep(1)
        try:
            req = urllib.request.Request(f"{host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3):
                logger.info(f"Ollama 已启动（等待了 {i+1}s）")
                return True
        except Exception:
            pass

    logger.error(f"Ollama 启动超时（等待 {max_wait}s）")
    return False


@main.command()
@click.option("--now", is_flag=True, default=False,
              help="Force processing now, ignore time window and load limits")
@click.option("--limit", type=int, default=None,
              help="Process at most N photos then stop")
@click.pass_context
def run(ctx, now, limit):
    """Run the processing loop."""
    config = ctx.obj["config"]
    ollama_host = config["ollama"]["host"]

    # Auto-start Ollama if needed (critical for unattended launchd runs)
    if not _ensure_ollama_running(ollama_host):
        logger.error("Ollama 无法启动，终止运行。")
        raise SystemExit(1)

    if not run_preflight(ollama_host=ollama_host, interactive=False):
        logger.error("权限预检失败，终止运行。请先执行 phototag preflight 检查并授权。")
        raise SystemExit(1)

    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)

    logger.info("开始扫描照片库...")
    scan_photos_into_db(db)

    # Check pending count before processing
    pending_row = db.execute("SELECT COUNT(*) as c FROM photos WHERE status='pending'").fetchone()
    logger.info(f"待处理照片: {pending_row['c']} 张")

    if now:
        # Force mode: completely disable load monitoring
        monitor = LoadMonitor(disabled=True)
    else:
        monitor = LoadMonitor(
            max_memory_pressure=config["load"]["max_memory_pressure"],
            min_cpu_idle=config["load"]["min_cpu_idle"],
        )

    if now:
        # Force mode: use permissive settings
        end_hour = 24  # never expire
        start_hour = 0
    else:
        end_hour = config["schedule"]["end_hour"]
        start_hour = config["schedule"]["start_hour"]

    with tempfile.TemporaryDirectory(prefix="phototag-") as tmp_dir:
        run_id = db.start_run()
        try:
            stats = process_batch(
                db=db,
                monitor=monitor,
                check_interval=limit or config["schedule"]["check_interval"],
                ollama_config=config["ollama"],
                tmp_dir=tmp_dir,
                end_hour=end_hour,
                start_hour=start_hour,
            )
            db.end_run(run_id, stats["processed"], stats.get("skipped", 0),
                       stats["errored"], stats["stop_reason"])
            logger.info(f"Run complete: {stats['processed']} processed, "
                        f"{stats['errored']} errors. Stop reason: {stats['stop_reason']}")
            click.echo(f"Run complete: {stats['processed']} processed, "
                       f"{stats['errored']} errors. Stop reason: {stats['stop_reason']}")
        except Exception as e:
            db.end_run(run_id, 0, 0, 0, f"error: {e}")
            logger.error(f"Run failed: {e}")
            raise

    # Post-processing: rebuild events + people + daily push
    if stats.get("processed", 0) > 0:
        try:
            logger.info("重建事件和人物图谱...")
            from photo_memory.events import build_events
            build_events(db, ollama_config=config["ollama"])
            build_people(db)
            logger.info("事件和人物图谱已更新")
        except Exception as e:
            logger.warning(f"事件/人物重建失败: {e}")

    # Daily memory push
    try:
        daily_dir = os.path.join(config["data_dir"], "daily")
        result = run_daily_push(db, config["ollama"], daily_dir)
        if result:
            logger.info(f"每日回忆已生成: {result}")
    except Exception as e:
        logger.warning(f"每日回忆生成失败: {e}")

    db.close()


@main.command()
@click.pass_context
def daily(ctx):
    """Generate today's memory push (On This Day / This Week)."""
    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)
    daily_dir = os.path.join(config["data_dir"], "daily")

    result = run_daily_push(db, config["ollama"], daily_dir)
    if result:
        click.echo(f"回忆已生成: {result}")
    else:
        click.echo("今天没有历史回忆可推送")
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
@click.option("--name", "name_args", nargs=2, type=str, default=None,
              help="Assign a user name: --name <face_cluster_id> <name>")
@click.option("--merge", "merge_args", nargs=3, type=str, default=None,
              help="Merge two clusters under one name: --merge <fc_a> <fc_b> <name>")
@click.option("--suggest-merges", is_flag=True, default=False,
              help="Suggest cluster pairs likely to be the same person")
@click.option("--min-photos", default=10, help="Minimum photo count to display")
@click.pass_context
def people(ctx, name_args, merge_args, suggest_merges, min_photos):
    """Build and list the people graph."""
    from photo_memory.merge import suggest_merges as compute_suggestions

    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)

    if merge_args:
        fc_a, fc_b, name = merge_args
        merge_clusters(db, fc_a, fc_b, name=name)
        click.echo(f"已将 {fc_a} 与 {fc_b} 合并为「{name}」")
        db.close()
        return

    if name_args:
        fc_id, user_name = name_args
        db.set_person_user_name(fc_id, user_name)
        click.echo(f"Named {fc_id} as {user_name}")
        db.close()
        return

    if suggest_merges:
        all_people = db.get_all_people()
        suggestions = compute_suggestions(all_people, min_photos=min_photos)
        click.echo(f"找到 {len(suggestions)} 对可能的合并候选:\n")
        for s in suggestions[:20]:
            click.echo(f"  jaccard={s['confidence']}  {s['fc_a'][:12]}... ↔ {s['fc_b'][:12]}...")
            click.echo(f"    共享联系人: {len(s['shared_contacts'])} 位")
            click.echo(f"    合并命令: phototag people --merge {s['fc_a']} {s['fc_b']} \"名字\"")
            click.echo("")
        db.close()
        return

    count = build_people(db)
    click.echo(f"Processed {count} unique people.\n")

    all_people = db.get_all_people()
    click.echo("高频人物:")
    shown = 0
    for p in all_people:
        if p["photo_count"] < min_photos:
            continue
        name = p["user_name"] or p["apple_name"] or "(未命名)"
        fc_short = p["face_cluster_id"][:12]
        first = (p["first_seen"] or "")[:7]
        last = (p["last_seen"] or "")[:7]
        trend = p["appearance_trend"] or ""
        click.echo(f"  {fc_short}... {name} — {p['photo_count']} 张, {first} ~ {last} [{trend}]")
        shown += 1
        if shown >= 20:
            break

    unnamed = [p for p in all_people if not p["user_name"] and not p["apple_name"] and p["photo_count"] >= min_photos]
    if unnamed:
        click.echo(f"\n发现 {len(unnamed)} 位高频未命名人物。运行以下命令命名：")
        for p in unnamed[:3]:
            click.echo(f"  phototag people --name {p['face_cluster_id']} \"名字\"")

    db.close()


@main.command()
@click.option("--person", "person_name", type=str, default=None,
              help="Generate a person's timeline story")
@click.option("--year", "year", type=int, default=None,
              help="Generate a year-in-review story")
@click.option("--relationship", "relationship_name", type=str, default=None,
              help="Generate a relationship-framed story")
@click.option("--output", "output_path", type=click.Path(), default=None,
              help="Write story to file instead of stdout (.md)")
@click.pass_context
def story(ctx, person_name, year, relationship_name, output_path):
    """Generate a Markdown story from L2 events and people."""
    modes = [person_name, year, relationship_name]
    mode_count = sum(1 for m in modes if m is not None)

    if mode_count == 0:
        click.echo("错误：需要指定 --person / --year / --relationship 之一", err=True)
        raise SystemExit(1)
    if mode_count > 1:
        click.echo("错误：--person / --year / --relationship 只能选一个", err=True)
        raise SystemExit(1)

    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)

    if person_name:
        if output_path and output_path.endswith(".html"):
            content = build_person_html(db, person_name, ollama_config=config["ollama"])
        else:
            content = generate_person_story(db, person_name, ollama_config=config["ollama"])
    elif year:
        if output_path and output_path.endswith(".html"):
            content = build_year_html(db, year, ollama_config=config["ollama"])
        else:
            content = generate_year_story(db, year, ollama_config=config["ollama"])
    else:
        content = generate_relationship_story(db, relationship_name, ollama_config=config["ollama"])

    db.close()

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        click.echo(f"Story written to {output_path}")
    else:
        click.echo(content)


@main.command()
@click.option("--person", type=str, default=None)
@click.option("--year", type=int, default=None)
@click.option("--city", type=str, default=None)
@click.option("--text", type=str, default=None)
@click.option("--limit", type=int, default=50)
@click.pass_context
def search(ctx, person, year, city, text, limit):
    """Search photos by person, year, city, or text keyword."""
    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)

    results = do_search(db, person=person, year=year, city=city, text=text, limit=limit)
    click.echo(f"找到 {len(results)} 张照片")
    for r in results[:limit]:
        date = (r.get("date_taken") or "")[:10]
        loc = r.get("location_city") or "-"
        desc = (r.get("description") or "")[:60]
        click.echo(f"  [{date}] {loc:<10s} {r['uuid'][:8]}... {desc}")
    db.close()


@main.command()
@click.option("--include-review", is_flag=True, default=False,
              help="Also list photos marked as 'review' (needs manual decision)")
@click.pass_context
def cleanup(ctx, include_review):
    """Report photos flagged for cleanup (cleanup_class=cleanup/review)."""
    config = ctx.obj["config"]
    db_path = os.path.join(config["data_dir"], "progress.db")
    db = Database(db_path)

    report = list_cleanup_candidates(db)
    cleanup_info = report["cleanup"]
    review_info = report["review"]

    click.echo(f"可清理照片：{cleanup_info['count']} 张")
    click.echo(f"待人工确认：{review_info['count']} 张")
    click.echo("")

    click.echo("=== cleanup （可直接删）===")
    for p in cleanup_info["photos"][:30]:
        desc = (p.get("description") or "")[:60]
        click.echo(f"  {p['uuid'][:8]}... {desc}")

    if include_review:
        click.echo("\n=== review （请人工判断）===")
        for p in review_info["photos"][:30]:
            desc = (p.get("description") or "")[:60]
            click.echo(f"  {p['uuid'][:8]}... {desc}")

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

    # Prefer PATH lookup, fall back to the venv that's currently executing us.
    script_path = shutil.which("phototag")
    if not script_path:
        import sys
        candidate = os.path.join(os.path.dirname(sys.executable), "phototag")
        if os.path.isfile(candidate):
            script_path = candidate
    if not script_path:
        click.echo("Error: phototag executable not found. Run 'pip install -e .' in the venv.")
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

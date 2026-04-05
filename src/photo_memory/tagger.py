"""Write tags, descriptions, and album assignments back to Apple Photos."""

import logging
import subprocess

logger = logging.getLogger(__name__)

SCREENSHOT_TAGS = {"截屏", "聊天记录"}


def _sanitize_for_applescript(s: str) -> str:
    """Sanitize a string for safe use in AppleScript double-quoted strings."""
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return s


def get_album_name(tag: str) -> str:
    """Extract first-level category from hierarchical tag and prefix with AI-."""
    first_level = tag.split("/")[0]
    return f"AI-{first_level}"


def get_special_album(result: dict) -> str | None:
    """Determine if photo belongs to a special album based on its AI result."""
    tag_prefixes = {t.split("/")[0] for t in result.get("tags", [])}
    is_screenshot_type = tag_prefixes & SCREENSHOT_TAGS
    importance = result.get("importance", "medium")

    if is_screenshot_type and importance == "high":
        return "AI-重要截图"
    elif is_screenshot_type and importance == "low":
        return "AI-截图待清理"
    return None


def _run_applescript(script: str) -> bool:
    """Run an AppleScript and return success status."""
    try:
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=30,
            check=True,
        )
        return True
    except Exception as e:
        logger.warning(f"AppleScript failed: {e}")
        return False


def _set_keywords(uuid: str, keywords: list[str]):
    """Set keywords on a photo using AppleScript."""
    kw_list = ", ".join(f'"{_sanitize_for_applescript(kw)}"' for kw in keywords)
    script = f'''
    tell application "Photos"
        set thePhoto to media item id "{uuid}"
        set keywords of thePhoto to {{{kw_list}}}
    end tell
    '''
    _run_applescript(script)


def _set_description(uuid: str, description: str):
    """Set description on a photo using AppleScript."""
    desc_escaped = _sanitize_for_applescript(description)
    script = f'''
    tell application "Photos"
        set thePhoto to media item id "{uuid}"
        set the description of thePhoto to "{desc_escaped}"
    end tell
    '''
    _run_applescript(script)


def _add_to_album(uuid: str, album_name: str):
    """Add a photo to an album, creating the album if needed."""
    safe_name = _sanitize_for_applescript(album_name)
    script = f'''
    tell application "Photos"
        -- Get or create album
        if not (exists album "{safe_name}") then
            make new album named "{safe_name}"
        end if
        set theAlbum to album "{safe_name}"
        -- Find and add photo
        set thePhoto to media item id "{uuid}"
        add {{thePhoto}} to theAlbum
    end tell
    '''
    _run_applescript(script)


def apply_tags_to_photo(uuid: str, result: dict):
    """Apply AI recognition results back to the photo in Apple Photos."""
    tags = result.get("tags", [])
    description = result.get("description", "")

    # Set keywords
    if tags:
        _set_keywords(uuid, tags)

    # Set description
    if description:
        _set_description(uuid, description)

    # Add to category albums
    album_names = set()
    for tag in tags:
        album_names.add(get_album_name(tag))

    special = get_special_album(result)
    if special:
        album_names.add(special)

    for album_name in album_names:
        _add_to_album(uuid, album_name)

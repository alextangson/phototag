# phototag

Local AI photo tagger for Apple Photos. Automatically recognizes, tags, describes, and organizes your entire photo library using on-device vision models. No cloud, no uploads, fully private.

## What it does

- Scans your Apple Photos library
- Sends each photo/video to a local [Ollama](https://ollama.com) vision model for recognition
- Writes back **keywords**, **descriptions**, and **smart albums** (prefixed `AI-`) directly into Apple Photos
- Extracts video frames + speech (via Whisper) for video tagging
- Detects duplicate photos using perceptual hashing
- Runs unattended overnight via launchd with adaptive load control

## Before & After

| Before | After |
|--------|-------|
| 10,000+ unsorted photos | Every photo tagged with Chinese keywords |
| No way to search by content | Search "sunset", "cat", "meeting" in Photos |
| Manual album creation | Auto-created `AI-` albums by category |

## Requirements

- macOS 13+ (Apple Silicon recommended)
- Python 3.12+
- [Ollama](https://ollama.com) with a vision model (default: `gemma4:e4b`)
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

```bash
# Clone and install
git clone https://github.com/alextangson/phototag.git
cd phototag
uv sync

# Pull the vision model
ollama pull gemma4:e4b

# Copy and customize config
mkdir -p ~/.phototag
cp examples/config.yaml ~/.phototag/config.yaml

# Check permissions (important — do this first!)
uv run phototag preflight

# Scan your library
uv run phototag scan

# Process photos (one-shot)
uv run phototag run

# Check progress
uv run phototag status
```

## Nightly Automation

```bash
# Install launchd agent (runs every night at 1 AM)
uv run phototag install

# Activate
launchctl load ~/Library/LaunchAgents/com.phototag.nightly.plist

# Deactivate
launchctl unload ~/Library/LaunchAgents/com.phototag.nightly.plist
```

The nightly job runs between configurable hours (default 1-7 AM) and automatically pauses when system load is high.

## Commands

| Command | Description |
|---------|-------------|
| `phototag preflight` | Check all required permissions |
| `phototag scan` | Scan Photos library into the database |
| `phototag run` | Process pending photos (recognize + tag + organize) |
| `phototag status` | Show processing progress |
| `phototag dedup` | Detect duplicate photos |
| `phototag install` | Install launchd agent for nightly runs |

## Configuration

Edit `~/.phototag/config.yaml`:

```yaml
ollama:
  host: "http://localhost:11434"
  model: "gemma4:e4b"       # any Ollama vision model
  timeout: 60

schedule:
  start_hour: 1              # nightly window start
  end_hour: 7                # nightly window end
  check_interval: 10         # batch size

load:
  max_memory_pressure: "warn" # warn | critical
  min_cpu_idle: 30            # pause if CPU idle < this %
```

## How It Works

```
Apple Photos ──scan──▶ SQLite DB ──process──▶ Ollama Vision ──tag──▶ Apple Photos
                          │                                            │
                          │                                     keywords, descriptions,
                          │                                     AI-* smart albums
                          │
                     phash dedup
```

1. **Scan** — reads your Photos library via [osxphotos](https://github.com/RhetTbull/osxphotos), stores metadata in a local SQLite database
2. **Recognize** — exports each photo to a temp file, sends to Ollama for AI analysis, parses structured JSON response
3. **Tag** — writes keywords, descriptions back to Apple Photos via AppleScript; creates `AI-` prefixed albums by category
4. **Video** — extracts keyframes + transcribes audio via Whisper, merges into a single tag set
5. **Dedup** — computes perceptual hashes, groups similar photos

## Permissions

phototag needs these macOS permissions (the `preflight` command checks all of them):

| Permission | Why | Where to grant |
|------------|-----|----------------|
| Photos Library | Read your photo library | System Settings → Privacy → Photos |
| Automation (Photos) | Write keywords/albums back | System Settings → Privacy → Automation |
| Full Disk Access | Read Photos database directly | System Settings → Privacy → Full Disk Access |

> **Tip**: Run `phototag preflight` interactively before setting up nightly automation. macOS cannot show permission dialogs during unattended launchd runs.

## License

Apache-2.0

#!/usr/bin/env bash
# preflight.sh — flush memory and kill background processes before training.
# Run with: bash scripts/preflight.sh
# Then start training with: python scripts/train_launch.py

set -e

echo "=== chatPDB Pre-Training Memory Flush ==="
echo ""

# 1. Kill orphaned Python processes (stale training runs holding memory)
echo "1. Checking for orphaned Python processes..."
ORPHANS=$(ps aux | grep -E "python.*mlx_lm|python.*train" | grep -v grep | grep -v $$ | awk '{printf "  PID %s — %.0f MB — %s\n", $2, $6/1024, $11}')
if [ -n "$ORPHANS" ]; then
    echo "$ORPHANS"
    read -p "   Kill these? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ps aux | grep -E "python.*mlx_lm|python.*train" | grep -v grep | grep -v $$ | awk '{print $2}' | xargs kill -9 2>/dev/null
        echo "   Killed."
    fi
else
    echo "   None found."
fi

# 2. Kill memory-heavy background apps
echo ""
echo "2. Killing sync/utility apps..."
for app in "Resilio Sync" "Putio Sync" "Grammarly Desktop" "Stats" "OneDrive"; do
    if pgrep -x "$app" > /dev/null 2>&1; then
        killall "$app" 2>/dev/null && echo "   Killed $app" || true
    fi
done

# 3. Stop iCloud sync daemons
echo ""
echo "3. Stopping iCloud sync..."
killall bird cloudd 2>/dev/null && echo "   Stopped bird/cloudd" || echo "   Not running"

# 4. Stop Spotlight indexing
# Real incident, 2026-07-21 (chatPDB Phase 4): sudo mdutil fails silently with no TTY (agent
# sessions have none, and the `!` prefix shell doesn't either -- only a real Terminal window does).
# When it fails, Spotlight's indexing family (corespotlightd, spotlightknowledged,
# spotlightknowledged.updater, mediaanalysisd) can run unimpeded and has been observed starving a
# training subprocess of CPU for 3+ hours at a time (confirmed: the subprocess accumulated under 1s
# of CPU time across a full 60s window while corespotlightd ran at 100-255% and mediaanalysisd at
# ~97%). Recurred 4 times in one day. `kill` (SIGTERM) is silently ignored -- `kill -9` (SIGKILL)
# actually works (these are user-owned, not root), but launchd respawns them within seconds either
# way, so neither is a durable fix alone. The likely actual root cause on this project:
# data/structures_all/ has 256,444 newly-added files (353 GB) -- a very plausible repeated re-index
# trigger. `.metadata_never_index` marker files (no sudo needed, real documented macOS mechanism)
# tell Spotlight to permanently skip a directory tree -- added below, which should reduce/stop
# future recurrences even without sudo, though an already-in-progress scan may not stop instantly.
echo ""
echo "4. Stopping Spotlight indexing..."
sudo mdutil -a -i off 2>/dev/null && echo "   Spotlight paused via mdutil" || echo "   mdutil failed (need sudo -- expected with no TTY in an agent session)"
touch "$(dirname "$0")/../data/.metadata_never_index" 2>/dev/null
touch "$(dirname "$0")/../data/structures_all/.metadata_never_index" 2>/dev/null
echo "   Marked data/ and data/structures_all/ as .metadata_never_index (no sudo needed)"
SPOTLIGHT_PIDS=$(ps aux | grep -E "corespotlightd|spotlightknowledged|mediaanalysisd|duetexpertd" | grep -v grep | awk '{print $2}')
if [ -n "$SPOTLIGHT_PIDS" ]; then
    SPOTLIGHT_CPU=$(ps aux | grep -E "corespotlightd|spotlightknowledged|mediaanalysisd|duetexpertd" | grep -v grep | awk '{sum+=$3} END {print sum}')
    echo "   Found Spotlight/media-analysis processes using ~${SPOTLIGHT_CPU}% combined CPU:"
    ps aux | grep -E "corespotlightd|spotlightknowledged|mediaanalysisd|duetexpertd" | grep -v grep | awk '{printf "     PID %s  %s%%CPU  %s\n", $2, $3, $11}'
    echo "$SPOTLIGHT_PIDS" | xargs kill -9 2>/dev/null && echo "   Sent SIGKILL (works without sudo, but launchd will likely respawn within seconds)" || true
    echo ""
    echo "   ####################################################################"
    echo "   # WARNING: Spotlight indexing could not be durably paused (no sudo)."
    echo "   # This has previously starved training of CPU for 3+ hours at a time."
    echo "   # If this run is long/unattended, open a REAL Terminal window (not"
    echo "   # this session) and run:  sudo mdutil -a -i off"
    echo "   ####################################################################"
    echo ""
else
    echo "   No active Spotlight/media-analysis processes found."
fi

# 5. Stop Time Machine
echo ""
echo "5. Stopping Time Machine..."
sudo tmutil disable 2>/dev/null && echo "   Time Machine paused" || echo "   Failed (need sudo)"

# 6. Kill mlx_lm server if running (frees model memory)
echo ""
echo "6. Checking for mlx_lm server..."
if lsof -ti :8080 > /dev/null 2>&1; then
    lsof -ti :8080 | xargs kill 2>/dev/null
    echo "   Killed server on port 8080"
else
    echo "   No server running"
fi

# 7. Flush file cache
echo ""
echo "7. Flushing file cache..."
sudo purge 2>/dev/null && echo "   Cache purged" || echo "   Failed (need sudo)"

# 8. Wait for memory to settle
echo ""
echo "8. Waiting 5 seconds for memory to settle..."
sleep 5

# 9. Report memory state
echo ""
echo "=== Memory Report ==="
echo ""
vm_stat | grep -E "Pages free|Pages active|Pages wired|Swapouts"
echo ""
sysctl -n vm.swapusage
echo ""

# Calculate approximate free memory
FREE_PAGES=$(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.')
INACTIVE_PAGES=$(vm_stat | grep "Pages inactive" | awk '{print $3}' | tr -d '.')
PAGE_SIZE=16384  # Apple Silicon uses 16K pages
FREE_GB=$(echo "scale=1; ($FREE_PAGES + $INACTIVE_PAGES) * $PAGE_SIZE / 1073741824" | bc 2>/dev/null || echo "?")
echo "Approximate free + reclaimable: ${FREE_GB} GB"
echo ""

# Check swap
SWAP_USED=$(sysctl -n vm.swapusage | grep -oE 'used = [0-9.]+' | awk '{print $3}')
if [ -n "$SWAP_USED" ] && [ "$(echo "$SWAP_USED > 100" | bc 2>/dev/null)" = "1" ]; then
    echo "WARNING: ${SWAP_USED}M swap in use. Consider rebooting for a clean slate."
else
    echo "Swap is clean."
fi

# Disk report (chatPDB-specific: data/structures_all/ is large and can eat headroom fast)
echo ""
echo "=== Disk Report ==="
df -H / | tail -1
echo ""

echo "=== Preflight complete ==="
echo "Run training with: python scripts/train_launch.py --config config/train_config.yaml"
echo ""
echo "After training, restore services with: bash scripts/postflight.sh"

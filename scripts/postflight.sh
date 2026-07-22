#!/usr/bin/env bash
# postflight.sh — restore services paused by preflight.sh
# Run with: bash scripts/postflight.sh

echo "=== Restoring services ==="

echo "1. Re-enabling Spotlight..."
sudo mdutil -a -i on 2>/dev/null && echo "   Done" || echo "   Failed (need sudo)"

echo "2. Re-enabling Time Machine..."
sudo tmutil enable 2>/dev/null && echo "   Done" || echo "   Failed (need sudo)"

echo "3. iCloud and sync apps will restart on next login."
echo ""
echo "=== Services restored ==="

#!/usr/bin/env bash
# Push chatPDB's web app from your Mac to the droplet and restart the service.
# Run from the web/flask_app directory (or the chatPDB repo root):
#
#   bash web/flask_app/deploy/deploy.sh
#
# Real fixes vs. chem_sage's own version of this script (found 2026-07-22):
#   1. Preserves the real repo structure under $APP_DIR (web/flask_app/, scripts/, rag/, config/)
#      instead of flattening web/flask_app/* into $APP_DIR -- chat_remote.py's own REPO_ROOT
#      (three levels up from itself) only resolves correctly if that nesting is kept intact.
#   2. Also syncs data/corpus/ (1.8GB) and .chroma/ (9.9GB) -- chem_sage's own deploy.sh only ever
#      synced scripts/ and rag/ (app code), never the actual corpus data those modules read at
#      runtime; without it, rag/corpus_lookup.py's fast-path and rag/retrieve.py's RAG both
#      silently fail to find their data. data/structures_all/ (353GB) is deliberately NOT synced --
#      not feasible on this droplet; rag/tool_exec.py's Biopython execution won't work in the
#      hosted demo, an accepted scope reduction (see PROJECT_PLAN.md Phase 9).
set -euo pipefail

DROPLET="${DROPLET_SSH:-root@45.55.102.228}"
APP_DIR=/opt/chatpdb

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASK_APP="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$FLASK_APP/../.." && pwd)"

echo "==> Syncing web/flask_app to ${DROPLET}:${APP_DIR}/web/flask_app"
ssh "$DROPLET" "mkdir -p $APP_DIR/web/flask_app"
rsync -az --delete \
  --exclude '.venv/' --exclude '__pycache__/' --exclude '*.pyc' \
  --exclude '.env' --exclude 'static/uploads/' --exclude 'deploy/' \
  "$FLASK_APP/" "$DROPLET:$APP_DIR/web/flask_app/"

echo "==> Syncing scripts/chat.py, rag/, config/system_prompt.txt (app code chat.py needs)"
ssh "$DROPLET" "mkdir -p $APP_DIR/scripts $APP_DIR/rag $APP_DIR/config"
rsync -az "$REPO_ROOT/scripts/chat.py" "$DROPLET:$APP_DIR/scripts/chat.py"
rsync -az --delete --exclude '__pycache__/' --exclude '*.pyc' \
  "$REPO_ROOT/rag/" "$DROPLET:$APP_DIR/rag/"
rsync -az "$REPO_ROOT/config/system_prompt.txt" "$DROPLET:$APP_DIR/config/system_prompt.txt"

echo "==> Syncing data/corpus/ (1.8GB) and .chroma/ (9.9GB) -- this will take a while first time"
ssh "$DROPLET" "mkdir -p $APP_DIR/data"
rsync -az --delete "$REPO_ROOT/data/corpus/" "$DROPLET:$APP_DIR/data/corpus/"
rsync -az --delete "$REPO_ROOT/.chroma/" "$DROPLET:$APP_DIR/.chroma/"

echo "==> Installing dependencies + restarting service"
ssh "$DROPLET" bash -s <<REMOTE
set -euo pipefail
FLASK_DIR="${APP_DIR}/web/flask_app"
cd "\$FLASK_DIR"
if [[ ! -x .venv/bin/python ]]; then
  echo "No venv yet -- run deploy/provision.sh as root first."; exit 1
fi
sudo -u chatpdb .venv/bin/pip install --quiet -r requirements.txt
find "${APP_DIR}" -not -path "${APP_DIR}/web/flask_app/.venv/*" -exec chown chatpdb:chatpdb {} + 2>/dev/null || true
sudo systemctl restart chatpdb-web.service
sudo systemctl --no-pager --lines=3 status chatpdb-web.service || true
REMOTE

echo "==> Deployed to https://chatpdb.mdeller.com"

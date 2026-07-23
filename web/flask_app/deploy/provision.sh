#!/usr/bin/env bash
# One-time droplet provisioning for chatPDB web app. Run as root ON the droplet after
# pushing code with deploy.sh:
#
#   sudo SERVER_NAME=chatpdb.mdeller.com bash /opt/chatpdb/web/flask_app/deploy/provision.sh
#
# Idempotent: safe to re-run.
#
# Real fix vs. chem_sage's own version of this script (found 2026-07-22): chem_sage's deploy.sh
# flattens web/flask_app/* directly into $APP_DIR (/opt/chemsage/*), but chat_remote.py's own
# REPO_ROOT = Path(__file__).resolve().parent.parent.parent assumes the real repo nesting
# (web/flask_app/chat_remote.py -> repo root, three levels up) -- flattened, that resolves to "/"
# and scripts/chat.py is never found. This script (and deploy.sh) preserve the real repo structure
# under $APP_DIR instead, so REPO_ROOT resolves correctly on the droplet exactly as it does here.
set -euo pipefail

APP_DIR=/opt/chatpdb
FLASK_DIR="$APP_DIR/web/flask_app"
APP_USER=chatpdb
BIND_ADDR="127.0.0.1:8002"
SERVER_NAME="${SERVER_NAME:-chatpdb.mdeller.com}"

echo "==> chatPDB provisioning for ${SERVER_NAME}"

if [[ $EUID -ne 0 ]]; then echo "Run as root."; exit 1; fi
if [[ ! -f "$FLASK_DIR/app.py" ]]; then
  echo "No code at $FLASK_DIR — push it first: bash deploy/deploy.sh"; exit 1
fi

echo "==> Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv python3-pip python3-dev build-essential \
  nginx certbot python3-certbot-nginx rsync

echo "==> Creating service user '${APP_USER}'"
id -u "$APP_USER" &>/dev/null || useradd --system --home "$APP_DIR" --shell /usr/sbin/nologin "$APP_USER"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

echo "==> Building Python venv"
if [[ ! -x "$FLASK_DIR/.venv/bin/python" ]]; then
  sudo -u "$APP_USER" python3 -m venv "$FLASK_DIR/.venv"
fi
sudo -u "$APP_USER" "$FLASK_DIR/.venv/bin/pip" install --quiet --upgrade pip
sudo -u "$APP_USER" "$FLASK_DIR/.venv/bin/pip" install --quiet -r "$FLASK_DIR/requirements.txt"

echo "==> Creating .env (HF Space URL + token)"
if [[ ! -f "$FLASK_DIR/.env" ]]; then
  cat > "$FLASK_DIR/.env" <<'EOF'
HF_SPACE_URL=https://dellboy-chatpdb-api.hf.space
HF_TOKEN=change-me
SECRET_KEY=change-me
EOF
  chown "$APP_USER:$APP_USER" "$FLASK_DIR/.env"
  chmod 600 "$FLASK_DIR/.env"
  echo "    Created $FLASK_DIR/.env — edit HF_TOKEN and SECRET_KEY before going live."
  echo "    (HF_TOKEN is required: ZeroGPU's anonymous quota is ~85s/day and requests without"
  echo "    a Bearer token silently start failing after a handful of calls.)"
fi

echo "==> Installing systemd unit"
cp "$FLASK_DIR/deploy/chatpdb-web.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now chatpdb-web.service

echo "==> Installing nginx site"
sed -e "s|__SERVER_NAME__|${SERVER_NAME}|g" \
  "$FLASK_DIR/deploy/nginx-chatpdb.conf" > /etc/nginx/sites-available/chatpdb
ln -sf /etc/nginx/sites-available/chatpdb /etc/nginx/sites-enabled/chatpdb
nginx -t && systemctl reload nginx

echo "==> Requesting TLS certificate"
if certbot certificates 2>/dev/null | grep -q "$SERVER_NAME"; then
  echo "    Certificate for ${SERVER_NAME} already present; skipping."
else
  certbot --nginx -d "$SERVER_NAME" --non-interactive --agree-tos \
    -m "marc@marcdeller.com" --redirect || \
    echo "    certbot failed (DNS not pointed yet?). Re-run: certbot --nginx -d ${SERVER_NAME}"
fi

echo "==> Done."
systemctl --no-pager --lines=5 status chatpdb-web.service || true

#!/bin/bash
# LIMEN deployment script
# Run on the machine that will host LIMEN (e.g., 192.168.5.49)
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/denster32/LIMEN/main/scripts/deploy.sh | bash
#   OR
#   git clone https://github.com/denster32/LIMEN.git && cd LIMEN && bash scripts/deploy.sh

set -e

LIMEN_DIR="${LIMEN_DIR:-$HOME/LIMEN}"
STATE_DIR="${STATE_DIR:-$HOME/.limen}"
PORT="${LIMEN_PORT:-8452}"

echo "=== LIMEN Deploy ==="
echo "Install dir: $LIMEN_DIR"
echo "State dir:   $STATE_DIR"
echo "Port:        $PORT"
echo ""

# Clone or update
if [ -d "$LIMEN_DIR" ]; then
    echo "Updating existing installation..."
    cd "$LIMEN_DIR"
    git pull origin main
else
    echo "Cloning LIMEN..."
    git clone https://github.com/denster32/LIMEN.git "$LIMEN_DIR"
    cd "$LIMEN_DIR"
fi

# Dependencies
echo "Installing dependencies..."
pip3 install mcp starlette uvicorn httpx 2>/dev/null || pip install mcp starlette uvicorn httpx

# State directory
mkdir -p "$STATE_DIR"

# Generate token if not set
if [ -z "$LIMEN_TOKEN" ]; then
    TOKEN_FILE="$STATE_DIR/.token"
    if [ -f "$TOKEN_FILE" ]; then
        LIMEN_TOKEN=$(cat "$TOKEN_FILE")
        echo "Using existing token from $TOKEN_FILE"
    else
        LIMEN_TOKEN=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
        echo "$LIMEN_TOKEN" > "$TOKEN_FILE"
        chmod 600 "$TOKEN_FILE"
        echo "Generated new token, saved to $TOKEN_FILE"
    fi
fi

echo ""
echo "=== LIMEN Ready ==="
echo ""
echo "Auth token: $LIMEN_TOKEN"
echo ""
echo "To run manually:"
echo "  cd $LIMEN_DIR"
echo "  LIMEN_TOKEN=$LIMEN_TOKEN python3 -m scripts.run_server --mode rest --state $STATE_DIR/limen.json --port $PORT"
echo ""
echo "To run as a background service (systemd):"
echo "  sudo cp $LIMEN_DIR/scripts/limen.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable --now limen"
echo ""
echo "Claude.ai context URL:"
echo "  https://YOUR_DOMAIN:$PORT/context?token=$LIMEN_TOKEN"
echo ""
echo "For public access, set up one of:"
echo "  - Cloudflare Tunnel: cloudflared tunnel --url http://localhost:$PORT"
echo "  - Tailscale Funnel: tailscale funnel $PORT"
echo "  - ngrok: ngrok http $PORT"

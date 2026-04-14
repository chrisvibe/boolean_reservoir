#!/bin/bash

# --- Configuration ---
REMOTE_HOST="hpc4"
LOCAL_PROJECT_ROOT="/home/leon/Documents/katt/phd/code/boolean_reservoir"
REMOTE_PROJECT_ROOT="/cluster/datastore/christvi/code/dev/boolean_reservoir"
DASHBOARD_PATH="deploy/scatter_dashboard"

# --- Functions ---

# Establish SSH Tunnel for the dashboard UI
tunnel() {
    echo "Opening SSH tunnels on 8051 and 8052..."
    ssh -NL 8051:localhost:8051 -L 8052:localhost:8052 $REMOTE_HOST
}

# Sync local code TO the HPC
push_deploy() {
    echo "Pushing project to $REMOTE_HOST..."
    rsync -zav --delete \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        --exclude='.git/' \
        "$LOCAL_PROJECT_ROOT/" "$REMOTE_HOST:$REMOTE_PROJECT_ROOT/"
}

# Restart the Docker container on HPC
restart_dashboard() {
    echo "Restarting dashboard container on $REMOTE_HOST..."
    ssh $REMOTE_HOST "cd $REMOTE_PROJECT_ROOT/$DASHBOARD_PATH && docker compose restart && docker compose logs -f"
}

# Sync generated views FROM the HPC to local
pull_views() {
    echo "Pulling views from $REMOTE_HOST..."
    rsync -av --delete \
        "$REMOTE_HOST:$REMOTE_PROJECT_ROOT/out/dashboard/" \
        "$LOCAL_PROJECT_ROOT/out/dashboard/"
}

# Combined action: Push code and then restart the service
deploy_dashboard() {
    push_deploy
    restart_dashboard
}

# Full workflow: Pull old views, Push, Restart
sync_all() {
    pull_views
    deploy_dashboard
}

# --- Execution ---
if [ -z "$1" ]; then
    echo "Usage: ./manage_dashboard.sh [tunnel | push_deploy | restart_dashboard | pull_views | deploy_dashboard | sync_all]"
fi

# Run the requested function
"$@"

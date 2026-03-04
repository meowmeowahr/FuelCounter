#!/usr/bin/env bash

set -e

# ===== CONFIG =====
MONACO_VERSION="latest"      # Change to specific version if desired
STATIC_DIR="static/monaco"
TMP_DIR="monaco_tmp"
# ==================

echo "Updating Monaco Editor (${MONACO_VERSION})..."

# Clean temp
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"

pushd "$TMP_DIR" > /dev/null

# Init temporary npm project
npm init -y > /dev/null 2>&1

# Install monaco
npm install monaco-editor@$MONACO_VERSION > /dev/null

popd > /dev/null

# Remove old static version
echo "Removing old static/monaco..."
rm -rf "$STATIC_DIR"

# Copy only production build (minified vs directory)
echo "Copying new version..."
mkdir -p "$STATIC_DIR"
cp -r "$TMP_DIR/node_modules/monaco-editor/min/vs" "$STATIC_DIR/"

# Cleanup
rm -rf "$TMP_DIR"

echo "Monaco successfully updated."
echo "Installed version:"
node -e "console.log(require('./node_modules/monaco-editor/package.json').version)" 2>/dev/null || echo "(version check skipped)"

echo "Done."
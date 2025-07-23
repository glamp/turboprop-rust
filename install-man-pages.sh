#!/bin/bash

# Install man pages for TurboProp
set -e

# Build to generate man pages
cargo build

# Find the generated man pages
OUT_DIR=$(find target -name "tp.1" | head -n 1 | xargs dirname)

if [ -z "$OUT_DIR" ]; then
    echo "Error: Could not find generated man pages. Try running 'cargo build' first."
    exit 1
fi

echo "Found man pages in: $OUT_DIR"

# Determine installation directory
if [ -w "/usr/local/share/man/man1" ]; then
    MAN_DIR="/usr/local/share/man/man1"
elif [ -w "/usr/share/man/man1" ]; then
    MAN_DIR="/usr/share/man/man1"
elif [ -d "$HOME/.local/share/man/man1" ]; then
    MAN_DIR="$HOME/.local/share/man/man1"
else
    # Create local man directory
    MAN_DIR="$HOME/.local/share/man/man1"
    mkdir -p "$MAN_DIR"
fi

echo "Installing man pages to: $MAN_DIR"

# Copy man page
cp "$OUT_DIR/tp.1" "$MAN_DIR/"

# Update man database if available
if command -v mandb >/dev/null 2>&1; then
    echo "Updating man database..."
    if [ "$MAN_DIR" = "/usr/local/share/man/man1" ] || [ "$MAN_DIR" = "/usr/share/man/man1" ]; then
        sudo mandb
    else
        mandb "$HOME/.local/share/man" 2>/dev/null || true
    fi
fi

echo "Installation complete!"
echo ""
echo "You can now view the manual with:"
echo "  man tp"
echo ""
echo "If the manual is not found, make sure your MANPATH includes:"
echo "  $MAN_DIR"
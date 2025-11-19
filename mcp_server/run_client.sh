#!/bin/bash

# Script to run the MCP client

echo "============================================================"
echo "Starting MCP Client"
echo "============================================================"
echo "Make sure the MCP server is running first!"
echo "============================================================"
echo ""

cd "$(dirname "$0")"
uv run client.py

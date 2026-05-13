#!/bin/bash
# Run end-to-end simulation for all domains
# Requires GOOGLE_API_KEY to be set

set -e

if [ -z "$GOOGLE_API_KEY" ]; then
    echo "Error: GOOGLE_API_KEY environment variable is required"
    exit 1
fi

OUTPUT_DIR="${1:-output}"
mkdir -p "$OUTPUT_DIR"

echo "=== Personal Timeline ==="
python -m cognifold.cli run data/generated/alex_chen_timeline.json --agent --steps 10 -o "$OUTPUT_DIR"

echo ""
echo "=== Computer Activity ==="
python -m cognifold.cli run data/generated/computer_software_developer_timeline.json --agent --steps 10 -o "$OUTPUT_DIR"

echo ""
echo "=== Service Logs ==="
python -m cognifold.cli run data/generated/service_ecommerce_timeline.json --agent --steps 10 -o "$OUTPUT_DIR"

echo ""
echo "=== Complete ==="
echo "Output files in: $OUTPUT_DIR"

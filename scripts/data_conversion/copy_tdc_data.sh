#!/bin/bash
# Copy all TDC datasets from therapeutic-tuning to OpenRLHF-Tools

SOURCE_DIR="/Users/jlee0/Desktop/research/therapeutic-tuning/data/TDC"
TARGET_DIR="/Users/jlee0/Desktop/research/OpenRLHF-Tools/data/tdc/raw"
METADATA_DIR="/Users/jlee0/Desktop/research/OpenRLHF-Tools/data/tdc/metadata"

# Create target directories
mkdir -p "$TARGET_DIR"
mkdir -p "$METADATA_DIR"

# Copy all datasets (rsync preserves structure)
echo "Copying TDC datasets from $SOURCE_DIR to $TARGET_DIR..."
rsync -av --progress \
    --include='*/' \
    --include='train.csv' \
    --include='test.csv' \
    --include='val.csv' \
    --exclude='*' \
    "$SOURCE_DIR/" "$TARGET_DIR/"

echo "✓ Copied TDC datasets to $TARGET_DIR"

# Copy prompt templates
echo "Copying prompt templates..."
cp /Users/jlee0/Desktop/research/therapeutic-tuning/prompts/templates/tdc_prompts.json \
   "$METADATA_DIR/prompts.json"

echo "✓ Copied prompt templates to $METADATA_DIR/prompts.json"
echo ""
echo "Setup complete! Next steps:"
echo "1. Run build_tool_definitions.py to extract tool schemas"
echo "2. Run convert_tdc_to_openai.py to convert CSV files to OpenAI format"

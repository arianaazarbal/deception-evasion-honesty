#!/bin/bash
# Script to disable flash attention in all Python files

echo "Disabling flash attention in all files..."

# Comment out all flash_attention_2 references
find /root/deception-evasion-honesty/solid_deception -name "*.py" -type f -exec sed -i 's/attn_implementation="flash_attention_2"/attn_implementation="sdpa"/g' {} +
find /root/deception-evasion-honesty/solid_deception -name "*.py" -type f -exec sed -i 's/"flash_attention_2"/"sdpa"/g' {} +

echo "Flash attention disabled. Using SDPA (scaled dot-product attention) instead."
echo "SDPA is PyTorch's built-in optimized attention that works well on H200."
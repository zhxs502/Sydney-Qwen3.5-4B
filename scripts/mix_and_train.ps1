# Data mix + training launch (run after P104 is plugged in)
#
# Prereq:
#   E:\llm\train\env         venv (torch 2.6+cu124 installed)
#   E:\llm\train\qwen35-4b   base model
#   E:\sydney-qwen3.5-4-b\data\train_v8.jsonl            782 records (old)
#   E:\sydney-qwen3.5-4-b\corpus\dataset\train_zh.jsonl  188 records (new)
#
# Output: E:\sydney-qwen3.5-4-b\data\train_mixed.jsonl
#
# Train (from scratch, NOT resume - avoid catastrophic forgetting):
#   E:\llm\train\env\Scripts\python.exe E:\sydney-qwen3.5-4-b\scripts\train_v3.py `
#     --model E:\llm\train\qwen35-4b `
#     --data  E:\sydney-qwen3.5-4-b\data\train_mixed.jsonl `
#     --out   D:\train_out\v9 `
#     --seq 512 --accum 8 --lr 1e-4 --r 16 --epochs 2 --bbox 1 --novis 1
#
# Notes:
#   - seq 512 is the sweet spot (896 drops to 11 tok/s)
#   - --bbox 1 = NF4, only viable config on 8GB VRAM
#   - --novis 1 excludes the visual tower
#   - do NOT write --out to E: (exFAT USB); use D:\train_out

$ErrorActionPreference = "Stop"
$repo = 'E:\sydney-qwen3.5-4-b'
$dst  = Join-Path $repo 'data\train_mixed.jsonl'
$old  = Join-Path $repo 'data\train_v8.jsonl'
$new  = Join-Path $repo 'corpus\dataset\train_zh.jsonl'

foreach ($p in @($old, $new)) {
	if (-not (Test-Path $p)) { throw "missing: $p" }
}

$pyPath = 'D:\_temp\sydney-src\mix_data.py'
& 'E:\llm\train\env\Scripts\python.exe' $pyPath $old $new $dst

"OK: $dst  ({0:N1} KB)" -f ((Get-Item $dst).Length/1KB)

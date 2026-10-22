# 三合一训练启动（P104 插上后跑）
#
# 数据：train_all.jsonl = 新中文 190 + 旧中文 782 + 英文 579
#       1,551 条 / 9,500 轮 / 208.8 万字符（中文 23% / 英文 76%）
#
# 训练时间估算（P104-100, 129 tok/s）：
#   1 epoch ~ 2.8h
#   2 epoch ~ 5.6h
#
# 建议先跑 1 epoch 看效果：
#   powershell -File scripts\train_all.ps1
# 想跑 2 epoch：
#   powershell -File scripts\train_all.ps1 -Epochs 2
#
# 注意：
#   - seq 512 是甜点（896 掉到 11 tok/s）
#   - --bbox 1 = NF4，8GB 显存唯一可行
#   - --novis 1 排除视觉塔
#   - --out 写 D:\train_out（别写 E:，exFAT USB）
#   - 训完必须测：中文 prompt 会不会蹦英文词（code-switching）

param(
	[int]$Epochs = 1,
	[string]$Out = 'D:\train_out\v9'
)

$ErrorActionPreference = "Stop"
$repo = 'E:\sydney-qwen3.5-4-b'
$data = Join-Path $repo 'data\train_all.jsonl'
$model = 'E:\llm\train\qwen35-4b'
$py = 'E:\llm\train\env\Scripts\python.exe'
$script = Join-Path $repo 'scripts\train_v3.py'

foreach ($p in @($data, $model, $py, $script)) {
	if (-not (Test-Path $p)) { throw "missing: $p" }
}

if (-not (Test-Path $Out)) { New-Item -ItemType Directory -Force $Out | Out-Null }

Write-Output "data=$data"
Write-Output "model=$model"
Write-Output "out=$Out"
Write-Output "epochs=$Epochs seq=512 accum=8 lr=1e-4 r=16 bbox=1 novis=1"

& $py $script `
	--model $model `
	--data $data `
	--out $Out `
	--seq 512 --accum 8 --lr 1e-4 --r 16 --epochs $Epochs --bbox 1 --novis 1

@echo off
cd /d "%~dp0"
if exist "adapter_model.safetensors" (
  echo adapter_model.safetensors already exists, skip.
  pause
  exit /b
)
copy /b adapter_model.safetensors.part1+adapter_model.safetensors.part2 adapter_model.safetensors
if errorlevel 1 (
  echo merge failed.
) else (
  echo merged to adapter_model.safetensors
)
pause

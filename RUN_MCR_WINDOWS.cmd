@echo off
setlocal
cd /d "%~dp0"

if not exist "mosaic_natural_chat.exe" (
  echo mosaic_natural_chat.exe is missing. Run MCR_BUILD_WINDOWS.cmd first.
  exit /b 1
)
if not exist "mosaic_dialogue.mos" (
  echo mosaic_dialogue.mos is missing. Extract the complete Kaggle ZIP package.
  exit /b 1
)

mosaic_natural_chat.exe --model mosaic_dialogue.mos --history mosaic_history.tsv --max-new 128 --temperature 0.82 --top-p 0.90

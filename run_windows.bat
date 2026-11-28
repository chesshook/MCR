@echo off
setlocal
cd /d "%~dp0"
if not exist build\mosaic_chat.exe call build_windows.bat
if errorlevel 1 exit /b 1

set MODEL=intent_model.mcr
if not exist "%MODEL%" set MODEL=-

if exist wordnet.tsv (
  build\mosaic_chat.exe --model "%MODEL%" --lexicon wordnet.tsv --memory mosaic_memory.tsv
) else (
  build\mosaic_chat.exe --model "%MODEL%" --memory mosaic_memory.tsv
)

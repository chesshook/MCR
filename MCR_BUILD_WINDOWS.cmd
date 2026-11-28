@echo off
setlocal EnableExtensions
cd /d "%~dp0"

where cmake >nul 2>nul
if errorlevel 1 (
  echo CMake was not found. Install CMake and add it to PATH.
  exit /b 1
)

cmake -S "mosaic_natural\cpp" -B "mosaic_natural\cpp\build-win" -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1

cmake --build "mosaic_natural\cpp\build-win" --config Release --parallel 2
if errorlevel 1 exit /b 1

set "EXE=mosaic_natural\cpp\build-win\Release\mosaic_natural_chat.exe"
if not exist "%EXE%" set "EXE=mosaic_natural\cpp\build-win\mosaic_natural_chat.exe"
if not exist "%EXE%" (
  echo The C++ build completed, but mosaic_natural_chat.exe was not found.
  exit /b 1
)

copy /y "%EXE%" "mosaic_natural_chat.exe" >nul
if errorlevel 1 exit /b 1

echo.
echo BUILD COMPLETE.
echo Run RUN_MCR_WINDOWS.cmd to chat with MCR.

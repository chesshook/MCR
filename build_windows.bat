@echo off
setlocal
cd /d "%~dp0"

where g++ >nul 2>nul
if errorlevel 1 (
  echo ERROR: MinGW-w64 g++ was not found in PATH.
  exit /b 1
)

if not exist build mkdir build

echo [1/3] Building native MOSAIC chat...
g++ -std=c++17 -O3 -DNDEBUG -mtune=generic -static -static-libgcc -static-libstdc++ ^
  -Wall -Wextra -Wpedantic -Imosaic_core\include ^
  mosaic_core\src\agent.cpp mosaic_core\src\main.cpp ^
  -o build\mosaic_chat.exe
if errorlevel 1 exit /b 1

echo [2/3] Building native benchmark...
g++ -std=c++17 -O3 -DNDEBUG -mtune=generic -static -static-libgcc -static-libstdc++ ^
  -Wall -Wextra -Wpedantic -Imosaic_core\include ^
  mosaic_core\src\agent.cpp mosaic_core\src\benchmark.cpp ^
  -o build\mosaic_bench.exe
if errorlevel 1 exit /b 1

echo [3/3] Building and running tests...
g++ -std=c++17 -O2 -mtune=generic -Imosaic_core\include ^
  mosaic_core\src\agent.cpp mosaic_core\tests\state_kernel_test.cpp ^
  -o build\mosaic_kernel_test.exe
if errorlevel 1 exit /b 1
build\mosaic_kernel_test.exe
if errorlevel 1 exit /b 1

g++ -std=c++17 -O2 -mtune=generic -Imosaic_core\include ^
  mosaic_core\src\agent.cpp mosaic_core\tests\agent_test.cpp ^
  -o build\mosaic_agent_test.exe
if errorlevel 1 exit /b 1
build\mosaic_agent_test.exe
if errorlevel 1 exit /b 1

g++ -std=c++17 -O2 -mtune=generic -Imosaic_core\include ^
  mosaic_core\tests\lexicon_test.cpp -o build\mosaic_lexicon_test.exe
if errorlevel 1 exit /b 1
build\mosaic_lexicon_test.exe
if errorlevel 1 exit /b 1

echo.
echo BUILD SUCCESSFUL
echo Executable: build\mosaic_chat.exe
echo The executable is compiled for generic x86-64 CPUs, not the build machine only.

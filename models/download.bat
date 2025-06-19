chcp 65001
@echo off
setlocal enabledelayedexpansion

REM Set download directory
set "DOWNLOAD_DIR=."

REM Create download directory if not exists
if not exist "%DOWNLOAD_DIR%" mkdir "%DOWNLOAD_DIR%"

echo Starting downloads...

REM Download model.onnx
echo Downloading model.onnx...
powershell -Command "Invoke-WebRequest -Uri 'https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx/resolve/main/model.onnx' -OutFile '%DOWNLOAD_DIR%\model.onnx'"
if !errorlevel! neq 0 (
    echo Failed to download model.onnx
    exit /b 1
)

REM Download tokenizer.json
echo Downloading tokenizer.json...
powershell -Command "Invoke-WebRequest -Uri 'https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx/resolve/main/tokenizer.json' -OutFile '%DOWNLOAD_DIR%\tokenizer.json'"
if !errorlevel! neq 0 (
    echo Failed to download tokenizer.json
    exit /b 1
)

REM Clean up existing files
if exist "%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0.zip" del /f /q "%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0.zip"
if exist "%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0" rmdir /s /q "%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0"

REM Download onnxruntime
echo Downloading onnxruntime...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip' -OutFile '%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0.zip'"
if !errorlevel! neq 0 (
    echo Failed to download onnxruntime
    exit /b 1
)

REM Extract onnxruntime
echo Extracting onnxruntime...
powershell -Command "Expand-Archive -Path '%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0.zip' -DestinationPath '%DOWNLOAD_DIR%' -Force"
if !errorlevel! neq 0 (
    echo Failed to extract onnxruntime
    del /f /q "%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0.zip"
    exit /b 1
)

REM Check if dll exists
if not exist "%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0\lib\onnxruntime.dll" (
    echo Cannot find onnxruntime.dll in the expected path
    echo Checking alternative path...
    if not exist "%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0\bin\onnxruntime.dll" (
        echo Cannot find onnxruntime.dll in alternative path
        echo Cleaning up...
        del /f /q "%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0.zip"
        rmdir /s /q "%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0"
        exit /b 1
    )
    set "DLL_PATH=%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0\bin\onnxruntime.dll"
) else (
    set "DLL_PATH=%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0\lib\onnxruntime.dll"
)

REM Copy onnxruntime.dll
echo Copying onnxruntime.dll...
copy /y "!DLL_PATH!" "%DOWNLOAD_DIR%\onnxruntime.dll"
if !errorlevel! neq 0 (
    echo Failed to copy onnxruntime.dll
    del /f /q "%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0.zip"
    rmdir /s /q "%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0"
    exit /b 1
)

REM Clean up
del /f /q "%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0.zip"
rmdir /s /q "%DOWNLOAD_DIR%\onnxruntime-win-x64-1.22.0"

echo Download completed successfully!
echo Downloaded files:
dir /b "%DOWNLOAD_DIR%\model.onnx" "%DOWNLOAD_DIR%\tokenizer.json" "%DOWNLOAD_DIR%\onnxruntime.dll"

endlocal

@echo off
chcp 65001 >nul
title Disney Vision - Character Detection
cd /d "%~dp0"

if "%~1"=="" (
    echo.
    echo ======================================================================
    echo   Disney Vision - Reconnaissance de Personnages Bolderbast
    echo ======================================================================
    echo.
    set /p "TARGET=Glissez-deposez un ou plusieurs fichiers (.png, .jpg, .cbz, .pdf) : "
    if defined TARGET (
        python bridge.py --visualize %TARGET%
    )
    echo.
    pause
    exit /b
)

python bridge.py --visualize %*
if errorlevel 1 (
    echo.
    pause
) else (
    echo.
    pause
)

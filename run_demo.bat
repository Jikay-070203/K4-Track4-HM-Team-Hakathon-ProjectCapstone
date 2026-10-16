@echo off
title VinFast ADAS Pedestrian Collision Warning (PCW)
echo ================================================================
echo    VINFAST ADAS - PEDESTRIAN COLLISION WARNING (PCW) SYSTEM
echo    Hackathon Capstone: Track 4 (Computer Vision & Robotics)
echo ================================================================
echo.
echo [1] Running Live Webcam Demo + Backup Recording...
echo.
"C:\Users\yuuki\miniconda3\envs\cv_robotics_lab21\python.exe" main_hud.py --source 0 --record

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [NOTE] If Webcam 0 is occupied, try: python main_hud.py --source 1
    pause
)

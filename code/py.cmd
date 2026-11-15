@echo off
rem Run a script of this folder with the project's Python environment:  py.cmd train_cv.py --model convnext
cd /d "%~dp0"
set PYTHONUNBUFFERED=1
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
..\_env\venv\Scripts\python.exe %*

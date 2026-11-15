@echo off
cd /d "%~dp0"
..\_env\venv\Scripts\python.exe review_soda_fruits.py Anbara Khalas Raziz Safawi Shalabi Sukari_Mofattal Sukari_Rutab Wanan
cd ..\data_ext\soda
python ..\..\code\stack_sheets.py _review_stack1.jpg _review_Anbara.jpg _review_Khalas.jpg _review_Raziz.jpg _review_Safawi.jpg
python ..\..\code\stack_sheets.py _review_stack2.jpg _review_Shalabi.jpg _review_Sukari_Mofattal.jpg _review_Sukari_Rutab.jpg _review_Wanan.jpg
echo STACKED

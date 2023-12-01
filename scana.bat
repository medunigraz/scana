ECHO ON
setlocal
set PYTHONPATH=%PYTHONPATH%;%cd%\scana
python bin\scanatk.py
endlocal

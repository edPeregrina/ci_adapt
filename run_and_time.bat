@echo off

rem Specify the full path for the log file
set LOG_FILE=C:\repos\ci_adapt\execution.log

rem Run the Python script and time it
echo Running the Python script...
python C:\repos\ci_adapt\run_rail_risk.py > %LOG_FILE% 2>&1

rem Print a message indicating completion
echo Script execution completed.

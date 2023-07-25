@ECHO OFF
@REM Using Python 3.10.12
ECHO Installing requirements...
pip install -r requirements.txt
ECHO Installing ffmpeg (Requires conda)
conda install -y ffmpeg
ECHO Done. Use the following command to start the app:
ECHO streamlit run app.py
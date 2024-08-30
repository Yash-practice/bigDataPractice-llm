#!/bin/bash

# Print the current date and time
echo "Starting script at $(date)"

# Port number to be used
PORT=8502

# Find the PID of the process using the specified port
PIDS=$(ps -ef | grep '[s]treamlit' | awk '{print $2}')

# If no PID is found, exit with a message
if [ -z "$PID" ]; then
  echo "No process found using port $PORT."
  exit 0
fi

# Kill the process
kill -9 $PID

# Print confirmation
echo "Killed process $PID using port $PORT."

# sudo su
# gcloud compute ssh vm-genai-llm --zone=us-central1-a
pwd
ls -lrt
cd /root/Doc_Chatbot/bigDataPractice-llm/Chatbot-Codebase-Sentiment-Analysis/
pwd
ls -lrt

python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
nohup python3 -m streamlit run script.py --server.port $PORT > streamlit_output.log 2>&1 &


# Print the completion time
echo "Script completed at $(date)"

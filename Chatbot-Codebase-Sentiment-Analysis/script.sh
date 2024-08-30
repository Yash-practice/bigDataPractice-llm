#!/bin/bash

# Print the current date and time
echo "Starting script at $(date)"
# sudo su
# gcloud compute ssh vm-genai-llm --zone=us-central1-a
pwd
ls -lrt
cd /root/Doc_Chatbot/bigDataPractice-llm/Chatbot-Codebase-Sentiment-Analysis/
pwd
ls -lrt

python3 -m pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
python3 -m streamlit run script.py --server.port 8502 > streamlit_output.log 2>&1


# Print the completion time
echo "Script completed at $(date)"

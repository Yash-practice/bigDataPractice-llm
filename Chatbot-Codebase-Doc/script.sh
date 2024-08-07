#!/bin/bash

# Print the current date and time
echo "Starting script at $(date)"
# sudo su
# gcloud compute ssh vm-genai-llm --zone=us-central1-a
pwd
ls -lrt
cd Medical_Chatbot_Llama2_Pinecone/
pwd
ls -lrt

python3 -m streamlit run script.py --server.port 8501 > streamlit_output.log 2>&1


# Print the completion time
echo "Script completed at $(date)"

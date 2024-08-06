#!/bin/bash

# Print the current date and time
echo "Starting script at $(date)"

sudo su
echo "Script in between at $(date)"
cd Medical_Chatbot_Llama2_Pinecone/

python3 -m streamlit run script.py --server.port 8501


# Print the completion time
echo "Script completed at $(date)"

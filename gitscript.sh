#!/bin/bash
# Print the current date and time
echo "Starting script at $(date)"
# Create the directory if it does not exist
mkdir -p "Doc_Chatbot"
# Navigate into the directory
cd "Doc_Chatbot"
REPO_URL="https://github.com/Yash-practice/bigDataPractice-llm.git"
TARGET_DIR="bigDataPractice-llm"
# Check if the target directory exists
if [ -d "$TARGET_DIR" ]; then
  echo "Directory $TARGET_DIR already exists. Skipping clone."
else
  echo "Directory $TARGET_DIR does not exist. Cloning repository."
  git clone "$REPO_URL" "$TARGET_DIR"
fi
cd "$TARGET_DIR"
# Fetch updates from the repository
git fetch
# Checkout the specific branch
git checkout test-branch-1
# Pull the latest changes from the branch
git pull origin test-branch-1
# Uncomment if needed for debugging
# pwd
# ls -lrt
# Uncomment and modify if needed to run a script
# python3 -m streamlit run script.py --server.port 8501 > streamlit_output.log 2>&1
# Print the completion time
echo "Script completed at $(date)"
#Script to download git files and models
#!/bin/bash
# sed -i 's/\r//' script.sh
# Print the current date and time
echo "Starting script at $(date)"
# Create the directory if it does not exist
mkdir -p "Chatbots"
# Navigate into the directory
cd "Chatbots"
REPO_URL="https://github.com/Yash-practice/bigDataPractice-llm.git"
TARGET_DIR="bigDataPractice-llm"
rm -r "$TARGET_DIR"
git clone "$REPO_URL" "$TARGET_DIR"
cd "$TARGET_DIR"
# Fetch updates from the repository
git fetch
# Checkout the specific branch
git checkout test-branch-1
# Pull the latest changes from the branch
git pull origin test-branch-1
# Copy the model to relevant folder 
cd /root/Chatbots/bigDataPractice-llm
gsutil cp -r gs://models_store/ .
# Uncomment if needed for debugging
# pwd
# ls -lrt
# Uncomment and modify if needed to run a script
# python3 -m streamlit run script.py --server.port 8501 > streamlit_output.log 2>&1
# Print the completion time
echo "Script completed at $(date)"
## Table of Contents

- [How to Start](#how-to-start)
- [Overview](#overview)
- [Features](#features)
- [Technology Used](#technology-used)
- [Installation](#installation)
- [License](#license)

---

# How to Start
- Enter the login credentials
- Once logged in, Navigate to the chatbot on which you want to continue
- The left section is for Document Analysis Chatbot and on right section is for Sentiment Analysis Chatbot
- Click on the chatbot button and you're ready to go.

--- 

# Overview

### Document QnA Chatbot:
A user-friendly Document QnA chatbot where users can query PDF documents using advanced language models (LLMs) through prompt engineering. The chatbot supports multiple languages and allows users to choose their preferred LLM, set specific parameters, and upload documents for interaction. Built using Streamlit for a seamless interface, the bot stores document embeddings in FAISS, a vector database, to ensure fast and efficient document querying.

### Sentiment Analysis:
This is a sentiment analysis chatbot that analyzes user input to determine the sentiment (positive, negative, or neutral) as well as emotions (happy, sad, angry, disappointment, etc.). The chatbot responds based on the sentiment, enhancing user experience through interaction. The backend uses a machine learning model for sentiment analysis, integrated into a user-friendly UI built with Streamlit.


---

## Features
### Document QnA Chatbot:

- Query PDF documents using advanced LLMs through prompt engineering.
- Multi-language support including English, Arabic, French, and German.
- Users can upload their documents and generate embeddings for storage in FAISS.
- Option to select different LLM models and customize parameters.
- Pre-uploaded documents across various domains are available for immediate interaction.
- Embedding storage in FAISS vector database for fast retrieval and querying.

### Sentiment Analysis:
- Sentiment and Emotion Analysis of text, dataset(csv file), email and audio inputs.
- Keyword based analysis is also provided.
- Supports multiple audio file with 5 different format, uploads for Sentiment and Emotion Analysis.
- Keyword based analysis is also given for both sentiment and emotion based analysis.
- Ask questions on particular audio transcription and dataset. 

## Technology Used
### Document QnA Chatbot:
- **Python**: Core development language.
- **LLM (Large Language Models)**: Downloaded from Huggingface.
- **FAISS**: Vector database to store and retrieve document embeddings.
- **Streamlit**: Used for building the user interface.
- **LangChain**: Framework used for integrating LLMs and vector stores.

### Sentiment Analysis:
- **Python**: Core programming language.
- **Streamlit**: For building the chatbot's user interface.
- **NumPy (1.26.4)**: library for numerical computing
- **whisper**: For audio file handling.
- **SciPy (1.14.0)**: library used for scientific and technical computing.
- **tqdm**: Create progress bars.
- **Transformers (4.42.4):**: pre-trained models for natural language processing (NLP) asks like text classification, question answering, translation etc.
- **Torch (2.4.0)**: It provides dynamic computational graphs and is popular for training neural networks and working on NLP.
- **Plotly (5.23.0)**: data visualization
- **TF-Keras (2.17.0)**: A high-level neural networks API that runs on top of TensorFlow.
- **SpeechRecognition (3.10.4)**: It allows Python programs to recognize spoken language and convert it to text.
- **nltk (3.9.1)**: NLTK provides tools for text processing, including classification, tokenization, stemming, tagging, parsing, and semantic reasoning.

---
 
## Installation
### Document QnA Chatbot:
To run the project locally, follow these steps:
### 1. Clone the repository and navigate to the project directory
Open Git Bash or your command line, and run the following commands to clone the repo and switch to the `document_qna_chatbot` branch:
```bash
git clone https://github.com/Yash-practice/bigDataPractice-llm.git
cd bigDataPractice-llm
git checkout document_qna_chatbot
```
### 2. Install Requirement file
To install requirement.txt
Go to this path bigDataPractice-llm\Chatbot-Codebase-Sentiment-Analysis\
Open command prompt
```bash
cd bigDataPractice-llm\Chatbot-Codebase-Doc
pip install -r requirements.txt
```
### 3. Downloading the Model
- Download the LLM model llama-2-7b-chat from this link https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin?download=true.
- Place the downloaded .bin file in the models folder within the Chatbot-Codebase-Doc directory.
### 4. Run the Streamlit application
To run the chatbot, navigate to the project directory and run the following command:
```bash
streamlit run script.py
```
### Sentiment Analysis
To run the project locally, follow these steps:
### 1. Clone the repository and switch to the sentiment analysis branch
Open Git Bash or your command line, and run the following commands to clone the repo and switch to the `sentiment_analysis` branch:
```bash
git clone https://github.com/Yash-practice/bigDataPractice-llm.git
cd bigDataPractice-llm
git checkout sentiment_analysis
```
### 2. Unzip the models
- Download the models using the links provided in the model link section.
- Unzip the models and put it inside bigDataPractice-llm\model_store\Chatbot-Codebase-Sentiment-Analysis\
```bash
- roberta-base-go_emotions
- twitter-roberta-base-sentiment-latest
- all-MiniLM-L6-v2
- llama-2-7b-chat.ggmlv3.q4_0.bin
```
### 3. Install Requirement file
install requirement.txt go to this path bigDataPractice-llm\Chatbot-Codebase-Sentiment-Analysis\
```bash
open command prompt
cmd -> pip install -r requirements.txt
```
### 4. Install ffmpeg (Required for audio file processing)
**For Windows :**
- Download ffmpeg-release-full.7z from https://www.gyan.dev/ffmpeg/builds/
- Extract the zip file to the C: drive.
- Open Environment Variables on your system.
- In User Variables, double-click on the Path variable.
- Paste the path to the bin folder of the extracted file (e.g., C:\ffmpeg-7.0.2-full_build\ffmpeg-7.0.2-full_build\bin).
- Click Apply and then OK.
### 5. Create a temp_files directory
Inside the Chatbot-Codebase-Sentiment-Analysis folder, create a folder named temp_files:
```bash
mkdir bigDataPractice-llm\Chatbot-Codebase-Sentiment-Analysis\temp_files
```
### 6. Run the Streamlit application
To run the chatbot, navigate to the project directory and run the following command:
```bash
cd bigDataPractice-llm\Chatbot-Codebase-Sentiment-Analysis
streamlit run script.py
```

---

## License
This project is licensed under the permissive open-source MIT License.

- [cardiffnlp/twitter-roberta-base-sentiment-latest]( 
https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
- [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [llama-2-7b-chat.ggmlv3.q4_0.bin](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin?download=true)

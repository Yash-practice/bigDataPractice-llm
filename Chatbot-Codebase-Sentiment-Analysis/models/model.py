from constants import analysis_type_constant, model_constant
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from langchain_text_splitters import CharacterTextSplitter
import numpy as np
from scipy.special import softmax
from sentence_transformers import SentenceTransformer
import streamlit as st
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


domain_model = {
    analysis_type_constant.GENERAL: [model_constant.TWITTER_ROBERTA_BASE_SENTIMENT_LATEST,"Sentiment"],
    analysis_type_constant.SOCIAL_MEDIA: [model_constant.ROBERTA_BASE_GO_EMOTIONS,"Emotion"]
}

def load_roberta_model(model, tokenizer):
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
    model = RobertaForSequenceClassification.from_pretrained(model)
    return tokenizer,model

def split_text(text, tokenizer, max_length=512, overlap=50):
    splitter = CharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=max_length, chunk_overlap=overlap)
    chunks = splitter.split_text(text)
    return chunks

def predict_sentiment(text, model, tokenizer, sentiment_mapping):     
    chunks = split_text(text, tokenizer)
    probs = {value:0 for key,value in sentiment_mapping.items()}
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=512)
        output = model(**inputs)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        for ranks in ranking:
            probs[sentiment_mapping[ranks]] += scores[ranks]
    
    sentiment = ''
    score = -1
    for key, value in probs.items():
        if value > score:
            score = value
            sentiment = key
        
    return {
        'output' : sentiment,
        'probs' : probs
    }

@st.cache_data(show_spinner=False)  
def load_minilm_embedding_model():
    embedding_model = SentenceTransformer(model_constant.SENTENCE_TRANSFORMER)
    return embedding_model
    
def encode_text(embedding_model,text):
    return embedding_model.encode(text)

@st.cache_data(show_spinner=False)
def create_speech_recognition_pipeline():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # model_id = "openai/whisper-large-v3"
    model_name = "../models_store/Chatbot-Codebase-Sentiment-Analysis/whisper-large-v3/model"
    processor_name = "../models_store/Chatbot-Codebase-Sentiment-Analysis/whisper-large-v3/preprocessor"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(processor_name)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    return pipe
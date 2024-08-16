from tqdm.auto import tqdm
from models import model
import streamlit as st
import pandas as pd
import time

def social_media_analysis(domain_name):
        
    st.markdown(
    f'''
        <style>
            .selectbox {{
                width: 175px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)    
    uploaded_file = st.file_uploader('Choose a Dataset(First line Will Be used as header)', type='csv', accept_multiple_files=False)
    model_name = model.domain_model[domain_name][0]
    if uploaded_file is not None and model_name:
        df = pd.read_csv(uploaded_file, header=0, encoding='ISO-8859-1', on_bad_lines='skip', nrows=100)
        st.selectbox(label = "", options=df.columns, index=None, placeholder="Select Column For Prediction", label_visibility="collapsed")
        tokenizer, model_instance = model.load_roberta_model(f'models/{model_name}/model', f'models/{model_name}/tokenizer')
        sentiment_mapping = model_instance.config.id2label
        res = []
        with st.spinner('Processing the file...'):
            for _, row in tqdm(df.iterrows(), total=len(df)):
                res.append(model.predict_sentiment(row["Review_Text"], model_instance, tokenizer, sentiment_mapping)['output'])
        st.success("Processing Complete!")
        df['label'] = res
        st.write(df)
        
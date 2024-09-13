from tqdm.auto import tqdm
from module.Sentence_Extraction import Sentence_Extractor
from models import model
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import uuid
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import torch
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from constants import analysis_type_constant
import math

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

@st.cache_data(show_spinner=False)
def split_frame(input_df, rows):
    df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df

def load_data(file):
    df = pd.read_csv(file, header=0, encoding='ISO-8859-1', on_bad_lines='skip')
    return df

@st.cache_data(show_spinner=False)
def get_extended_dataframe(reviews_df, column_name):
    rows = []
    for _, row in reviews_df.iterrows():
        rows.extend(Sentence_Extractor.split_review_text_to_rows(row, column_name))
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def get_analysis(df, domain_name):
    df.reset_index(inplace=True)
    model_name = model.domain_model[domain_name][0]
    tokenizer, model_instance = model.load_roberta_model(f'models/{model_name}/model', f'models/{model_name}/tokenizer')
    mapping = model_instance.config.id2label
    res = {}
    for _, row in tqdm(df.iterrows(), total=len(df)):
        myid = row["index"]
        try:
            res[myid] = [model.predict_sentiment(row[st.session_state["review_text_column"]], model_instance, tokenizer, mapping)['output']]
        except RuntimeError:
            print(f"Broke for id {myid}")
    analysed_df = (
        pd.DataFrame(res)
        .T.reset_index()
        .rename(columns={"index": "index", 0: "Reviewer_"+model.domain_model[domain_name][1]})
    )
    df = analysed_df.merge(df, how="left")
    df.drop(columns=["index"],inplace=True)
    return df

@st.cache_data(show_spinner=False)
def index_creation(df):
    embedding_model = model.load_minilm_embedding_model()
    df["embeddings"] = df[st.session_state["review_text_column"]].apply(lambda text: model.encode_text(embedding_model, text))
    embeddings = np.array(df['embeddings'].tolist()).astype('float32')
    dimension = embeddings.shape[1]
    faissindex = faiss.IndexFlatL2(dimension)
    faissindex.add(embeddings)
    df.drop(columns=["embeddings"],inplace=True)
    return df,faissindex

def create_vector_store(df,faiss_index):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    embeddings = HuggingFaceEmbeddings(model_name='models/all-MiniLM-L6-v2', model_kwargs={'device': device})
    index_to_docstore_id = {index: str(uuid.uuid4()) for index, row in df.iterrows()}
    documents = {}
    for index, row in df.iterrows():
        metadata = row.drop(st.session_state["review_text_column"]).to_dict()
        documents[index_to_docstore_id[index]] = Document(page_content=row[st.session_state["review_text_column"]],metadata=metadata)
    docstore = InMemoryDocstore(documents)
    vector_store = FAISS(embeddings, index=faiss_index, docstore = docstore, index_to_docstore_id = index_to_docstore_id)   
    return vector_store

@st.cache_data(show_spinner=False)
def create_prompt():                     
    custom_prompt_template = """Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}
                            
        Only return the helpful answer below and nothing else.
        Helpful answer:
    """  
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_path = "models/llama-2-7b-chat.ggmlv3.q4_0.bin"
llm = CTransformers(model=model_path, model_type="llama", config={'max_new_tokens': 512, 'temperature': 0.6}, gpu_layers=50, device=device)
 
def dataset_analysis(domain_name):
        
    tab1, tab2, tab3 = st.tabs(["Upload Dataset", "Show Report", "Ask AI"])
    with tab2:
        placeholder = st.empty()
        placeholder.write("Please Submit a Dataset")
    with tab1:    
        uploaded_file = st.file_uploader('Choose a Dataset(First line Will Be used as header)', type='csv', accept_multiple_files=False)
        if uploaded_file is None and 'analysed' in st.session_state:
            del st.session_state['analysed']
        if uploaded_file is not None:            
            df = load_data(uploaded_file)
            df = df.sample(frac=1).reset_index(drop=True)
            col1, col2 = st.columns(2,vertical_alignment="bottom")
            with col1:
                no_of_rows = st.number_input("No of rows to analyse", min_value=1, max_value=len(df), value=100)
                df = df[:no_of_rows]
            with col2:
                text_column_name = st.selectbox("Headers", df.columns, index=None, on_change=None, placeholder="Column for Prediction", label_visibility="collapsed")
            submit_btn_clicked = st.button("Submit", type="primary", disabled=False, use_container_width=False)    
            if no_of_rows is not None and text_column_name is not None and submit_btn_clicked:
                with st.spinner('Processing the file...'):
                    with tab2:
                        placeholder.write("Analyzing Dataset")
                    model_name = model.domain_model[domain_name][0]
                    if model_name:
                        tokenizer, model_instance = model.load_roberta_model(f'models/{model_name}/model', f'models/{model_name}/tokenizer')
                        sentiment_mapping = model_instance.config.id2label
                        res = []
                        for _, row in tqdm(df.iterrows(), total=len(df)):
                            try:
                                res.append(model.predict_sentiment(row[text_column_name], model_instance, tokenizer, sentiment_mapping)['output'])
                            except RuntimeError:
                                res.append('Error')
                        df[model.domain_model[domain_name][1]] = res
                        st.session_state.analysed = True
                        st.session_state.analysed_df = df
                        st.success("Success!!! Please Click On Show Report.")
            if 'analysed' in st.session_state and st.session_state.analysed==True:
                with tab2:
                    placeholder.write("")
                    with st.expander("Analyzed DataFrame", expanded=True):
                        pagination = st.container()
                        bottom_menu = st.columns((4, 1))
                        batch_size = 10
                        with bottom_menu[1]:
                            total_pages = (
                                int(len(st.session_state.analysed_df) / batch_size) if int(len(st.session_state.analysed_df) / batch_size) > 0 else 1
                            )
                            current_page = st.number_input(
                                "Page", min_value=1, max_value=total_pages, step=1
                            )
                        with bottom_menu[0]:
                            st.markdown(f"Page **{current_page}** of **{total_pages}** ")
            
                        pages = split_frame(st.session_state.analysed_df, batch_size)
                        pagination.dataframe(data=pages[current_page - 1], use_container_width=True)
        
                        csv = convert_df(st.session_state.analysed_df)
                        uploadedfilename = "".join(uploaded_file.name.split(".")[:-1])
                        st.download_button(
                            "Download",
                            csv,
                            f"{uploadedfilename}-Analyzed.csv",
                            "text/csv",
                            key='download-csv'
                        )
    with tab3:
        col1,col2 = st.columns([2,3])
        with col1:
            with st.container(height=420,border=True):
                uploaded_dataset_rag = st.file_uploader('Choose a Dataset(First line Will Be used as header)', type='csv', accept_multiple_files=False, key="dataset_file")
                if uploaded_dataset_rag:            
                    df = load_data(st.session_state['dataset_file'])
                    st.number_input("No of rows to analyse", min_value=1, max_value=len(df), value=100,key="No_Of_Rows_In_Datset")
                    df = df[:st.session_state["No_Of_Rows_In_Datset"]]
                    st.selectbox("Headers", df.columns, index=None, on_change=None, placeholder="Column for Prediction", label_visibility="collapsed",key="review_text_column")
                    st.button("Submit", type="primary", disabled=False, use_container_width=False, key="Ask_AI_Dataset_Submission")
                    st.selectbox("RAG Type", ["Ask Question", "Analyse Question"], key="rag_type" , label_visibility="collapsed")
                    if st.session_state["No_Of_Rows_In_Datset"] is not None and st.session_state["review_text_column"] is not None and st.session_state["Ask_AI_Dataset_Submission"]:
                        extended_df = get_extended_dataframe(df, st.session_state["review_text_column"])
                        extended_df = get_analysis(extended_df, analysis_type_constant.GENERAL)
                        extended_df = get_analysis(extended_df, analysis_type_constant.SOCIAL_MEDIA)
                        extended_df,faiss_index = index_creation(extended_df)
                        st.session_state['extended_df'] = extended_df
                        st.session_state['faiss_index'] = faiss_index
                        vector_store = create_vector_store(extended_df,faiss_index)
                        prompt = create_prompt()
                        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=vector_store.as_retriever(search_kwargs={'k': 10}), return_source_documents=True, chain_type_kwargs={'prompt': prompt}) 
                        st.session_state['qa_chain'] = qa_chain               
        with col2:
            with st.container(height=420,border=True):
                if uploaded_dataset_rag:
                    if 'rag_type' in st.session_state and st.session_state['rag_type']=="Ask Question":
                        with st.container(height=330,border=True):
                            if 'user_query' in st.session_state:
                                st.markdown(f'<div class="chat-response">User: {st.session_state["user_query"]}</div>', unsafe_allow_html=True) 
                                if st.session_state["user_query"]:
                                    with st.spinner('Asking....'):
                                        response = st.session_state['qa_chain'].invoke({'query': st.session_state["user_query"]})
                                        st.markdown(f'<div class="chat-response">Bot: {response['result']}</div>', unsafe_allow_html=True)
                    elif 'rag_type' in st.session_state and st.session_state['rag_type']=="Analyse Question":
                        with st.container(height=230,border=True):
                            col1,col2 = st.columns([3,2])
                            with col1:
                                with st.container(height=200,border=True):
                                    placeholder_dataframe_rows = st.empty()
                            with col2:
                                with st.container(height=200,border=True):
                                    placeholder_output = st.empty()
                                    
                        if 'user_query' in st.session_state:
                            if st.session_state['user_query']:
                                embedding_model = model.load_minilm_embedding_model()
                                query_embedding = model.encode_text(embedding_model, st.session_state["user_query"]).astype('float32')
                                D, I = st.session_state['faiss_index'].search(np.array([query_embedding]), k=len(st.session_state['extended_df']))
                                distance_threshold = st.slider("Choose a Threshold", 0.00, float(math.ceil(max(D[0]))), step=0.01)
                                
                                filtered_indices = I[0][D[0] <= distance_threshold]
                                filtered_df = st.session_state['extended_df'][st.session_state['extended_df'].index.isin(filtered_indices)]
                                ordered_filtered_df = filtered_df.loc[filtered_indices]
                                formatted_df = '\n'.join(f'{i + 1} - {value}\n' for i, value in enumerate(ordered_filtered_df.tail(5)[st.session_state["review_text_column"]]))
                                no_of_rows = len(ordered_filtered_df)
                                if no_of_rows==0:
                                    placeholder_dataframe_rows.write("No Related Data Found")
                                    placeholder_output.write("No Output")
                                else:
                                    placeholder_dataframe_rows.write(str(no_of_rows)+"  Related Data Rows Found \n\n"+formatted_df)
                                    sentiment_counts = ordered_filtered_df['Reviewer_Sentiment'].value_counts()
                                    total_rows = len(ordered_filtered_df)
                                    # Calculate the percentage for each unique value
                                    formatted_sentiments_results = "Sentiment Analysis\n\n"
                                    for col, counts in sentiment_counts.items():
                                        formatted_sentiments_results += f'{round((counts / total_rows) * 100, 2)}% of reviews are {col}\n\n'
                                    formatted_sentiments_results += "----------------------------\n\n Emotional Analysis \n\n"
                                    emotional_counts = ordered_filtered_df['Reviewer_Emotion'].value_counts()
                                    for col, counts in emotional_counts.items():
                                        formatted_sentiments_results += f'{round((counts / total_rows) * 100, 2)}% of reviews are {col}\n\n'
                                    placeholder_output.write(formatted_sentiments_results)
                    st.text_input("text_input_user_query", value="", max_chars=None, key="user_query", type="default", help=None, autocomplete=None, on_change=None, args=None, kwargs=None, placeholder="Write Your Queries", disabled=False, label_visibility="collapsed")
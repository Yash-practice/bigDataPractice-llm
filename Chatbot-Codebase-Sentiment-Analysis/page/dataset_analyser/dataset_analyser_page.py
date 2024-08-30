from tqdm.auto import tqdm
from models import model
import streamlit as st
import pandas as pd

@st.cache_data
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

@st.cache_data(show_spinner=False)
def split_frame(input_df, rows):
    df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df

@st.cache_data(show_spinner=True)
def load_data(file):
    df = pd.read_csv(file, header=0, encoding='ISO-8859-1', on_bad_lines='skip', nrows=100)
    return df
 
def dataset_analysis(domain_name):
        
    tab1, tab2 = st.tabs(["Upload Dataset", "Show Report"])
    with tab2:
        placeholder = st.empty()
        placeholder.write("Please Submit a Dataset")
    with tab1:    
        uploaded_file = st.file_uploader('Choose a Dataset(First line Will Be used as header)', type='csv', accept_multiple_files=False)
        if uploaded_file is None and 'analysed' in st.session_state:
            del st.session_state['analysed']
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            text_column_name = st.selectbox("Headers", df.columns, index=None, on_change=None, placeholder="Column for Prediction", label_visibility="collapsed")
            submit_btn_clicked = st.button("Submit", type="primary", disabled=False, use_container_width=False)    
            if submit_btn_clicked:
                with st.spinner('Processing the file...'):
                    with tab2:
                        placeholder.write("Analyzing Dataset")
                    model_name = model.domain_model[domain_name][0]
                    if model_name:
                        tokenizer, model_instance = model.load_roberta_model(f'{model_name}/model', f'{model_name}/tokenizer')
                        sentiment_mapping = model_instance.config.id2label
                        res = []
                        for _, row in tqdm(df.iterrows(), total=len(df)):
                            try:
                                res.append(model.predict_sentiment(row[text_column_name], model_instance, tokenizer, sentiment_mapping)['output'])
                            except RuntimeError:
                                res.append('Error')
                        df['Emotion'] = res
                        st.session_state.analysed = True
                        st.session_state.analysed_df = df
                        st.success("Success!!! Please Click On Show Report.")
            if 'analysed' in st.session_state and st.session_state.analysed==True:
                with tab2:
                    placeholder.write("")
                    with st.expander("Analyzed DataFrame", expanded=False):
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
                        st.download_button(
                            "Download",
                            csv,
                            f"{"".join(uploaded_file.name.split(".")[:-1])}-Analyzed.csv",
                            "text/csv",
                            key='download-csv'
                        )
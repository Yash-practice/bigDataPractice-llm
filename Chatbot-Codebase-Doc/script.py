import streamlit as st
import os
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import CTransformers
from googletrans import Translator
import sqlite3
import streamlit as st


# Path configurations
DATA_PATH = 'data/'
VECTORSTORE_PATH = 'vectorstore/'

# Step 2: Add a title to your Streamlit Application on Browser
st.set_page_config(page_title="QA Chatbot ðŸ¤–")

# Custom CSS for hover tooltip
st.markdown("""
    <style>
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: pointer;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        max-width: 600px; /* Adjust max-width as needed */
        background-color: black;
        color: #fff;
        text-align: center;
        padding: 5px 10px;
        border-radius: 6px;
        position: absolute;
        z-index: 1;
        bottom: 125%; /* Position the tooltip above the text */
        left: 50%;
        margin-left: -300px; /* Center the tooltip */
        opacity: 0;
        transition: opacity 0.3s;
        white-space: normal; /* Allow text to wrap */
        word-wrap: break-word; /* Ensure long words break */
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# Create a Sidebar
with st.sidebar:
    st.title("QA Chatbot ðŸ¤–")
    st.header("Settings")

    st.subheader("Domain Selection")
    domain = st.selectbox("Choose a Domain", ['Medical', 'KSA', 'Legal'], key='select_domain')

    # Update paths based on domain selection
    domain_path = os.path.join(DATA_PATH, domain)
    vectorstore_path = os.path.join(VECTORSTORE_PATH, f'db_faiss_{domain}')

    st.subheader("Models and Parameters")
    model_folder = 'models/'
    model_files = [f for f in os.listdir(model_folder) if f.endswith('.bin')]
    selected_model = st.selectbox("Choose a Model", model_files, key='select_model')
    llm_model_path = os.path.join(model_folder, selected_model)

    st.subheader("PDF Documents")
    pdf_files = [f for f in os.listdir(domain_path) if f.endswith('.pdf')]
    selected_pdf = st.selectbox("Choose a PDF Document", pdf_files, key='select_pdf')

# Function to create the vector database if it doesn't exist
def create_vector_db(data_path, db_faiss_path):
    # Determine if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check for the presence of the index file
    index_file = os.path.join(db_faiss_path, 'index.faiss')
    if not os.path.exists(index_file):
        # Load and process documents only if the index file does not exist
        loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        
        # Use the appropriate device for embeddings
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': device})
        print("Created embeddings")
        
        # Create and save the vector database
        db = FAISS.from_documents(texts, embeddings)
        print("Saving embeddings")
        db.save_local(db_faiss_path)
        return db
    else:
        # Load the existing vector database
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': device})
        print("Searching from already created embeddings")
        db = FAISS.load_local(db_faiss_path, embeddings)
        return db

# Function to load LLM model
def load_llm(model_path):
    # Determine if GPU is available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Load the model with the appropriate device and configuration
    llm = CTransformers(model=model_path, model_type="llama", config={'max_new_tokens': 512, 'temperature': 0.8}, gpu_layers=50, device=device)
    return llm

# Set custom prompt template
def set_custom_prompt():
    custom_prompt_template = """Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

# Create RetrievalQA chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={'k': 2}), return_source_documents=True, chain_type_kwargs={'prompt': prompt})
    return qa_chain

# Initialize the QA bot
def qa_bot(db_faiss_path, model_path):
    db = create_vector_db(domain_path, db_faiss_path)
    llm = load_llm(model_path)
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# Translation functions
def translate_arabic_to_english(text):
    translator = Translator()
    print("Translating arabic_to_english")
    translation = translator.translate(text, src='ar', dest='en')
    return translation.text

def translate_english_to_arabic(text):
    translator = Translator()
    print("Translating english_to_arabic")
    translation = translator.translate(text, src='en', dest='ar')
    return translation.text

# FAQ Database functions
def connect_faq_db():
    conn = sqlite3.connect("QaDatabase/faq_database.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS faq (question TEXT, answer TEXT)")
    return conn, cursor

def add_faq(cursor, question, answer):
    cursor.execute("INSERT INTO faq (question, answer) VALUES (?, ?)", (question, answer))
    cursor.connection.commit()

def get_answer_from_database(cursor, question):
    cursor.execute("SELECT answer FROM faq WHERE question=?", (question,))
    result = cursor.fetchone()
    return result[0] if result else None

# Connect to the FAQ database
faq_conn, faq_cursor = connect_faq_db()

# Store the LLM Generated Response
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Function to create tooltip HTML for message
def create_tooltip(message, translation):
    return f'<div class="tooltip">{message}<span class="tooltiptext">{translation}</span></div>'

# Display the chat messages with tooltips
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Only translate user messages
        if message["role"] == "user":
            # Translate user messages for tooltip
            translator = Translator()
            detected_lang = translator.detect(message["content"]).lang
            if detected_lang == 'ar':
                translated_text = translate_arabic_to_english(message["content"])
                st.markdown(create_tooltip(message["content"], translated_text), unsafe_allow_html=True)
            else:
                st.write(message["content"])
        else:
            # Display assistant messages directly
            st.write(message["content"])

# Clear the Chat Messages
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function to generate the response using the LLM and FAISS
def generate_response(prompt_input, domain):
    # Detect the language of the input text
    translator = Translator()
    detected_lang = translator.detect(prompt_input).lang

    english_prompt = prompt_input
    arabic_response = ""
    source_info = []

    if domain == 'KSA' and detected_lang == 'ar':
        # If the input is in Arabic, translate to English for processing
        english_prompt = translate_arabic_to_english(prompt_input)

    # Check for answer in the FAQ database
    faq_answer = get_answer_from_database(faq_cursor, english_prompt)
    if faq_answer:
        print("Giving answer from database")
        english_response = faq_answer
    else:
        qa = qa_bot(vectorstore_path, llm_model_path)
        response = qa({'query': english_prompt})
        english_response = response["result"]
        # Extract source information
        source_info = [doc.metadata.get('page', 'Unknown page') for doc in response['source_documents']]
        # Add the new FAQ to the database
        print("Saving answer to database")
        add_faq(faq_cursor, english_prompt, english_response)
    
    if domain == 'KSA' and detected_lang == 'ar':
        # If the input was in Arabic, translate the response to Arabic
        arabic_response = translate_english_to_arabic(english_response)
    else:
        arabic_response = english_response

    return arabic_response, english_response, source_info

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="tooltip">{prompt}<span class="tooltiptext">{translate_arabic_to_english(prompt)}</span></div>', unsafe_allow_html=True)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            arabic_response, english_response, source_info = generate_response(st.session_state.messages[-1]["content"], domain)
            
            if domain == 'KSA':
                translator = Translator()
                detected_lang = translator.detect(st.session_state.messages[-1]["content"]).lang
                if detected_lang == 'ar':
                    st.markdown(f'<div class="tooltip">{arabic_response}<span class="tooltiptext">{english_response}</span></div>', unsafe_allow_html=True)
                    st.write(arabic_response)
                else:
                    st.write(english_response)
            else:
                st.write(arabic_response)

            # Display source information
            if source_info:
                st.write("Source Pages:")
                for page in source_info:
                    st.write(f"- Page {page}")

    message = {"role": "assistant", "content": arabic_response if domain != 'KSA' else arabic_response}
    st.session_state.messages.append(message)

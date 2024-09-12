import re
import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from langchain.llms import CTransformers
from googletrans import Translator
import sqlite3
import streamlit as st
import torch
import speech_recognition as sr
import base64
import html
from google.cloud import texttospeech
from dotenv import load_dotenv

load_dotenv()

# Path configurations
DATA_PATH = 'data/'
VECTORSTORE_PATH = 'vectorstore/'

gen_parms = {
  GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
  GenParams.TEMPERATURE: 0.3,
  GenParams.MIN_NEW_TOKENS: 0,
  GenParams.MAX_NEW_TOKENS: 150,
}

# Step 2: Add a title to your Streamlit Application on Browser
st.set_page_config(page_title="QA Chatbot 🤖")

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
        max-width: 800px; /* Adjust max-width as needed */
        background-color: black;
        color: #fff;
        text-align: center;
        padding: 5px 10px;
        border-radius: 6px;
        position: absolute;
        z-index: 1;
        bottom: 110%; /* Position the tooltip above the text */
        left: 50%;
        margin-left: -300px; /* Center the tooltip */
        opacity: 0;
        transition: opacity 0.3s;
        white-space: pre-line; /* Preserve new lines and wrap text */
        word-break: break-word; /* Break words when necessary */
        word-wrap: break-word; /* Ensure long words break */
        overflow-wrap: break-word; /* Handle overflows gracefully */
        overflow-y: auto;  /* Scrollable tooltip */
        /* Hide any overflow content overflow: hidden;*/
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)

# Language selection at the top
if 'preferred_language' not in st.session_state:
    st.session_state.preferred_language = 'English'  # Default to English

st.sidebar.title("Language Selection")
st.sidebar.subheader("Choose your preferred language:")
selected_language = st.sidebar.selectbox("Language", ['English ', 'Arabic', 'French', 'German'], key='select_language')
# st.session_state.preferred_language = selected_language

# Use the selected language for the rest of the app
translator = Translator()

def translate_text(text, dest_language):
    if dest_language == 'Arabic':
        translation = translator.translate(text, dest='ar').text
    elif dest_language == 'French':
        translation = translator.translate(text, dest='fr').text
    elif dest_language == 'German':
        translation = translator.translate(text, dest='de').text
    elif dest_language == 'English':
        translation = translator.translate(text, dest='en').text
    else:
        translation = text
    return translation

def initialize_messages():
    initial_message = translate_text("How may I assist you today?", st.session_state.preferred_language)
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

# Check if the selected language has changed
if st.session_state.preferred_language != selected_language:
    st.session_state.preferred_language = selected_language
    initialize_messages()

# After language selection, display the main content
st.title(translate_text("Welcome to the QA Chatbot 🤖", st.session_state.preferred_language))

# Function to display PDF
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Helper function to recursively list files in a directory
def list_files_in_directory(directory, parent=''):
    files_structure = {}
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            files_structure[item] = list_files_in_directory(item_path, parent=item)
        else:
            files_structure[item] = item_path
    return files_structure

# Function to display folder and file structure with open buttons
def display_files_structure(files_structure, level=0):
    for name, content in files_structure.items():
        if isinstance(content, dict):
            st.write(f"{'  ' * level}📂 {name}")
            display_files_structure(content, level + 1)
        else:
            cols = st.columns(2)
            with cols[0]:
                st.write(f"{'  ' * (level + 1)}&emsp;&emsp;📄{name}")
            with cols[1]:
                if st.button("Open", key=f"open_{name}"):
                    st.session_state.view_pdf = True
                    st.session_state.selected_pdf = content

# Function to create the vector database if it doesn't exist
def create_vector_db(data_path, db_faiss_path):
    # Determine if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Check for the presence of the index file
    index_file = os.path.join(db_faiss_path, 'index.faiss')
    if not os.path.exists(index_file):
        all_documents = []
        
        # Traverse all subdirectories in the data_path
        for root, dirs, files in os.walk(data_path):
            loader = DirectoryLoader(root, glob='*.pdf', loader_cls=PyPDFLoader)
            documents = loader.load()
            all_documents.extend(documents)
        
        if all_documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(all_documents)
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', model_kwargs={'device': device})
            print("Created embeddings")
            db = FAISS.from_documents(texts, embeddings)
            print("Saving embeddings")
            db.save_local(db_faiss_path)
            return db
        else:
            print("No PDFs found in the specified path.")
            return None
    else:
        # Load the existing vector database
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', model_kwargs={'device': device})
        print("Searching from already created embeddings")
        db = FAISS.load_local(db_faiss_path, embeddings)
        return db

# Initialize session state variables to control which dropdown to show
if 'show_documents' not in st.session_state:
    st.session_state.show_documents = True
if 'selected_domain' not in st.session_state:
    st.session_state.selected_domain = 'Medical'
if 'llm_model_path' not in st.session_state:
    st.session_state.llm_model_path = None
if 'view_pdf' not in st.session_state:
    st.session_state.view_pdf = False  # Control PDF view state
if 'folder_files' not in st.session_state:
    st.session_state.folder_files = {}
if 'embeddings_generated' not in st.session_state:
    st.session_state.embeddings_generated = False  # To track if embeddings are generated
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Create a Sidebar
with st.sidebar:
    st.title("QA Chatbot 🤖")
    st.header("Settings")

    # Create two columns for inline buttons
    col1, col2 = st.columns(2)

    # Button to toggle document and PDF dropdown
    with col1:
        if st.button("Documents & PDFs"):
            st.session_state.show_documents = True
            st.session_state.view_pdf = False  # Reset PDF view when toggling

    # Button to toggle models and parameters dropdown
    with col2:
        if st.button("Models & Parameters"):
            st.session_state.show_documents = False
            st.session_state.view_pdf = False  # Reset PDF view when toggling

    # Conditionally render the appropriate section
    if st.session_state.show_documents:
        st.subheader("Domain Selection")
        domain = st.selectbox("Choose a Domain", ['Medical', 'KSA', 'Legal'], key='select_domain',
                              index=['Medical', 'KSA', 'Legal'].index(st.session_state.selected_domain))
        st.session_state.selected_domain = domain  # Save selected domain to session state

        # Update paths based on domain selection
        domain_path = os.path.join(DATA_PATH, domain)
        st.session_state.domain_path = domain_path
        vectorstore_path = os.path.join(VECTORSTORE_PATH, f'db_faiss_{domain}')
        st.session_state.vectorstore_path = vectorstore_path

        # List all folders and files in the selected domain folder
        files_structure = list_files_in_directory(domain_path)
        st.session_state.folder_files = files_structure

        # Display the folder and file structure with open buttons
        st.write("Folder and File Structure:")
        display_files_structure(files_structure)

        st.subheader("Upload a PDF")

        # Option to upload a PDF into an existing folder
        upload_option = st.radio("Upload Option", ["Upload into Existing Folder", "Create New Folder and Upload"])

        if upload_option == "Upload into Existing Folder":
            existing_folder = st.selectbox("Select a Folder", list(st.session_state.folder_files.keys()), key='select_upload_folder')
            uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

            if uploaded_file is not None:
                save_path = os.path.join(domain_path, existing_folder, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded {uploaded_file.name} into {existing_folder}")
                st.warning("You can only chat with this PDF after generating its embeddings")
                st.session_state.uploaded_file = uploaded_file

        elif upload_option == "Create New Folder and Upload":
            new_folder_name = st.text_input("Enter New Folder Name")
            uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

            if new_folder_name and uploaded_file is not None:
                new_folder_path = os.path.join(domain_path, new_folder_name)
                os.makedirs(new_folder_path, exist_ok=True)
                save_path = os.path.join(new_folder_path, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded {uploaded_file.name} into new folder {new_folder_name}")
                st.session_state.uploaded_file = uploaded_file

        # Show the "Generate Embeddings" button only if a PDF has been uploaded
        if st.session_state.uploaded_file is not None:
            if st.button("Generate Embeddings"):
                # Disable all other UI elements
                st.session_state.embeddings_generated = False
                st.write("Generating embeddings, please wait...")

                # Delete the previous embeddings
                if os.path.exists(vectorstore_path):
                    os.remove(os.path.join(vectorstore_path, 'index.faiss'))

                # Generate new embeddings
                print("Generating embeddings for new pdf file")
                db = create_vector_db(domain_path, vectorstore_path)

                if db:
                    st.success("Embeddings generated successfully!")
                    st.session_state.embeddings_generated = True
                else:
                    st.error("Failed to generate embeddings.")
        else:
            pass
            # st.warning("Please upload a PDF to generate embeddings.")

    else:
        st.subheader(f"Domain: {st.session_state.get('selected_domain', 'Not Selected')}")

        st.subheader("Models and Parameters")
        model_folder = 'models/'
        model_files = [f for f in os.listdir(model_folder) if f.endswith('.bin')]
        model_files.append("allam-1-13b-instruct")
        if st.session_state.get('selected_domain') == 'KSA':
            selected_model = st.selectbox("Choose a Model", ["allam-1-13b-instruct"], key='select_model')
        else:
            selected_model = st.selectbox("Choose a Model", model_files, key='select_model')
        print(selected_model)
        if selected_model == "allam-1-13b-instruct":
            llm_model_path = "api_path"
        else:
            llm_model_path = os.path.join(model_folder, selected_model)
        
        st.session_state.llm_model_path = llm_model_path

        temperature = st.slider('temperature', min_value=0.01, max_value=5.0, value=0.1, step=0.01)
        top_p = st.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01)
        max_length = st.slider('max_length', min_value=64, max_value=4096, value=512, step=8)

    # st.subheader("Domain Selection")
    # domain = st.selectbox("Choose a Domain", ['Medical', 'KSA', 'Legal'], key='select_domain')

    # # Update paths based on domain selection
    # domain_path = os.path.join(DATA_PATH, domain)
    # vectorstore_path = os.path.join(VECTORSTORE_PATH, f'db_faiss_{domain}')

    # st.subheader("Models and Parameters")
    # model_folder = 'models/'
    # model_files = [f for f in os.listdir(model_folder) if f.endswith('.bin')]
    # selected_model = st.selectbox("Choose a Model", model_files, key='select_model')
    # # llm_model_path = os.path.join(model_folder, selected_model)
    # llm_model_path = "dummy_path"

    # st.subheader("PDF Documents")
    # pdf_files = [f for f in os.listdir(domain_path) if f.endswith('.pdf')]
    # selected_pdf = st.selectbox("Choose a PDF Document", pdf_files, key='select_pdf')

# Main screen logic for displaying the PDF
if st.session_state.view_pdf and st.session_state.selected_pdf:
    pdf_path = st.session_state.selected_pdf

    # Use base64 to display the actual PDF
    show_pdf(pdf_path)

    # Button to close the PDF view
    if st.button("Close PDF"):
        st.session_state.view_pdf = False



# Function to load LLM model
def load_llm(model_path):
    # Determine if GPU is available
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Load the model with the appropriate device and configuration
    if model_path == "api_path":
        watsonx_llm = WatsonxLLM(
            model_id="sdaia/allam-1-13b-instruct",
            url=os.getenv('api_url'),
            project_id=os.getenv('project_id'),
            apikey=os.getenv('api_key'),
            params=gen_parms,
        )
        print("Using Allam...")
        return watsonx_llm
    else:
        llm = CTransformers(model=model_path, model_type="llama", config={'max_new_tokens': 512, 'temperature': 0.8}, gpu_layers=50, device=device)
        print("Using Local LLM...")
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
    db = create_vector_db(st.session_state.domain_path, db_faiss_path)
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

# Function for speech-to-text
def record_audio():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Recording...")
        audio = recognizer.listen(source)
        st.write("Processing...")
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.write("Sorry, I did not understand that.")
            return ""
        except sr.RequestError:
            st.write("Sorry, there was an error connecting to the speech recognition service.")
            return ""

# Function for text-to-speech
def speak_text(text):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Play the audio response
    st.audio(response.audio_content, format="audio/mp3")

# FAQ Database functions
def connect_faq_db():
    conn = sqlite3.connect("QaDatabase/faq_database.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS faq (question TEXT, answer TEXT, page_number TEXT)")
    return conn, cursor

def add_faq(cursor, question, answer, page_number):
    cursor.execute("INSERT INTO faq (question, answer, page_number) VALUES (?, ?, ?)", (question, answer, page_number))
    cursor.connection.commit()

def get_answer_from_database(cursor, question):
    cursor.execute("SELECT answer, page_number FROM faq WHERE question=?", (question,))
    result = cursor.fetchone()
    return (result[0], result[1]) if result else (None, None)

# Connect to the FAQ database
faq_conn, faq_cursor = connect_faq_db()

# Store the LLM Generated Response
if "messages" not in st.session_state.keys():
    # initial_message = translate_text("How may I assist you today?", st.session_state.preferred_language)
    # st.session_state.messages = [{"role": "assistant", "content": initial_message}]
    initialize_messages()

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
                # translated_text = translate_arabic_to_english(message["content"])
                # print(translated_text)
                translated_text = translate_text(message["content"], 'English')
                st.markdown(create_tooltip(message["content"], translated_text), unsafe_allow_html=True)
            elif detected_lang == 'fr':
                translated_text = translate_text(message["content"], 'English')
                st.markdown(create_tooltip(message["content"], translated_text), unsafe_allow_html=True)
            elif detected_lang == 'de':
                translated_text = translate_text(message["content"], 'English')
                st.markdown(create_tooltip(message["content"], translated_text), unsafe_allow_html=True)
            else:
                st.write(message["content"])
        else:
            # Display assistant messages directly
            st.markdown(message["content"], unsafe_allow_html=True)

# Clear the Chat Messages
def clear_chat_history():
    # initial_message = translate_text("How may I assist you today?", st.session_state.preferred_language)
    # st.session_state.messages = [{"role": "assistant", "content": initial_message}]
    initialize_messages()

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Buttons for voice features
if st.button("🎙️"):
    user_input = record_audio()
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

if st.button("🔊"):
    if st.session_state.messages:
        response_text = st.session_state.messages[-1]["content"]
        speak_text(response_text)

# Function to generate the response using the LLM and FAISS
def generate_response(prompt_input, domain):
    # Detect the language of the input text
    translator = Translator()
    detected_lang = translator.detect(prompt_input).lang

    llm_model_path = st.session_state.get('llm_model_path')

    english_prompt = prompt_input
    # arabic_response = ""
    lang_response = ""
    source_info = []

    if domain == 'KSA' and detected_lang == 'ar':
        # If the input is in Arabic, translate to English for processing
        # english_prompt = translate_arabic_to_english(prompt_input)
        english_prompt = prompt_input
    elif detected_lang != 'en':
        # If the input is in other language, translate to English for processing
        english_prompt = translate_text(prompt_input, 'English')
    
    # Check for answer in the FAQ database
    faq_answer, faq_page = get_answer_from_database(faq_cursor, english_prompt)
    if faq_answer:
        print("Giving answer from database")
        english_response = faq_answer
        source_info = [faq_page]  # Use the stored page number
    else:
        qa = qa_bot(st.session_state.vectorstore_path, llm_model_path)
        response = qa({'query': english_prompt})
        english_response = response["result"]
        # Extract source information
        print(response["source_documents"])
        # source_info = ["Doc: " + os.path.join(*doc.metadata.get('source').split(os.sep)[2:]) + 
        #                ", Page: " + str(doc.metadata.get('page', 'Unknown page'))
        #                for doc in response['source_documents']]
        for doc in response['source_documents']:
            source_path = doc.metadata.get('source')
            source_path = source_path.replace("\\", "/")
            normalized_path = os.path.normpath(source_path)
            doc_path = os.path.join(*normalized_path.split(os.sep)[2:])
            page_no = str(doc.metadata.get('page', 'Unknown page'))
            source_info.append("Doc: " + doc_path + ", Page: " + page_no)
        
        # Add the new FAQ to the database
        print("Saving answer to database")
        add_faq(faq_cursor, english_prompt, english_response, source_info[0])
    
    if domain == 'KSA' and detected_lang == 'ar':
        # If the input was in Arabic, translate the response to Arabic
        # arabic_response = translate_english_to_arabic(english_response)
        lang_response = english_response
        english_response = translate_arabic_to_english(lang_response)
    elif detected_lang == 'fr':
        # If the input was in French, translate the response to French
        lang_response = translate_text(english_response, 'French')
    elif detected_lang == 'de':
        # If the input was in German, translate the response to German
        lang_response = translate_text(english_response, 'German')
    else:
        lang_response = english_response

    return lang_response, english_response, source_info

def preprocess_response(response):
    # Escape HTML, replace multiple newlines with <br>
    response = html.escape(response)
    response = re.sub(r'\n+', '<br>', response)
    return response.rstrip(':').strip() + '.'

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        eng_prompt = translate_text(prompt, 'English')
        st.markdown(f'<div class="tooltip">{prompt}<span class="tooltiptext">{eng_prompt}</span></div>', unsafe_allow_html=True)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            lang_response, english_response, source_info = generate_response(st.session_state.messages[-1]["content"], st.session_state.selected_domain)
            domain = st.session_state.selected_domain
            translator = Translator()
            detected_lang = translator.detect(st.session_state.messages[-1]["content"]).lang
            if domain == 'KSA':
                if detected_lang == 'ar':
                    arabic_response = preprocess_response(lang_response)
                    english_response = preprocess_response(english_response)
                    tooltip_html = f"""<div class="tooltip">{arabic_response}<span class="tooltiptext">{english_response}</span></div>"""
                    st.markdown(tooltip_html, unsafe_allow_html=True)
                else:
                    arabic_response = preprocess_response(lang_response)
                    st.write(english_response)
            elif detected_lang != 'en':
                translator = Translator()
                detected_lang = translator.detect(st.session_state.messages[-1]["content"]).lang
                lang_response = preprocess_response(lang_response)
                english_response = preprocess_response(english_response)
                tooltip_html = f"""<div class="tooltip">{lang_response}<span class="tooltiptext">{english_response}</span></div>"""
                st.markdown(tooltip_html, unsafe_allow_html=True)
            else:
                st.write(lang_response)

            # Display source information
            if source_info:
                st.write("Source:")
                for page in source_info:
                    st.write(f"{page}")

    message = {"role": "assistant", "content": lang_response if domain != 'KSA' else arabic_response}
    st.session_state.messages.append(message)

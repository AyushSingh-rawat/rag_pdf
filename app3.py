import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates2 import css, bot_template, user_template
from langchain.prompts import PromptTemplate
import speech_recognition as sr
from streamlit_mic_recorder import speech_to_text
from gtts.lang import tts_langs
import streamlit as st
from gtts import gTTS
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser





# Custom question prompt
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)


# Functions for PDF processing and vector store
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    os.environ['HF_TOKEN'] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversationchain(vectorstore):
    os.environ['GROQ_API_KEY2'] = os.getenv("GROQ_API_KEY2")
    groq_api_key = os.getenv("GROQ_API_KEY2")
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory
    )
    return conversation_chain


def listen_to_user():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio)
        print(f"User said: {query}")
        return query
    except sr.UnknownValueError:
        print("Could not understand audio")
    except sr.RequestError:
        print("Could not request results; check your network connection")
    return ""

def handle_userinput(user_question):
    # Check if the conversation chain has been set up
    if st.session_state.conversation is None:
        st.error("Please upload and process documents before asking questions.")
        return

    response_container = st.container()    
    response = st.session_state.conversation({'question': user_question})

    # Update the chat history in session state
    st.session_state.chat_history = response['chat_history']
     # Add button to listen to the response
    bot_response = response['answer']

    tts = gTTS(bot_response, lang='en')  # You can change the language code if necessary
    temp_audio_dir = "temp_audio"
    os.makedirs(temp_audio_dir, exist_ok=True)
    audio_file = os.path.join(temp_audio_dir,f"responseaudio{len(st.session_state.chat_history) // 2 + 1}.mp3")

        # Save the TTS audio to a file in memory
    tts.save(audio_file)

        # Play the audio
    st.audio(audio_file)
    # Display each message in the chat history in reverse order
    for i, message in enumerate(reversed(st.session_state.chat_history)):
        cleaned_message = message.content.replace("According to the provided context, ", "")
        if i % 2 != 0:
            st.write(user_template.replace("{{MSG}}", cleaned_message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", cleaned_message), unsafe_allow_html=True)


# def handle_userinput(user_question):
#     # Check if the conversation chain has been set up
#     if st.session_state.conversation is None:
#         st.error("Please upload and process documents before asking questions.")
#         return

#     response_container = st.container()    
#     response = st.session_state.conversation({'question': user_question})
    
#     # Update the chat history in session state
#     st.session_state.chat_history = response['chat_history']
#      # Add button to listen to the response
#     bot_response = response['answer']
        
#     tts = gTTS(bot_response, lang='en')  # You can change the language code if necessary
#     temp_audio_dir = "temp_audio"
#     os.makedirs(temp_audio_dir, exist_ok=True)
#     audio_file = os.path.join(temp_audio_dir,f"response_audio_{len(st.session_state.chat_history) // 2 + 1}.mp3")
            
#     # Save the TTS audio to a file in memory
#     tts.save(audio_file)
            
#     # Play the audio
#     st.audio(audio_file)
#     # Display each message in the chat history in reverse order
#     for i, message in enumerate((st.session_state.chat_history)):
#         cleaned_message = message.content.replace("According to the provided context, ", "")
#         if i % 2 == 0:
#             st.write(user_template.replace("{{MSG}}", cleaned_message), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}", cleaned_message), unsafe_allow_html=True)





def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    if "hold_active" not in st.session_state:
        st.session_state.hold_active = False

    # Retain the vector store to avoid re-processing
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Chat with multiple PDFs :books:")

    if st.sidebar.button("New Chat"):
        st.session_state.conversation = None
        st.session_state.chat_history = None
        if st.session_state.vectorstore:
            # Recreate conversation chain with existing vector store
            st.session_state.conversation = get_conversationchain(st.session_state.vectorstore)
        st.success("Chat history cleared!")


    user_question1 = st.text_input("Ask a question about your documents:")
    spoken_text_placeholder = st.empty()

    

    # # Toggle hold button for voice input

    user_question2 = speech_to_text(language="en")
    if user_question2:
        spoken_text_placeholder.text_area("You said:", user_question2, height=70)


    if user_question1:
        handle_userinput(user_question1)
    elif user_question2:
        handle_userinput(user_question2)
    
    

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'PROCESS'", accept_multiple_files=True)
        if st.button("PROCESS"):
            if pdf_docs:  # Only process if new documents are uploaded
                with st.spinner("Processing"):
                    # Process PDF content as before
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversationchain(st.session_state.vectorstore)
                st.success('Document is processed!', icon="✅")
            else:
                st.info("Document is already processed. Start asking questions!")

                
    # model = os.environ['GROQ_API_KEY2'] = os.getenv("GROQ_API_KEY2")
    # groq_api_key = os.getenv("GROQ_API_KEY2")
    # llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
    # # chain = chat_template | model | StrOutputParser()
    # chain = LLMChain(llm=llm, prompt=chat_template)
        
    

if __name__ == '__main__':
    main()




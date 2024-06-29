import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from chat.chatWithVSDB import get_conversation_chain
from embeddings import huggingfaceEmbedding

from htmlTemplates import bot_template,css,user_template



def getPdfText(docs):
    text =""""""
    for doc in docs:
        doc_reader = PdfReader(doc)
        for page in doc_reader.pages:
            text+= page.extract_text()
    return text

textSpliter = CharacterTextSplitter(
    separator="\n",
    chunk_overlap=200,
    chunk_size=1028,
    length_function=len
    )
def extractChunks(text:str):
    return textSpliter.split_text(text)


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    if "conversation"  not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None 
        
    st.set_page_config(page_title="PDF Analyser", page_icon=":book:")
    st.header("PDF Analyser")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
        
        
    with st.sidebar:
        st.subheader("Document:")
        docs = st.file_uploader("Upload Pdf Here : ", type="pdf",accept_multiple_files=True)
        
        if st.button("Process"):  # Pass uploaded_file to the function
            with st.spinner("In Progress"):
                chunks=extractChunks(getPdfText(docs))
                # st.write(chunks)
                vsdb = huggingfaceEmbedding.get_vectore_db(chunks=chunks)
                st.session_state.conversation = get_conversation_chain(vsdb)
                
                
if __name__ == "__main__":
    main()
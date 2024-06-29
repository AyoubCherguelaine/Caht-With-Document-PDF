# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"), model_name="sentence-transformers/all-MiniLM-L6-v2"
)

from . import vectoreStore

def get_vectore_db(chunks):
    return vectoreStore.generateEmbeddingDb(chunks=chunks,embeddings=embeddings)
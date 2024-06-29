from langchain_community.vectorstores import FAISS


def generateEmbeddingDb(chunks,embeddings):
    vs_db = FAISS.from_texts(texts=chunks,embedding=embeddings)
    return vs_db
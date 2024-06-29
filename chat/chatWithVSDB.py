from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI    
    
def get_conversation_chain(vector_store):
    """
    Creates a ConversationalRetrievalChain with a ChatGoogleGenerativeAI LLM,
    ConversationBufferMemory, and the provided vector store retriever.

    Args:
        vector_store (object): A vector store object that can be used as a retriever.

    Returns:
        ConversationalRetrievalChain: The constructed conversational chain.
    """

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return conversation_chain
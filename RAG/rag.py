import os
from dotenv import load_dotenv
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from llama_index.core.indices.loading import  load_index_from_storage
from llama_index.core.postprocessor.llm_rerank import LLMRerank
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.chat_engine.condense_plus_context import CondensePlusContextChatEngine
from llama_index.embeddings.openai import (
                OpenAIEmbedding,
                OpenAIEmbeddingModelType
            )

from llama_index.core import Settings


load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")



def get_index():
    """
    Loads and returns the LlamaIndex index for health-related documents.

    - Configures the LLM (`gpt-4o-mini`) and embedding model (`text-embedding-3-large`).
    - Loads the index from storage using `StorageContext`.
    
    Returns:
        index (LlamaIndex): The health-related index for document retrieval.
    """

    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model=OpenAIEmbeddingModelType.TEXT_EMBED_3_LARGE)
    storage_context = StorageContext.from_defaults(persist_dir='./VDB-1k-samples/PDF')
    index =load_index_from_storage(storage_context, index_id='health')
    return index





    
def generate_answer(index, question,prev_conversation,topk=4): # Sometimes it is better to use only the question and not the conversation that is why i am keeping it
    """
    Generates an answer to a question using a specified index, and provides proof from the source documents.
    
    Parameters:
    - index: The index to search within.
    - question: The question to ask.
    - previous conversation: The previous conversation to provide context for the answer.
    - top_k: The number of top documents to retrieve as proof (default: 4).
    
    Returns:
    - The generated answer to the question.
    """

    llm = OpenAI(model="gpt-4o-mini")
    memory = ChatMemoryBuffer.from_defaults(token_limit=500)
    retriever = index.as_retriever(similarity_top_k=topk)
    
    chat_engine = CondensePlusContextChatEngine(
        retriever=retriever,
        llm=llm,
        memory=memory,
        context_prompt = (
            "You are a knowledgeable and professional assistant specializing in health, you are doing a conversation with a user who is asking you a question about health records, the conversation is going as follows:\n "
            "---------------------\n"
            f"{prev_conversation}\n"
            "---------------------\n"
            "you have to answer the following question from the following medical records.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Using only the provided context and avoiding reliance on prior knowledge, respond to the user's query in a clear, academic, and professional tone. "
            "If the query cannot be answered based on the provided context, politely indicate that the information is not available\n"
            
            "Answer:"
        ),
        node_postprocessors=[
        LLMRerank(
             top_n=4, llm=OpenAI(model="gpt-4o-mini")
        )
    ]


    )
    
    response = chat_engine.chat(prev_conversation)

    return response.response







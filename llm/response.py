import os
from dotenv import load_dotenv
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# === 🔧 LLM Setup ===
llm = ChatGroq(
    temperature=0.2,
    model_name="llama3-70b-8192",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# === 🧠 Conversation Memory ===
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="input",
    output_key="output"
)

# === 🧩 HR Policy Assistant Prompt ===
custom_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an HR policy assistant. Use ONLY the provided context to answer the question.\n"
     "Be accurate, concise, and easy to understand.\n"
     "If unsure or if the answer is not in the context, respond with:\n"
     "'The provided documents do not contain information about this question.'"),
    ("human", "{context}\n\nChat history:\n{chat_history}\n\nUser query: {input}")
])

# === 🔗 Chain: Prompt + LLM ===
chat_chain: RunnableSequence = custom_prompt | llm

# === 🚀 Core Function ===
def get_llm_response(query: str, reranked_docs: List[Document]) -> Tuple[str, ConversationBufferMemory]:
    """
    Given a user query and reranked docs, generate an LLM response with memory tracking.

    Args:
        query (str): User input.
        reranked_docs (List[Document]): BM25 reranked documents.

    Returns:
        Tuple[str, ConversationBufferMemory]: LLM response and memory object.
    """
    context = "\n\n".join(doc.page_content.strip() for doc in reranked_docs)

    try:
        inputs = {
            "context": context,
            "chat_history": memory.load_memory_variables({})["chat_history"],
            "input": query
        }

        response = chat_chain.invoke(inputs)

        # Save interaction to memory
        memory.save_context({"input": query}, {"output": response.content})

        return response.content.strip(), memory

    except Exception as e:
        return f"❌ LLM Error: {str(e)}", memory

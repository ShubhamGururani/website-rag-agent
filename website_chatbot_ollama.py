import os
import requests
import tempfile

from bs4 import BeautifulSoup

from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage


# -------------------------------------------------------------------
# SIMPLE MANUAL MEMORY (WORKS 100% WITH CHATOLLAMA)
# -------------------------------------------------------------------
conversation_history = []   # list[BaseMessage]


def add_to_history(message):
    conversation_history.append(message)


def get_history_messages():
    return conversation_history[:]   # copy


# -------------------------------------------------------------------
# SCRAPER
# -------------------------------------------------------------------
def fetch_html(url):
    if not url.startswith("http"):
        url = "https://" + url

    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        return r.text
    except Exception as e:
        print("Error fetching:", e)
        return None


def process_website(url):
    html = fetch_html(url)
    if not html:
        raise ValueError("Cannot load website")

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
        f.write(html)
        path = f.name

    try:
        loader = BSHTMLLoader(path)
        docs = loader.load()
    except:
        loader = BSHTMLLoader(path, bs_kwargs={"features": "html.parser"})
        docs = loader.load()
    finally:
        os.unlink(path)

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return splitter.split_documents(docs)


# -------------------------------------------------------------------
# BUILD RAG (NO RUNNABLES — PURE PYTHON)
# -------------------------------------------------------------------
def build_rag(llm, retriever):

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Context:
{context}

Question:
{question}

Answer ONLY using context. 
If not enough info, say so.
"""
    )

    def rag_fn(question: str):

        # retrieve docs
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])

        final_prompt = prompt.format(context=context, question=question)

        # MANUAL MEMORY:
        messages = get_history_messages() + [
            HumanMessage(content=final_prompt)
        ]

        # ASK OLLAMA
        response = llm.invoke(messages)

        # store in memory
        add_to_history(HumanMessage(content=question))
        add_to_history(AIMessage(content=response.content))

        return response.content

    return rag_fn


# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------
def main():

    print("\n=== Local Ollama RAG Chatbot ===")

    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0.4,
    )

    global conversation_history
    conversation_history = []  # reset memory per session

    while True:
        url = input("\nEnter website URL (or quit): ").strip()
        if url.lower() == "quit":
            return

        chunks = process_website(url)
        print(f"Loaded {len(chunks)} chunks.")

        print("\nBuilding embeddings...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        rag = build_rag(llm, retriever)

        print("\nReady! Ask questions. Type 'new' to reload.\n")

        while True:
            q = input("You: ").strip()

            if q.lower() == "quit":
                return
            if q.lower() == "new":
                conversation_history = []  # reset memory
                break

            ans = rag(q)
            print("\nRAG:", ans, "\n")


if __name__ == "__main__":
    main()

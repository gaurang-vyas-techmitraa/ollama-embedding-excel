import pandas as pd
import streamlit as st
import os

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter


# At the top of your Streamlit app
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

VECTORSTORE_PATH = "xlsx_faiss_index"

st.title("📄 Ollama Embedding Excel File")
uploaded_file = st.file_uploader("Upload a xlsx file", type="xlsx")

if uploaded_file:


    if os.path.exists(VECTORSTORE_PATH):

        embeddings = OllamaEmbeddings(model="llama3")
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
      
    else:
        # Load Excel File
        df = pd.read_excel(uploaded_file,sheet_name="Sheet1")
        text_data = df.to_string(index=False)

        # Convert DataFrame to Document
        doc = Document(page_content=text_data)

        # Split into smaller chunks
        text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=10)
        docs = text_splitter.split_documents([doc])

        # Create vector store
        embedding = OllamaEmbeddings(model='llama3')
        vectorstore = FAISS.from_documents(docs, embedding)
        vectorstore.save_local(VECTORSTORE_PATH)
        # Set up retriever
    retriever = vectorstore.as_retriever()
    st.success("✅ FAISS index created and cached.")
    # Load LLM
    llm = Ollama(model='llama3')

    # Optional: Custom prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
    You are an intelligent assistant. Use the below context to answer the user's question.

    Context:
    {context}

    Question: {question}
    Answer:"""
    )

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_template}
    )
    question = st.text_input("Ask a question about your document:") 

    if question:
        # Loop to ask questions
        with st.spinner("Thinking..."):
            answer = qa_chain.run(question)
        #st.success("Answer:")
        # Save prompt & answer
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })
        st.write(answer)
    # Show previous prompts
    st.subheader("📜 Previous Prompts")
    for i, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** {chat['question']}")
        st.markdown(f"**A{i+1}:** {chat['answer']}")

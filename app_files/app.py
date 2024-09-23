# flake8: noqa: E501

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Qdrant
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.multi_query import MultiQueryRetriever

import chainlit as cl

load_dotenv()

pdf_paths = [
    "AI_Risk_Management_Framework.pdf",
    "Blueprint-for-an-AI-Bill-of-Rights.pdf",
]

documents = []
for pdf_path in pdf_paths:
    loader = PyPDFLoader(pdf_path)
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=100,
)

docs = text_splitter.split_documents(documents)

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Qdrant.from_documents(
    documents=docs,
    embedding=embedding,
    location=":memory:",
    collection_name="Midterm Embedding Eval",
)

custom_template = """
You are an expert in artificial intelligence policy, ethics, and industry trends. Your task is to provide clear and accurate answers to questions related to AI's role in politics, government regulations, and its ethical implications for enterprises. Use reliable and up-to-date information from government documents, industry reports, and academic research to inform your responses. Make sure to consider how AI is evolving, especially in relation to the current political landscape, and provide answers in a way that is easy to understand for both AI professionals and non-experts.

Remember these key points:
1. Use "you" when addressing the user and "I" when referring to yourself.
2. If you encounter complex or legal language in the context, simplify it for easy understanding. Imagine you're explaining it to someone who isn't familiar with legal terms.
3. Be prepared for follow-up questions and maintain context from previous exchanges.
4. If there's no information from a retrieved document in the context to answer a question or if there are no documents to cite, say: "I'm sorry, I don't know the answer to that question."

Here are a few example questions you might receive:

How are governments regulating AI, and what new policies have been implemented?
What are the ethical risks of using AI in political decision-making?
How can enterprises ensure their AI applications meet government ethical standards?

One final rule for you to remember. You CANNOT under any circumstance, answer any question that does not pertain to the AI. If you do answer an out-of-scope question, you could lose your job. If you are asked a question that does not have to do with AI, you must say: "I'm sorry, I don't know the answer to that question."
Context: {context}
Chat History: {chat_history}
Human: {question}
AI:"""

PROMPT = PromptTemplate(
    template=custom_template, input_variables=["context", "question", "chat_history"]
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10},
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)

multiquery_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)


@cl.on_chat_start
async def start_chat():
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key="answer"
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=multiquery_retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )

    cl.user_session.set("qa", qa)

    await cl.Message(
        content="Hi! What questions do you have about AI?", author="AI"
    ).send()


@cl.on_message
async def main(message: cl.Message):
    qa = cl.user_session.get("qa")
    cb = cl.AsyncLangchainCallbackHandler()

    callbacks = [cb]

    response = await cl.make_async(qa)(
        {"question": message.content}, callbacks=callbacks
    )

    answer = response["answer"]
    source_pdf = response["source_documents"][0].metadata["source"]
    if source_pdf == "Blueprint-for-an-AI-Bill-of-Rights.pdf":
        source = "Blueprint for an AI Bill of Rights"
    elif source_pdf == "AI_Risk_Management_Framework.pdf":
        source = "AI Risk Management Framework"
    else:
        source = source_pdf

    page_number = str(response["source_documents"][0].metadata["page"])

    content = (
        response["answer"] + f"\n\n(**Source**: {source}, **Page**: {page_number})"
    )

    await cl.Message(content=content, author="AI").send()

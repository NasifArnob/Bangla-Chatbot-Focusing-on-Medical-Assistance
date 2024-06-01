from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
import socket
socket.getaddrinfo('localhost',8080)
import googletrans
from googletrans import Translator
translator = Translator()

Db_faiss_path = "vectorstores/db_faiss" #store embeddings in this folder

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say I don't know, don't try to make up an answer and do not look for the answer on the internet. 

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt


#Loading the model
def load_llm():
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type= "llama",
        max_new_tokens = 512,
        temperature = 0.7
    )
    return llm

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 1}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(Db_faiss_path, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response


#chainlit code fronter er
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    engMsg ="This is a Medical Bot. What is your query?"
    engMsg2 = translator.translate(engMsg, dest='bn')
    msg.content = engMsg2.text
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    messageFinal = translator.translate(message, dest='en',src='bn')
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(messageFinal.text, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]
    answer1 = translator.translate(answer, dest='bn')
   # if sources:
    #    answer1.text += f"\nSources:" + str(sources)
   # else:
    #    answer1.text += "\nNo sources found"

    await cl.Message(content=answer1.text).send()
## Credit to : https://github.com/clairelovesgravy/slack_bot_demo/blob/main/app.py

import os
from flask import Flask, request
from dotenv import load_dotenv, find_dotenv
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from key import OPENAI_API_KEY, SLACK_BOT_TOKEN, SIGNING_SECRET

# Load the environment variables
load_dotenv(find_dotenv())

slack_bot_token = SLACK_BOT_TOKEN
openai_api_key = OPENAI_API_KEY
signing_secret = SIGNING_SECRET





from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
import os
from dotenv import load_dotenv
from key import OPENAI_API_KEY

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
llm = ChatOpenAI(model="gpt-4o-mini")


import pandas as pd

file_path = ('anonymized_data_final_acronyms(in).csv') # insert the path of the csv file
data = pd.read_csv(file_path)



# Drop columns that are empty
data = data.loc[:, data.columns[:11]]  # Keep only the first 11 columns (adjust as needed)

#preview the csv file
print(data.head())

loader = CSVLoader(file_path=file_path)
docs = loader.load_and_split()


import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
index = faiss.IndexFlatL2(len(OpenAIEmbeddings().embed_query(" ")))
vector_store = FAISS(
    embedding_function=OpenAIEmbeddings(),
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

vector_store.add_documents(documents=docs)

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

retriever = vector_store.as_retriever()

# Set up system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Keep the answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    
])

# Create the question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


answer= rag_chain.invoke({"input": "How many grants have been considered since 2022 ?"})
answer['answer']





app = App(token=slack_bot_token, signing_secret=signing_secret)

flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

@app.event("app_mention")
def handle_app_mentions(body, say, logger):
    print(f"app mentioned, body : {body}, say : {say}")
    user_id = body["event"]["user"]
    say(f"Hi there, <@{user_id}>!")

    text = body["event"]["text"]
    answer= rag_chain.invoke({"input": text})
    print(f"answer : {answer}")
    say(answer['answer'])

@app.event("message")
def handle_message_events(body, say, logger):
    print(f"app message, body : {body}, say : {say}")
    user_id = body["event"]["user"]
    say(f"Hi there, <@{user_id}>!")

    text = body["event"]["text"]
    answer= rag_chain.invoke({"input": text})
    print(f"answer : {answer}")
    say(answer['answer'])

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

if __name__ == "__main__":
    flask_app.run(port=5001)
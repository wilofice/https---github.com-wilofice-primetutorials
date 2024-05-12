# Load our emails deta using a json loader : JSONLoader

from langchain_community.document_loaders import JSONLoader

def extract_email_metadata(record: dict, metadata: dict) -> dict:

    metadata["SenderName"] = record.get("SenderName")
    metadata["Subject"] = record.get("Subject")
    metadata["SenderEmail"] = record.get("SenderEmail")
    return metadata


loader = JSONLoader(
    # Make sure you have set the path to your emails extracted data
    file_path='./data/emails_2024_05_08_134553.json',
    jq_schema='.emails[]',
    text_content=False,
    metadata_func=extract_email_metadata
)

docs = loader.load()

# Split our documents into small chunks of text so that searching gets more efficient

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

# Use an embeddings model to transform our emails data into vectors
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the vectors store using Chroma
from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_function)

# Gets the VectorStoreRetriever instance we will be using to search through our vectors database
retriever = vectorstore.as_retriever()


# BUILDING OUR CHAIN

# Defining our prompt template
from langchain_core.prompts import PromptTemplate
prompt_template = PromptTemplate.from_template(
    "You are my assistant that helps manage my emails. Use the retreived context to accomplish the task you are asked. The context contains emails data extracted from my Gmail account. Analyze each email document carefully because each detail may be important to me. If you can't run the task, just say that you can't. Be explicit and give details as mush as you can. \nTask: {task} \nContext: {context} \nAnswer:"
)

# Defining the prompt inputs
from langchain_core.runnables import RunnablePassthrough
task_input = RunnablePassthrough()

# Defining our prompt context 
def format_retrieved_documents(documents):
    return "\n".join(doc.page_content for doc in documents)

retrieved_emails_data = retriever | format_retrieved_documents

prompt_inputs = {"context" : retrieved_emails_data, "task": task_input}

# Defining  the prompt to send to the LLM with the prompts inputs
prompt = prompt_inputs | prompt_template

# Instantiating our AI Brain : the LLM
from langchain_community.chat_models import ChatOllama
llm = ChatOllama(model="llama3", temperature=0.1)

# Creating the chain
chain = prompt | llm

# Last step is parsing the chain output into a readable text for human
from langchain_core.output_parsers import StrOutputParser
chain = chain | StrOutputParser()

# Invoking our chain
#query = "Check the email I received from Karen TOLE WA DOKO and suggest a dfrat email to respond to her" # Ask your question here
#print(chain.invoke(query))


def response_generator(prompt):
    return chain.invoke(prompt)
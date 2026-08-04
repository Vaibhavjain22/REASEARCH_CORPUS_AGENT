import os
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv()


# data loading 
loader=CSVLoader(file_path="./data/arxiv_paper.csv", encoding="utf-8")
documents=loader.load()

# data cleaning
from langchain_core.documents import Document
docs=[]
for doc in documents:
    Content=doc.page_content.replace("\n"," ",)
    docs.append(Document(page_content=Content))

print(f"Loaded {len(documents)} documents")


# chunking the data

text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=50)
texts=text_splitter.split_documents(docs)

print(f"Total chunks created: {len(texts)}")

# embedding the data
print("Loading embedding model...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)



# creating vector store
print("Creating ChromaDB vector store...")
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db",  # Where to save data locally
)

# storing  the chunks in the vector store
from tqdm import tqdm

BATCH_SIZE = 500
texts = texts[0:20000]

print(f"Adding {len(texts)} documents in batches of {BATCH_SIZE}...")

for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i : i + BATCH_SIZE]
    vector_store.add_documents(batch)

print("All documents added successfully!")
print(f"Total documents in ChromaDB: {vector_store._collection.count()}")

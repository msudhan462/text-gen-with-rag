from typing import Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


from vector_database import vdb

class Indexing():


    def __init__(self) -> None:
        
        self.embd_model_config = {
            "model_name" : "BAAI/bge-small-en", 
            "model_kwargs" : {"device": "cpu"}, 
            "encode_kwargs" : {"normalize_embeddings": True}
        }
    
    def load_and_split(self):
        loader = PyPDFLoader("data.pdf")
        pages = loader.load_and_split()
        print(pages[1])
        return pages
    
    def store_embeddings(self, pages):
        
        print("Loading Embedding Model..............!!!!!")
        self.hf = HuggingFaceBgeEmbeddings(
            **self.embd_model_config
        )
        test_ = "hi this is harrison"
        embedding = self.hf.embed_query(test_)
        print("Loaded........Embedding model")
        print(f"Test embeddings for {test_}= {embedding} ")
        
        print("Storing embedding vectors")
        vdb.add_vectors(
            pages=pages,
            embeddings= self.hf,
        )   
        print("Stored total embedding vectors")    
    
    def run_indexing(self):
        pages = self.load_and_split()
        self.store_embeddings(pages)
        print("Completed Indexing........")
        
i = Indexing()
i.run_indexing()
from langchain_community.vectorstores import Qdrant
import os
from constants import API_KEY


class VDB:
    
    def __init__(self) -> None:
        
        self.url="https://7c362b2c-3f21-4df3-b8e6-3deb7113a1e6.us-east4-0.gcp.cloud.qdrant.io:6333"
        self.api_key=API_KEY
        self.collection_name="test_collection"
        
    
    def add_vectors(self, pages, embeddings):
        
        doc_store = Qdrant.from_documents(
            pages, embeddings, 
            url=self.url, 
            api_key=self.api_key, 
            collection_name=self.collection_name,
            prefer_grpc=True,
        )
        return doc_store
    
    
vdb = VDB()


from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline 
from langchain.prompts import PromptTemplate
from flask import Flask, jsonify, request 
from constants import API_KEY
  
app = Flask(__name__) 
  

class Retriever:
    
    def __init__(self) -> None:
        
        self.url="https://7c362b2c-3f21-4df3-b8e6-3deb7113a1e6.us-east4-0.gcp.cloud.qdrant.io:6333"
        self.api_key=API_KEY
        self.collection_name="test_collection"
        
        client = QdrantClient(api_key = self.api_key, url=self.url, )
        
        self.embd_model_config = {
            "model_name" : "BAAI/bge-small-en", 
            "model_kwargs" : {"device": "cpu"}, 
            "encode_kwargs" : {"normalize_embeddings": True}
        }
        
        
        collection_name = "test_collection"
        self.hf = HuggingFaceBgeEmbeddings(
            **self.embd_model_config
        )
        self.qdrant = Qdrant(client, collection_name, self.hf)

    def get_context(self, text):
        
        found_docs = self.qdrant.similarity_search_with_score(text)
        return found_docs

context_retriever = Retriever()
    

llm = None
def load_llm():
    print("LOADING LLM ###############################################")
    # Specify the model name you want to use
    model_name = "Intel/dynamic_tinybert"

    global llm 
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        pipeline_kwargs={"max_new_tokens": 10},
    )
    

@app.route('/get-context', methods = ['POST']) 
def start(): 
    data = request.json
    query = data.get("query", None)
    if query:
        context = context_retriever.get_context(query)    
        d = [{"context":i[0].page_content,"metadata":i[0].metadata,"score": i[1]} for i in context]
        return jsonify({"query_output":d}) 
    else:
        return jsonify({'context': "No Query Provided" }) 



@app.route('/chat', methods = ['POST']) 
def chat(): 
    data = request.json
    query = data.get("query", None)
    
    if query:
        context = context_retriever.get_context(query)  
        
        template = """Context: {context}
        Question: {query}
        Answer: """
        prompt = PromptTemplate.from_template(template)
        
        print(llm)
        chain = prompt | llm
        
        context = context[0][0].page_content
        print(context)
        op = chain.invoke({
            "query": query,
            "context":context
        })
        print(op)
        
        return jsonify({"response":op}) 
    else:
        return jsonify({'context': "No Query Provided" }) 
    

if __name__ == '__main__':
    load_llm() 
    app.run(debug = True) 
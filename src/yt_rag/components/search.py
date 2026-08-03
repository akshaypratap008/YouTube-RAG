from src.yt_rag.logger import logging
from src.yt_rag.exceptions import CustomException
from src.yt_rag.components.data_loader import DataLoader
from src.yt_rag.components.embedding import EmbeddingManager
from src.yt_rag.components.vectorstore import FaissVectorStore

import os
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class RAGSearch:
    """
    Handles query search in faiss vector store
    """

    def __init__(self, persist_dir:str = 'faiss_store', llm_model:str= "gpt-4o-mini", url:str | None = None, model_name: str = "text-embedding-3-large"):
        self.persist_dir = persist_dir
        self.url = url
        self.model_name = model_name
        self.vectorstore = FaissVectorStore(persist_dir=self.persist_dir)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        metadata_path = os.path.join(persist_dir, "metadata.pkl")
    

        if not os.path.exists(faiss_path) or not os.path.exists(metadata_path):
            loader = DataLoader(url =self.url)
            video_id = loader.video_id
            video_data = loader.fetch_video_data()      # fetch transcript, preprocess the data and merge small segments
            chunks = loader.create_sementic_chunks(video_data=video_data)       # generate semantic chunks
            embedding_manager = EmbeddingManager()
            embeddings = embedding_manager.generate_embeddings(chunks = chunks)     # generate embeddings
            embedding_manager.save_embeddings(embeddings=embeddings)        # save them on disk
            vector_store = FaissVectorStore()
            vector_store.add_embeddings(embeddings = embeddings)     # adds embeddings and metadata and saves them inside vector store
            vector_store.load()
        else:
            self.vectorstore.load()

        try:
            self.llm = ChatOpenAI(
                model=llm_model, 
                api_key=OPENAI_API_KEY,
                temperature=0
            )

            logging.info(f"[INFO] OpenAI chat model initialised: {llm_model}")
        except Exception as e:
            raise CustomException(e, sys)

    def search(self, query:str, top_k:int=5) -> str:
        pass

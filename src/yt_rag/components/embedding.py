from src.yt_rag.logger import logging
from src.yt_rag.exceptions import CustomException
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import sys
from typing import List, Dict, Any
import numpy as np
import pickle

from src.yt_rag.utils import save_object

load_dotenv()

class EmbeddingManager:
    """
    Handles document embeddings using OpenAI  
    """
    def __init__(self, model_name:str = "text-embedding-3-large"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        "Load OpenAI embedding model"
        try:
            logging.info(f"[INFO] Loading Embedding Model: {self.model_name}")
            self.model = OpenAIEmbeddings(model = self.model_name)
        except Exception as e:
            raise CustomException(e, sys)

    def generate_embeddings(self, chunks:List[Dict[str, any]]) -> np.ndarray:
        """
        Generate embeddings from the list of dicts containing transcript text, start and end time
        Args:
            chunks: List of Dictionaries with text -> chunk text, start -> start time, end -> end time
        Returns: numpy array of embeddings of shape (len(chunks), embedding_dim)
        """

        if not self.model:
            raise ValueError("Model Not Loaded")

        logging.info(f"[INFO] Generating embeddings for {len(chunks)} chunks")
        texts = [item['text'] for item in chunks]

        embeddings = np.array(self.model.embed_documents(texts))
        logging.info(f"[INFO] Embeddings for {len(chunks)} chunks generate. Embeddings shape: {embeddings.shape}")
        return embeddings

    def save_embeddings(self, embeddings:np.ndarray, file_path:str = "data/embeddings.pkl"):
        """
        Save embeddings in the form of pickle file
        """
        save_object(obj = embeddings, file_path=file_path)
        logging.info(f"Embeddings with shape {embeddings.shape} saved on disk")

    def embed_query(self, query:str) -> np.ndarray:
        """
        Generates embeddings for the new query
        Args: query
        Returns: array of shape (1, embeding dim)
        """
        query_embedding = self.model.embed_query(text = query)      # this will return a list
        query_embedding = np.array([query_embedding])       # will return a 2D array of shape (1, embeding dim)    
        logging.info(f"[INFO] Query embedding generated for query: {query}. Query embedding dimension: {query_embedding.shape}")
        return query_embedding

        

        
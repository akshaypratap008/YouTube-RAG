import os
import sys
import faiss
import numpy as np
from typing import List, Dict, Any
import shutil

from src.yt_rag.logger import logging
from src.yt_rag.exceptions import CustomException
from src.yt_rag.utils import load_object, save_object
from pathlib import Path

class FaissVectorStore:
    """
    Vector store to store embeddings for the chunks
    """

    def __init__(self, video_id:str, persist_dir:str = "faiss_store"):
        self.video_id = video_id
        self.persist_dir = persist_dir

        if video_id and os.path.basename(self.persist_dir) != video_id:
            self.persist_dir = os.path.join(self.persist_dir, self.video_id)
            
        self.index = None
        self.metadata = None

        os.makedirs(self.persist_dir, exist_ok=True)

    def add_embeddings(self, embeddings:np.ndarray, metadata_file_path: str = "data/chunks.pkl"):
        """
        Adds embeddings to vector store as faiss.index
        Adds metadata to vector store as metadata.pkl
        Args:
            embeddings : embedding vectors
            metadata_file_path: metadata_file_path which was saved in data folder during data loading
        Returns: None
        """
        try:
            dim = embeddings.shape[1]           # dimension to initialise vector store
            if self.index is None:
                self.index = faiss.IndexFlatL2(dim)     # initialize vectorestore with dimension same as embeddings 
                logging.info(f"[INFO] Empty Faiss index initialised")
            self.index.add(embeddings)
            logging.info(f"[INFO] Embeddings with shape {embeddings.shape} added to faiss index")
            self.add_metadata(metadata_file_path = metadata_file_path)
            self.save()
        except Exception as e:
            raise CustomException(e, sys)

    def add_metadata(self, metadata_file_path:str = "data/chunks.pkl"):
        """
        Adds metadata inside vector store as pickle file. add_embeddings adds metadata as well. 
        """
        metadatas = load_object(file_path=metadata_file_path)
        if metadatas:
            if self.metadata is None:
                self.metadata = metadatas
        logging.info(f"[INFO] Metadata stored in {self.persist_dir}")
        
    def save(self):
        """
        Save faiss index and metadata in vetor store
        """
        try:
            faiss_path = os.path.join(self.persist_dir, "faiss.index")
            metadata_path = os.path.join(self.persist_dir, "metadata.pkl")
            faiss.index = faiss.write_index(self.index, faiss_path)          # write faiss index
            save_object(obj=self.metadata, file_path=metadata_path)          # save metadata inside faiss_store folder
                
            logging.info(f"[INFO] Saved faiss index and metadata to {self.persist_dir}")
        except Exception as e:
            raise CustomException(e, sys)

    def load(self):
        """
        Load faiss index and metadata
        """
        try:
            faiss_path = os.path.join(self.persist_dir, "faiss.index")
            metadata_path = os.path.join(self.persist_dir, "metadata.pkl")
            self.index = faiss.read_index(faiss_path)
            self.metadata = load_object(file_path=metadata_path)

            logging.info(f"[INFO] Faiss index and Metadata loaded")
        except Exception as e:
            raise CustomException(e, sys)

    def search(self, query_embedding:np.ndarray, top_k:int = 5) -> List[Dict[str, Any]]:
        """
        Searches the query embedding inside faiss.index and return the most similar embeddings
        Args:  
            query_embedding: numpy array of dimensions same as embeddings
            top_k: top k results based on distance
        Returns:
            List of Dict with "index", "distance" and "metadata"
        """
        if self.index is None:
            self.load()
        D, I = self.index.search(query_embedding, k=top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if 0 <= idx < len(self.metadata) else None
            results.append({
                "index": idx,
                "distance": dist,
                "metadata": meta
            })
        logging.info(f"[INFO] Vectorestore search for nearest vectors completed for query embedding of dimension {query_embedding.shape}")
        return results
        
class VectorStoreManager:

    def __init__(self, persist_dir:str = "faiss_store"):
        self.persist_dir = Path(persist_dir)

    def reset(self):
        try:
            if self.persist_dir.exists():
                shutil.rmtree(self.persist_dir)
            logging.info(f"[INFO] Faiss store cleared")
        except Exception as e:
            raise CustomException(e, sys)



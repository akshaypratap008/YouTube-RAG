import os
import sys
import faiss
import numpy as np

from src.yt_rag.logger import logging
from src.yt_rag.exceptions import CustomException
from src.yt_rag.utils import load_object, save_object

class FaissVectorStore:
    """
    Vector store to store embeddings for the chunks
    """

    def __init__(self, persist_dir:str = "faiss_store"):
        self.persist_dir = persist_dir
        self.index = None
        self.metadata = None

        os.makedirs(self.persist_dir, exist_ok=True)

    def add_embeddings(self, embeddings:np.ndarray, metadata_file_path: str = "data/chunks.pkl"):
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
        metadatas = load_object(file_path=metadata_file_path)
        if metadatas:
            if self.metadata is None:
                self.metadata = metadatas
        logging.info(f"[INFO] Metadata stored in {self.persist_dir}")
        
    def save(self):
        try:
            faiss_path = os.path.join(self.persist_dir, "faiss.index")
            metadata_path = os.path.join(self.persist_dir, "metadata.pkl")
            self.index = faiss.write_index(self.index, faiss_path)          # write faiss index
            save_object(obj=self.metadata, file_path=metadata_path)          # save metadata inside faiss_store folder
                
            logging.info(f"[INFO] Saved faiss index and metadata to {self.persist_dir}")
        except Exception as e:
            raise CustomException(e, sys)

    def load(self):
        try:
            faiss_path = os.path.join(self.persist_dir, "faiss.index")
            metadata_path = os.path.join(self.persist_dir, "metadata.pkl")
            self.index = faiss.read_index(faiss_path)
            self.metadata = load_object(file_path=self.metadata)

            logging.info(f"[INFO] Faiss index and Metadata loaded")
        except Exception as e:
            raise CustomException(e, sys)
        



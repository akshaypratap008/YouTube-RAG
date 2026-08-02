from src.yt_rag.components.data_loader import DataLoader
from src.yt_rag.components.embedding import EmbeddingManager
from src.yt_rag.components.vectorstore import FaissVectorStore

url = "https://www.youtube.com/watch?v=sD468LfeVdc&t=39s"
loader = DataLoader(url = url)
# video_data = loader.fetch_video_data()
# chunks = loader.create_sementic_chunks(video_data)

chunks = loader.load_chunks(file_path="data/chunks.pkl")

embedding_manager = EmbeddingManager(model_name = "text-embedding-3-large")
embeddings = embedding_manager.generate_embeddings(chunks = chunks)

# embedding_manager.save_embeddings(embeddings=embeddings, file_path="data/embeddings.pkl")
vectorstore=  FaissVectorStore(persist_dir="faiss_store")
vectorstore.add_embeddings(embeddings = embeddings, metadata_file_path="data/chunks.pkl")

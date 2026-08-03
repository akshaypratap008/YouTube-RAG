from src.yt_rag.components.data_loader import DataLoader
from src.yt_rag.components.embedding import EmbeddingManager
from src.yt_rag.components.vectorstore import FaissVectorStore, VectorStoreManager
from src.yt_rag.components.search import RAGSearch

url = "https://www.youtube.com/watch?v=mn9z_VpGYoo&list=PLe0At5xTDM9GsH8jO1ZVmJjVRP9sI10sn"
# loader = DataLoader(url = url)
# video_data = loader.fetch_video_data()
# chunks = loader.create_sementic_chunks(video_data)

# chunks = loader.load_chunks(file_path="data/chunks.pkl")

# embedding_manager = EmbeddingManager(model_name = "text-embedding-3-large")
# embeddings = embedding_manager.generate_embeddings(chunks = chunks)

# embedding_manager.save_embeddings(embeddings=embeddings, file_path="data/embeddings.pkl")
# vectorstore=  FaissVectorStore(persist_dir="faiss_store")
# vectorstore.add_embeddings(embeddings = embeddings, metadata_file_path="data/chunks.pkl")
# vectorstore.load()

# iniating full pipeline directly through RAGSearch
# VectorStoreManager().reset()
rag = RAGSearch(url = url)

query = "What is the summary of the full video in 100-200 words?"
print("Query: ", query)
print()

context = rag.search(query= query, top_k = 5)

result = rag.generate_response(context=context, query=query)
print(result)

timestamps = rag.get_video_timestamps()
print("Timestamps: ", timestamps)




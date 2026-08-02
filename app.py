from src.yt_rag.components.data_loader import DataLoader

url = "https://www.youtube.com/watch?v=sD468LfeVdc&t=39s"
loader = DataLoader(url = url)
video_data = loader.fetch_video_data()
chunks = loader.create_sementic_chunks(video_data)


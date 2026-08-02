from src.components.data_loader import DataLoader

url = "https://www.youtube.com/watch?v=n_3XDVOVraI&t=1341s"
loader = DataLoader(url = url)
video_data = loader.fetch_video_data()
chunks = loader.create_sementic_chunks(video_data)
print(len(chunks))
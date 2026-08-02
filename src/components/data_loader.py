from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Any, Dict
import re
import sys
import spacy
from sentence_transformers import SentenceTransformer

from src.logger import logging
from src.exceptions import CustomeException

class DataLoader:
    """
    Loads video data 
    """

    def __init__(self, url:str):
        self.url = url
        self.video_id = self._extract_video_id()

    def _extract_video_id(self):
        """
        Extract video id using regex
        """
        try:
            pattern = re.compile(
                r"(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]{11})"
            )
            match = re.search(pattern, self.url)
            video_id = match.group(1)
            logging.info(f"[INFO] Extracted video id from url: {video_id}")
            return video_id
        except Exception as e:
            raise CustomeException(e, sys)

    def fetch_video_data(self) -> List[Dict[str, Any]]:
        try:
            ytt = YouTubeTranscriptApi()
            fetched_transcript = ytt.fetch(video_id=self.video_id)

            video_data = []
            for snippet in fetched_transcript:
                video_data.append({
                    "text": snippet.text,
                    "start": snippet.start,
                    "duration": snippet.duration
                })
            logging.info(f"[INFO] Video data for video id {self.video_id} fetched.")
            return self.preprocess_transcript(video_data = video_data, min_length = 10)
        except Exception as e:
            raise CustomeException(e, sys)

    def preprocess_transcript(self, video_data:List[Dict[str, Any]], min_length:int=10) -> List[Dict[str, Any]]:
        """
        Merges small segments in the transcript text and create 1 segment. Also updates the start and end time for the segment. 
        Args:
            transcript: List with dict containing segment text, start, duration
            min_length: minimum number of words to coninue merging segments
        """
        merged = []
        i = 0
        while i < len(video_data):
            current = video_data[i].copy()

            # calculate start and end time
            current_start = current['start']
            current_end = current['start'] + current['duration']

            # count words
            word_count = len(current['text'].split())

            j = i+1

            # keep merging until min length is reached.
            while word_count < min_length and j < len(video_data):
                next_seg = video_data[j]
                current['text'] += " " + next_seg['text']
                current_start = next_seg['start']
                current_end = next_seg['start'] + next_seg['duration']
                word_count = len(current['text'].split())
                j += 1

            merged.append({
                "text" : current['text'].replace("\n", " ").strip(),
                "start": current_start,
                "end": current_end
            })

            i = j
        
        logging.info(f"[INFO] Video data preprocessing completed")
        return merged    

    def create_sementic_chunks(self, video_data:List[Dict[str, Any]], similarity_threshold:float = 0.20, max_tokens:int = 500, overlap = 1):
        text = [item['text'] for item in video_data]
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for i, item in enumerate(text):
            item_tokens = len(item.split())

            if not current_chunk:
                


# if __name__ == "__main__":
#     url = "https://www.youtube.com/watch?v=n_3XDVOVraI&t=1341s"
#     loader = DataLoader(url = url)
#     video_id = loader.fetch_video_id()
#     video_data = loader.fetch_video_data(video_id = video_id)
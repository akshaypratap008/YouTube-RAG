from youtube_transcript_api import YouTubeTranscriptApi
from typing import List, Any, Dict
import re
import sys
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv
import pickle
import os
from pathlib import Path

from src.yt_rag.logger import logging
from src.yt_rag.exceptions import CustomeException
from src.yt_rag.utils import save_object, load_object

load_dotenv()

class DataLoader:
    """
    Responsible for fetching the video data using api. Preprocess the data. Create chunks using semantic chunking and save chunks on the disk
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
        """
        Extract transcript using api.
        Returns List of dicts with different segments of the transcript as text anlong with start and end time in the actual video
        """
        try:
            ytt = YouTubeTranscriptApi()
            fetched_transcript = ytt.fetch(video_id=self.video_id)
            if fetched_transcript:
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
            raise CustomeException("No transcript found", sys)

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
                current_end = round((next_seg['start'] + next_seg['duration']), 2)
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

    def create_sementic_chunks(self, video_data:List[Dict[str, Any]], similarity_threshold:float = 0.45, max_tokens:int = 300, overlap = 1) -> List[Dict[str, Any]]:
        """
        Create semantic chunks of the transcripts. Uses SentenceTransformers to generate embeddings of the segments and then use cosine similarity tk the semantic meaning to define the end of a chunk
        Args:
            video_data : List of Dicts(text, start, end)
            similarity_threshold: similarity scroe above with the segment is broken into chunks
            max_token: total number of tokens in a particular chunk. Single chunk should not exceed the max token limit
            overlap = overlaping chunks
        Returns: List of Dicts with chunk text, start and end time in the video
        """
        try:
            text = [item['text'] for item in video_data]
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = embedder.encode(text)

            chunks = []
            current_chunk = []
            current_tokens = 0
            overlap = 1

            chunk_start = None
            chunk_end = None

            for i, item in enumerate(text):
                item_tokens = len(item.split())

                if not current_chunk:
                    # when current chunk is empty. Means skip the first one as there is nothing before the first one
                    current_chunk.append(item)
                    current_tokens += item_tokens
                    chunk_start = video_data[i]['start']
                    chunk_end = video_data[i]['end']
                    continue

                # calculate similarity between current segment and segment before it. 
                sim = np.dot(embeddings[i], embeddings[i-1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1]))

                if sim < similarity_threshold and current_tokens > max_tokens:
                    # break the segment and ppend to chunks. 
                    # if the current segment and the previous segment are too disimilar or maximum token per chunk limit is reached.
                    chunks.append({
                        "text": " ".join(current_chunk),
                        "start": chunk_start,
                        "end": chunk_end
                    })

                    if overlap > 0:
                        current_chunk = current_chunk[-overlap:]
                        current_tokens = sum(len(s.split()) for s in current_chunk)

                        # calculate timestamp from overlap chunk
                        chunk_start = video_data[i-overlap]['start']
                        chunk_end = video_data[i-overlap]['end']

                    else:
                        current_chunk = []
                        current_tokens = 0
                        chunk_start = None
                        chunk_end = None
                # add current segment 
                current_chunk.append(item)
                current_tokens += item_tokens
                chunk_end = video_data[i]['end']

            if current_chunk:
                chunks.append({
                    "text" : " ".join(current_chunk),
                    "start": chunk_start,
                    "end": chunk_end
                })
            logging.info(f"[INFO] {len(chunks)} chunks created from processed transcript")
            self.save_chunks(chunks = chunks, file_path = "data/chunks.pkl")
            return chunks
        except Exception as e:
            raise CustomeException(e, sys)

    def save_chunks(self, chunks:List[Dict[str, Any]], file_path:str = "data/chunks.pkl"):
        """
        Saves the chunks on disk
        """
        save_object(obj = chunks, file_path=file_path)
        logging.info(f'[INFO] {len(chunks)} chunks saved to the disk')


    def load_chunks(self, file_path:str = "data/chunks.pkl") -> List[Dict[str, Any]]:
        """
        Loads chunks from pkl file. Chunks will be a list of dicts containing chunk text, start and end time
        """

        chunks= load_object(file_path=file_path)
        logging.info(f"[INFO] {len(chunks)} chunks loaded")
        return chunks
        
        


# if __name__ == "__main__":
#     url = "https://www.youtube.com/watch?v=n_3XDVOVraI&t=1341s"
#     loader = DataLoader(url = url)
#     video_id = loader.fetch_video_id()
#     video_data = loader.fetch_video_data(video_id = video_id)
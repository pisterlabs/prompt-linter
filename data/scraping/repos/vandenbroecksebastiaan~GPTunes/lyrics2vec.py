from typing import List
import os
import json
import umap
import numpy as np
import matplotlib.pyplot as plt
import openai
open_ai_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = open_ai_key


class LyricsEmbedder:
    def __init__(self, album_names: List[str]):
        self.album_names = album_names
        self.song_data = [] # [[album_name, song_name, lyrics]]
        
        self.model_name = "text-embedding-ada-002" 

    def get_lyrics(self):
        """Gets the lyrics from an album."""
        from lyricsgenius import Genius
        access_token = os.environ.get("GENIUS_API_KEY")
        genius = Genius(access_token, timeout=10, sleep_time=0.1, verbose=True,
                        retries=5)
        
        for album_name in self.album_names:

            album = genius.search_album(album_name)
            album_json = album.to_json()
            album_json = json.loads(album_json)
            album_tracks = album_json["tracks"]
            album_lyrics = [track["song"]["lyrics"] for track in album_tracks]
            # Delet all lines with brackets, these don't contain lyrics
            album_lyrics = ["\n".join([line for line in text.split("\n")
                            if "[" not in line and "]" not in line])
                            for text in album_lyrics]
            album_song_titles = [track["song"]["title"] for track in album_tracks]
        
            for song_title, lyrics in zip(album_song_titles, album_lyrics):
                self.song_data.append([album_name, song_title, lyrics])
                
    def get_embeddings(self):
        # Get the embeddings
        album_embeddings = np.array([self._gpt_embedding_call(i[2]) for i
                                     in self.song_data])
        # Dim reduction
        album_embeddings = (album_embeddings - album_embeddings.mean(axis=0))\
                         / album_embeddings.std(axis=0)
        red_embeddings = umap.UMAP(n_neighbors=5).fit_transform(album_embeddings)
                         
        for idx in range(len(self.song_data)):
            self.song_data[idx][2] = red_embeddings[idx]
            
    def make_visualization(self):
        # Create a color map
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
                  "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
        album_to_color = {}
        for idx, album_name in enumerate(self.album_names):
            album_to_color[album_name] = colors[idx]
            
        # Prevent parse error
        for idx in range(len(self.song_data)):
            self.song_data[idx][1] = self.song_data[idx][1].replace("$", "S")

        fig, ax = plt.subplots(figsize=(7, 7))
        for album_name, song_name, embedding in self.song_data:
            x, y = embedding
            ax.scatter(x, y, c=album_to_color[album_name])
            ax.annotate(song_name, (x, y), fontsize=8, ha="center", va="bottom",
                        c="white")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.set_title("")

        plt.savefig("static/images/tsne.png", bbox_inches="tight", dpi=300,
                    transparent=True)

    def _gpt_embedding_call(self, prompt: List[str]):
        for idx in range(10):
            try:
                response = openai.Embedding.create(
                    input=prompt,
                    model=self.model_name
                )
                return response['data'][0]['embedding']
            except openai.error.RateLimitError:
                print(f"Error in GPT-3 call: Rate limit exceeded. Trying again... {idx}")

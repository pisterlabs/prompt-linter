import openai
import json

openai.api_key = "censored"

def main():
    print("Generating new music recommendations based on converted playlist. Please wait...")
    
    # Read the JSON file containing the list of tracks from playlist
    with open("track_data.json") as file:
        playlist_data = json.load(file)

    # Create a list of tracks in the format: "Song name" by "Artist"
    playlist_tracks = [f'"{track["name"]}" by {track["artist"]}' for track in playlist_data]

    # Queries ChatGPT to give new music and artist recommendations based on playlist
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who loves music."},
            {"role": "user", "content": f"Give me some new music and artist recommendations based on this list of songs from one of my playlists then thank the user for using the program: {playlist_tracks}"},
        ]
    )

    print(completion.choices[0].message['content'])

if __name__ == "__main__":
    main()
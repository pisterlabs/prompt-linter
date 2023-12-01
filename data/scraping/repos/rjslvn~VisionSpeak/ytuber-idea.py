import os
import argparse
import datetime
import openai  # You need to install the 'openai' library
from googleapiclient.discovery import build

def set_api_keys():
    openai_api_key = input("Enter your OpenAI API key: ")
    youtube_api_key = input("Enter your YouTube API key: ")

    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["YOUTUBE_API_KEY"] = youtube_api_key

def create_folder_structure(base_folder, year, month):
    # Create the base folder if it doesn't exist
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # Create subfolders for the specified year and month
    year_folder = os.path.join(base_folder, str(year))
    month_folder = os.path.join(year_folder, str(month))

    # Create the year and month folders if they don't exist
    if not os.path.exists(year_folder):
        os.makedirs(year_folder)
    if not os.path.exists(month_folder):
        os.makedirs(month_folder)

    return month_folder

def generate_script(video_title):
    # Use GPT-3 to generate a script based on the video title or description
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Create a script for a video titled '{video_title}':",
        max_tokens=150  # Adjust the max tokens as needed for the script length
    )
    return response.choices[0].text

def get_trending_videos(api_key, region_code, max_results=10):
    youtube = build("youtube", "v3", developerKey=api_key)

    # Get trending videos in the specified region
    request = youtube.videos().list(
        part="snippet",
        chart="mostPopular",
        regionCode=region_code,
        maxResults=max_results
    )

    response = request.execute()
    return response.get("items", [])

def main():
    parser = argparse.ArgumentParser(description="Create folder structures, scripts, and retrieve trending YouTube content.")
    parser.add_argument("--base-folder", default="YouTubeContent", help="Specify the base folder for folder structures.")
    parser.add_argument("--iterations", type=int, default=3, help="Specify the number of iterations.")
    parser.add_argument("--region-code", default="US", help="Specify the region code for trending videos (e.g., US).")
    args = parser.parse_args()

    set_api_keys()  # Prompt for and set API keys as environment variables

    # Get the current date to determine the year and month
    now = datetime.datetime.now()

    for i in range(args.iterations):
        year = now.year
        month = now.strftime("%B")

        # Create the folder structure
        content_folder = create_folder_structure(args.base_folder, year, month)

        # Retrieve trending videos
        trending_videos = get_trending_videos(os.environ["YOUTUBE_API_KEY"], args.region_code)

        # Generate scripts for trending videos
        for index, video in enumerate(trending_videos):
            video_title = video["snippet"]["title"]
            script = generate_script(video_title)

            # Create a script file and write the generated script
            script_filename = os.path.join(content_folder, f"Trending_{index + 1}_script.txt")
            with open(script_filename, "w") as file:
                file.write(script)

            print(f"Created folder structure for {month} {year} at: {content_folder}")
            print(f"Generated script for '{video_title}' and saved it to: {script_filename}")

        # Move to the previous month for the next iteration
        now = now - datetime.timedelta(days=30)

if __name__ == "__main__":
    main()

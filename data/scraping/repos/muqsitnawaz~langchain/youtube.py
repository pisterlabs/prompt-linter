"""
YouTube Package provides a way to download transcripts of YouTube videos.
"""
import json
import logging

from scrapetube import get_channel
from langchain.document_loaders import YoutubeLoader
from alive_progress import alive_bar


def get_all_videos_links(username: str) -> list[str]:
    videos = get_channel(channel_username=username)

    videos_ids = []
    for video in videos:
        video.get('')
        videos_ids.append(video['videoId'])

    return videos_ids


def get_all_transcripts(video_ids: list[str]) -> list[(str, str)]:
    """
    Returns a list of tuples (title, transcript)
    """
    global link
    for video_id in video_ids:
        try:
            link = 'https://www.youtube.com/watch?v={}'.format(video_id)
            loader = YoutubeLoader.from_youtube_url(link, add_video_info=True)

            doc = loader.load()
            data = doc[0].dict()
            title = data['metadata']['title']
            transcript = json.dumps(data, indent=4)

            yield title, transcript
        except Exception as e:
            logging.error('failed to process {}'.format(link, e))
        finally:
            continue


if __name__ == '__main__':
    username = 'TheDiaryOfACEO'

    print('getting all videos for {}'.format(username))
    video_ids = get_all_videos_links(username)
    print('found {} videos'.format(len(video_ids)))

    count = 1
    for (title, transcript) in get_all_transcripts(video_ids):
        print('{}/{} processing {}'.format(count, len(video_ids), title))
        count += 1

        # Check if title is valid
        if title is None or len(title) == 0:
            logging.warning('skipping {}'.format(title))
            continue

        # Write transcript to file
        try:
            path = '../data/videos/youtube/@{}/{}.json'.format(username, title)
            with open(path, 'w') as f:
                f.write(transcript)
                f.close()
        except Exception as e:
            logging.error('failed to write {}, err: {}'.format(title, e))
        finally:
            continue

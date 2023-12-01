"""Demo the component when it's not part of a Lightning App."""
from tqdm import tqdm

from lit_video_stream import LitVideoStream
from lit_video_stream.feature_extractors import OpenAIClip
from lit_video_stream.stream_processors import YouTubeStreamProcessor


class PBar:
    def __init__(self) -> None:
        self._prog_bar = None

    def update(self, current_frame):
        self._prog_bar.update(1)

    def reset(self, total_frames):
        if self._prog_bar is not None:
            self._prog_bar.close()
        self._prog_bar = tqdm(total=total_frames)


if __name__ == "__main__":

    lit_video_stream = LitVideoStream(
        feature_extractor=OpenAIClip(batch_size=256),
        stream_processor=YouTubeStreamProcessor(),
        prog_bar=PBar(),
        process_every_n_frame=100,
        num_batch_frames=256,
        length_limit=None,
    )

    one_hour = "https://www.youtube.com/watch?v=rru2passumI"
    one_min = "https://www.youtube.com/watch?v=8SQL4knuDXU"
    lit_video_stream.download(video_urls=[one_min, one_min])

import argparse
import json_tricks
import logging
from typing import List
from pathlib import Path
from auto_shot_list.openai_frame_analyser import OpenAIFrameAnalyser
from auto_shot_list.scene_manager import VideoManager


def get_video_files(directory: Path) -> List[Path]:
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv']
    video_files = []

    for file in directory.glob('**/*'):
        if file.suffix.lower() in video_extensions:
            video_files.append(file)

    return video_files


def analyse_files(videos_dir: Path, output_dir: Path, subtitles_dir: Path = None, subtitles_rules: Path = None):
    """
    Analyses all files in given folder
    :param subtitles_dir:
    :param subtitles_rules:
    :param videos_dir: path to a directory with video files
    :param output_dir: path to a directory to store created shot lists
    :return:
    """

    analyser = OpenAIFrameAnalyser()

    if not videos_dir.exists():
        raise ValueError(f"Videos directory does not exists: {videos_dir}")

    if not output_dir.exists():
        raise ValueError(f"Output directory does not exists: {videos_dir}")

    if subtitles_dir and not subtitles_dir.exists():
        raise ValueError(f"Subtitles directory does not exists: {subtitles_dir}")

    if subtitles_rules and not subtitles_rules.exists():
        raise ValueError(f"Subtitles rules file does not exists: {subtitles_rules}")

    video_paths = get_video_files(videos_dir)

    if len(video_paths) == 0:
        raise ValueError(f"No videos found in directory {videos_dir}")
    logging.info(f"Found {len(video_paths)} files. Starting shot list extraction")

    for video_path in video_paths:
        logging.info(f"Processing {video_path}")

        subtitles_path = None
        if subtitles_dir:
            subtitles_path = subtitles_dir / (video_path.stem + ".ass")
            if not subtitles_path.exists():
                raise ValueError("Subtitle path not found")
        video_manager = VideoManager(
            str(video_path), frame_analyser=analyser, subtitle_path=subtitles_path, subtitle_filter_path=subtitles_rules
        )
        video_manager.detect_scenes()
        video_manager.analyse_scenes()
        video_manager.frame_analyser = None
        video_manager.subtitle_filter = None
        with open(output_dir / (video_path.stem + ".json"), "w") as f:
            json_tricks.dump(video_manager, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", '--videos_dir', type=str, help='path to the directory containing the videos')
    parser.add_argument("-o", '--output_dir', type=str, help='path where the output json files will be saved')
    parser.add_argument("-s", '--subtitles_dir', type=str, help='path to the directory containing the subtitles',
                        default=None)
    parser.add_argument("-sr", '--subtitles_rules', type=str, help='path to the rules file for the subtitles',
                        default=None)

    args = parser.parse_args()

    analyse_files(
        Path(args.videos_dir),
        Path(args.output_dir),
        subtitles_dir=Path(args.subtitles_dir),
        subtitles_rules=Path(args.subtitles_rules)
    )

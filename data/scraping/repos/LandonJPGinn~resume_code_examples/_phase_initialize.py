# Start a new episode - from an Episode listed already in the Database.
# This is not an episode idea command

import argparse
import sys
from pathlib import Path
import json
import re
from CreatorPipeline import _openai as oai
from CreatorPipeline.constants import DEFAULTS, PROMPTS, STATUS
from CreatorPipeline.directory import create_episode_directory
from CreatorPipeline.episode import Episode
from CreatorPipeline.database import ActiveEpisodeDatabase


class PhaseBuild:
    """Build out dependencies for a phase step. Currently simple, may need to be more complex later"""
    def __init__(self, episode):
        self.episode = [episode]
        initialize_episodes(self.episode)


def initialize_episodes(episodes):
    """Initialize a episodes from the database into folders"""
    selected_episodes = []

    for episode in episodes:
        # episode_id = episode.get("ID")
        if not isinstance(episode, Episode):
            current_episode = Episode(episode)
        else:
            current_episode = episode
        if not current_episode:
            print(
                f"{episode} failed to find an Episode. Make sure the hash provided is in the database.\nStopping"
            )
            sys.exit()
        selected_episodes.append(current_episode)

    # confirmation block
    print("You have selected the following: ")
    [ep.summarize() for ep in selected_episodes]

    print("\n\nContinue?")
    resp = input("Y/n")
    if resp.lower() != "y":
        print("cancelling...")
        sys.exit()

    for episode in selected_episodes:
        # ensure that the episode row contains values for:
        """
        Keyword Phrase
        Tactic
        Gain
        Avatar
        Method
        Format
        Playlist
        All sheet columns G:N
        """
        exceptions = ("VideoID", "Thumb_Text", "next_video_id","Status")
        for k, v in episode.__dict__.items():
            print(f"{k}: {v}")
            if k in exceptions:
                continue
            assert v, f"No Value found for {k}. Fill out Episode first in Sheet."

        directory = create_episode_directory(episode)
        print(directory)
        assert directory, f"{directory} is not a valid directory"
        openai_initial(episode_root=directory, episode=episode)

        episode.change_status(STATUS.start)
        episode.queue_up()
        ActiveEpisodeDatabase().add_episode(episode.ID)

def clean_up_json(proposal_file=None):
    """Cleans up the JSON file from OpenAI to be valid JSON object for system"""
    try:
        with open(proposal_file, "r") as f:
            data = f.readlines()
        data = "".join(data)
        data = re.sub(r'\s+', ' ', data)
        pattern = r"\{(?:[^{}]|(.*))\}"
        find_json = re.findall(pattern, data)
        for json_obj in find_json:
            try:
                jdata = json.loads("{" + f"{json_obj}" + "}")
                with open(proposal_file, "w") as f:
                    json.dump(jdata, f, indent=4)
            except json.decoder.JSONDecodeError:
                print(f"Invalid JSON Manually clean file: {proposal_file}")

    except IOError as err:
        print(err)
        print("Error Occured while cleaning up JSON. Please check file manually.")


def openai_initial(episode_root=".", episode=None):
    """Generates the initial text ideas for an episode using OpenAI."""
    episode_root = Path(episode_root).expanduser().resolve()
    params = episode.__dict__

    prompts = [
        [
            PROMPTS.proposed_episode.substitute(**params),
            DEFAULTS.define_proposals,
        ],
        [
            PROMPTS.proposed_description.substitute(**params),
            DEFAULTS.define_description,
        ],
        [
            PROMPTS.qotd.substitute(**params),
            DEFAULTS.define_qotd,
        ],
        [
            PROMPTS.thumbnail_prompts.substitute(**params),
            DEFAULTS.define_thumbnail_prompts,
        ],
    ]

    for prompt, filepath in prompts:
        oai.generate_text(prompt, episode_root / filepath)

    clean_up_json(episode_root / DEFAULTS.define_proposals)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Initialize Episode Directory and Queue up"
    )
    parser.add_argument(
        "episodes", type=str, nargs="+", help="push episode(s) to initialize"
    )
    args = parser.parse_args()
    initialize_episodes(args.episodes)

"""
When I run Initialize Episode
I expect to provide it a hash value and look for the closest match, asking to confirm with a display showing details about it.

If I confirm, a directory will be generated with the correct name from a template
The meta data that was shown ( database values )
Get added to a prompt for openai to run a few generators for episode content.
The Define step is over and the status switches to research.

"""

### Imports
import datetime
import difflib
import json
import re
import requests
import sys
import time
from typing import Dict, List, NamedTuple, Optional, Union

print('imported successfully')

#@markdown Default seed for generation (default: 1)
DEFAULT_SEED =  1  #@param {type:"integer"}
#@markdown Sampling top-p probability (default: 0.9) and temperature (default 1.0)
SAMPLING_PROB =  0.9  #@param {type:"slider", min:0.8, max:1.0, step:0.01}
SAMPLING_TEMP =  1.  #@param {type:"slider", min:0.8, max:1.0, step:0.01}
#@markdown Max length for the generated title, place description and others, in tokens (defaults: 64, 128 and 511 respectively)
SAMPLE_LENGTH_TITLE = 64 #@param [64, 128, 256, 511]
SAMPLE_LENGTH_PLACE = 128 #@param [128, 256, 511]
SAMPLE_LENGTH = 511 #@param [128, 256, 511]
#@markdown Max lengths during repeated sampling, in case `<end>` is not found (default: 2048)
MAX_PARAGRAPH_LENGTH_CHARACTERS = 1024 #@param [511, 1024, 2048, 4096]
MAX_PARAGRAPH_LENGTH_SCENES = 1024 #@param [511, 1024, 2048, 4096]
MAX_PARAGRAPH_LENGTH = 1024 #@param [511, 1024, 2048, 4096]
#@markdown Unavailable API: max number of retries before giving up (default: 10)
MAX_RETRIES = 10 #@param {type:"slider", min:1, max:20, step:1}
#@markdown Loop detection: max number of repetitions before resampling, and number of attempts to get out of the loop (default: 3)
MAX_NUM_REPETITIONS = 3 #@param {type:"slider", min:1, max:10, step:1}
MAX_NUM_ATTEMPTS_GET_OUT_OF_LOOP = 3 #@param {type:"slider", min:1, max:10, step:1}
#@markdown (GPT-3 engine only) Name of the GPT-3 engine used for the Language API (default: `text-davinci-002`)
GPT3_ENGINE = "text-davinci-003" #@param {type: "string"}

print('Dramatron hyperparameters set.')

# ------------------------------------------------------------------------------
# Script markers
# ------------------------------------------------------------------------------

END_MARKER = '<end>'
STOP_MARKER = '<stop>'
CHARACTER_MARKER = '<character>'
DESCRIPTION_MARKER = '<description>'
SCENES_MARKER = '<scenes>'
DIALOG_MARKER = '<dialog>'
TITLE_ELEMENT = 'Title: '
CHARACTERS_ELEMENT = 'Characters: '
DESCRIPTION_ELEMENT = 'Description: '
PLACE_ELEMENT = 'Place: '
PLOT_ELEMENT = 'Plot element: '
PREVIOUS_ELEMENT = 'Previous beat: '
SUMMARY_ELEMENT = 'Summary: '
BEAT_ELEMENT = 'Beat: '

# ------------------------------------------------------------------------------
# Dramatron script entities
# ------------------------------------------------------------------------------


class Title(NamedTuple):
  """Title class."""

  title: str

  @classmethod
  def from_string(cls, text: str):
    title = extract_elements(text, TITLE_ELEMENT, END_MARKER)[0]
    return cls(title)

  def to_string(self):
    s = ''
    s += TITLE_ELEMENT + self.title
    s += END_MARKER
    return s


def get_title(title: Title) -> str:
  return title.title


class Character(NamedTuple):
  """Character class."""

  # Name of the character.
  name: str

  # A single sentence describing the character.
  description: str

  @classmethod
  def from_string(cls, text: str):
    elements = text.split(DESCRIPTION_MARKER)
    if len(elements) == 2:
      name = elements[0].strip()
      description = elements[1].strip()
      return cls(name, description)
    else:
      return None


def get_character_description(character: Character) -> str:
  return character.description


class Characters(NamedTuple):
  """Characters class, containing main characters and their descriptions."""

  # A dictionary of character descriptions.
  character_descriptions: Dict[str, str]

  @classmethod
  def from_string(cls, text: str):
    """Parses the characters from the generated text."""
    text = text.strip()

    # Extracts the character descriptions.
    character_descriptions = {}
    elements = extract_elements(text, CHARACTER_MARKER, STOP_MARKER)
    for text_character in elements:
      character = Character.from_string(text_character)
      if character is not None:
        character_descriptions[character.name] = character.description
    return cls(character_descriptions)

  def to_string(self):
    s = '\n'
    for name, description in self.character_descriptions.items():
      s += '\n' + CHARACTER_MARKER + ' ' + name + ' ' + DESCRIPTION_MARKER + ' '
      s += description + ' ' + STOP_MARKER + '\n'
    s += END_MARKER
    return s


def get_character_descriptions(characters: Characters) -> Dict[str, str]:
  return characters.character_descriptions


class Scene(NamedTuple):
  """Scene class."""

  # The name of the place where the scene unfolds.
  place: str

  # Name of the plot element (e.g., Beginning, Middle, Conclusion).
  plot_element: str

  # A short description of action/story/dramatic event occuring in the scene.
  beat: str

  def to_string(self):
    s = PLACE_ELEMENT + ' ' + self.place + '\n'
    s += PLOT_ELEMENT + ' ' + self.plot_element + '\n'
    s += BEAT_ELEMENT + ' ' + self.beat + '\n'
    return s


class Scenes(NamedTuple):
  """Scenes class."""

  # A list of scenes, with place, characters, plot element and beat.
  scenes: List[Scene]

  @classmethod
  def from_string(cls, text: str):
    """Parse scenes from generated scenes_text."""

    places = extract_elements(text, PLACE_ELEMENT, PLOT_ELEMENT)
    plot_elements = extract_elements(text, PLOT_ELEMENT, BEAT_ELEMENT)
    beats = extract_elements(text, BEAT_ELEMENT, '\n')

    # Get the number of complete scenes.
    num_complete_scenes = min([len(places), len(plot_elements), len(beats)])
    scenes = []
    for i in range(num_complete_scenes):
      scenes.append(
          Scene(Place.format_name(places[i]), plot_elements[i], beats[i]))
    scenes = cls(scenes)
    return scenes

  def to_string(self):
    s = ''
    for scene in self.scenes:
      s += '\n' + scene.to_string()
    s += END_MARKER
    return s

  def num_places(self):
    return len(set([scene.place for scene in self.scenes]))

  def num_scenes(self) -> int:
    return len(self.scenes)


class Place(NamedTuple):
  """Place class."""

  # Place name.
  name: str

  # Place description.
  description: str

  @classmethod
  def format_name(cls, name: str):
    if name.find('.') == -1:
      name = name + '.'
    return name

  @classmethod
  def from_string(cls, place_name: str, place_text: str):
    place_text += END_MARKER
    description = extract_elements(place_text, DESCRIPTION_ELEMENT, END_MARKER)
    return cls(place_name, description[0])

  @classmethod
  def format_prefix(cls, name):
    s = PLACE_ELEMENT + name + '\n' + DESCRIPTION_ELEMENT
    return s

  def to_string(self):
    s = self.format_prefix(self.name) + self.description + '\n\n'
    return s


def get_place_description(place: Place):
  return place.description


class Story(NamedTuple):
  """Story class."""

  # A storyline is a single sentence summary of the whole plot.
  storyline: str

  # A title for the story.
  title: str

  # Map from character names to full descriptions.
  character_descriptions: Dict[str, str]

  # Map from place names to full descriptions.
  place_descriptions: Dict[str, Place]

  # List of scenes.
  scenes: Scenes

  # List of dialogs, one for each scene.
  dialogs: List[str]


def extract_elements(text: str, begin: str, end: str) -> List[str]:
  """Extracts elements from a text string given string and ending markers."""

  results = []
  start = 0
  while True:
    start = text.find(begin, start)
    if start == -1:
      return results
    finish = text.find(end, start)
    if finish == -1:
      return results
    results.append(text[start + len(begin):finish].strip())
    start = finish + len(end)


def strip_remove_end(text: str) -> str:
  text = text.strip()
  end_marker_stripped = END_MARKER.strip()
  if text.endswith(end_marker_stripped):
    text = text[:-len(end_marker_stripped)]
  return text


# ------------------------------------------------------------------------------
# Rendering of generated stories
# ------------------------------------------------------------------------------


def render_story(story: Story) -> str:
  """Render the story in fountain format."""

  lines = []
  lines.append(f'Title: {story.title}')
  lines.append('Author: Co-written by ________ and Dramatron')
  lines.append(
      'Dramatron was developed by Piotr Mirowski and Kory W. Mathewson, '
      'with additional contributions by Juliette Love and Jaylen Pittman, '
      'and is based on a prototype by Richard Evans.')
  lines.append('Dramatron relies on user-provided language models.')
  lines.append('')
  lines.append('====')
  lines.append('')

  lines.append(f'The script is based on the storyline:\n{story.storyline}')
  lines.append('')
  if story.character_descriptions is not None:
    for name, description in story.character_descriptions.items():
      lines.append(f'{name}: {description}')
      lines.append('')

  # For each scene, render scene information.
  if story.scenes is not None:
    scenes = story.scenes.scenes
    for i, scene in enumerate(scenes):
      lines.append(f'Scene {i+1}')
      lines.append(f'{PLACE_ELEMENT}{scene.place}')
      lines.append(f'{PLOT_ELEMENT}{scene.plot_element}')
      lines.append(f'{BEAT_ELEMENT}{scene.beat}')
      lines.append('')
  else:
    scenes = []

  lines.append('====')
  lines.append('')

  # For each scene, render the scene's place description, characters and dialog.
  for i, scene in enumerate(scenes):

    # Output the places and place descriptions.
    lines.append(f'INT/EXT. {scene.place} - Scene {i+1}')
    place_descriptions = story.place_descriptions
    if (not place_appears_earlier(scene.place, story, i) and
        place_descriptions is not None and scene.place in place_descriptions):
      lines.append('')
      lines.append(get_place_description(place_descriptions[scene.place]))

    # Output the characters and descriptions.
    lines.append('')
    for c in story.character_descriptions.keys():
      if c in scene.beat and not character_appears_earlier(c, story, i):
        lines.append(story.character_descriptions[c])

    # Output the dialog.
    if story.dialogs is not None and len(story.dialogs) > i:
      lines.append('')
      lines_dialog = strip_remove_end(str(story.dialogs[i]))
      lines.append(lines_dialog)
      lines.append('')
      lines.append('')

  return '\n'.join(lines)


def place_appears_earlier(place: str, story: Story, index: int) -> bool:
  """Return True if the place appears earlier in the story."""

  for i in range(index):
    scene = story.scenes.scenes[i]
    if scene.place == place:
      return True
  return False


def character_appears_earlier(character: str, story: Story, index: int) -> bool:
  """Return True if the character appears earlier in the story."""

  for i in range(index):
    scene = story.scenes.scenes[i]
    if character in scene.beat:
      return True
  return False


def render_prompts(prompts):
  """Render the prompts."""

  def _format_prompt(prompt, name):
    prompt_str = '=' * 80 + '\n'
    prompt_str += 'PROMPT (' + name + ')\n'
    prompt_str += '=' * 80 + '\n\n'
    prompt_str += str(prompt) + '\n\n'
    return prompt_str

  prompts_str = _format_prompt(prompts['title'], 'title')
  prompts_str += _format_prompt(prompts['characters'], 'characters')
  prompts_str += _format_prompt(prompts['scenes'], 'scenes')
  places = prompts['places']
  if places is not None:
    for k, prompt in enumerate(places):
      prompts_str += _format_prompt(prompt, 'place ' + str(k + 1))
  dialogs = prompts['dialogs']
  if dialogs is not None:
    for k, prompt in enumerate(dialogs):
      prompts_str += _format_prompt(prompt, 'dialog ' + str(k + 1))
  return prompts_str


# ------------------------------------------------------------------------------
# Language API definition
# ------------------------------------------------------------------------------

_MAX_RETRIES = 10
_TIMEOUT = 120.0


class LanguageResponse(NamedTuple):
  prompt: str
  prompt_length: int
  text: str
  text_length: int


class LanguageAPI:
  """Language model wrapper."""

  def __init__(self,
               sample_length: int,
               model: Optional[str] = None,
               model_param: Optional[str] = None,
               config_sampling: Optional[dict] = None,
               seed: Optional[int] = None,
               max_retries: int = _MAX_RETRIES,
               timeout: float = _TIMEOUT):
    """Initializer.

    Args:
      sample_length: Length of text to sample from model.
      model: The model name to correct to. An error will be raised if it does
        not exist.
      model_param: Model parameter.
      config_sampling: Sampleing parameters.
      seed: Random seed for sampling.
      max_retries: Maximum number of retries for the remote API.
      timeout: Maximum waiting timeout
    """
    self._sample_length = sample_length
    self._model = model
    self._model_param = model_param
    self._config_sampling = config_sampling
    self._seed = seed
    self._max_retries = max_retries
    self._timeout = timeout

  @property
  def default_sample_length(self):
    return self._sample_length

  @property
  def model(self):
    return self._model

  @property
  def model_param(self):
    return self._model_param

  @property
  def model_metadata(self):
    return None

  @property
  def seed(self):
    return self._seed

  @property
  def config_sampling(self):
    return self._config_sampling

  def sample(self,
             prompt: str,
             sample_length: Optional[int] = None,
             seed: Optional[int] = None,
             num_samples: int = 1):
    """Sample model with provided prompt, optional sample_length and seed."""
    raise NotImplementedError('sample method not implemented in generic class')


class FilterAPI:
  """Filter model wrapper."""

  def validate(self, text: str):
    raise NotImplementedError('validate not implemented in generic class')


# ------------------------------------------------------------------------------
# Dramatron Generator
# ------------------------------------------------------------------------------


def generate_text(generation_prompt: str,
                  client: LanguageAPI,
                  filter: Optional[FilterAPI] = None,
                  sample_length: Optional[int] = None,
                  max_paragraph_length: int = MAX_PARAGRAPH_LENGTH,
                  seed: Optional[int] = None,
                  num_samples: int = 1,
                  max_num_repetitions: Optional[int] = None) -> str:
  """Generate text using the generation prompt."""

  # To prevent lengthy generation loops, we cap the number of calls to the API.
  if sample_length is None:
    sample_length = client.default_sample_length
  max_num_calls = int(max_paragraph_length / sample_length) + 1
  num_calls = 0

  result = ''
  while True:
    prompt = generation_prompt + result
    success, current_seed = False, seed
    while success is False:
      t0 = time.time()
      responses = client.sample(
          prompt=prompt,
          sample_length=sample_length,
          seed=current_seed,
          num_samples=num_samples)
      t1 = time.time()
      # Get the first result from the list of responses
      response = responses[0]
      if filter is not None and not filter.validate(response.text):
        return 'Content was filtered out.' + END_MARKER
      if max_num_repetitions:
        success = not detect_loop(
            response.text, max_num_repetitions=max_num_repetitions)
        if not success:
          current_seed += 1
          if current_seed > (seed + MAX_NUM_ATTEMPTS_GET_OUT_OF_LOOP):
            success = True
          else:
            continue
      else:
        success = True

    result = result + response.text
    num_calls += 1

    # Attempt to find the END_MARKER
    index = result.find(END_MARKER)
    if index != -1:
      return result[:index] + END_MARKER

    # Attempt to find the start of a new example
    index = result.find('Example ')
    if index != -1:
      return result[:index] + END_MARKER

    if max_paragraph_length is not None and len(result) > max_paragraph_length:
      return result + END_MARKER
    if num_calls >= max_num_calls:
      return result + END_MARKER

  return result


def generate_text_no_loop(generation_prompt: str,
                          client: LanguageAPI,
                          filter: Optional[FilterAPI] = None,
                          sample_length: Optional[int] = None,
                          max_paragraph_length: int = MAX_PARAGRAPH_LENGTH,
                          seed: Optional[int] = None,
                          num_samples: int = 1) -> str:
  """Generate text using the generation prompt, without any loop."""
  return generate_text(
      generation_prompt=generation_prompt,
      client=client,
      filter=filter,
      sample_length=sample_length,
      max_paragraph_length=sample_length,
      seed=seed,
      max_num_repetitions=None,
      num_samples=num_samples)


def generate_title(storyline: str,
                   prefixes: Dict[str, str],
                   client: LanguageAPI,
                   filter: Optional[FilterAPI] = None,
                   seed: Optional[int] = None,
                   num_samples: int = 1):
  """Generate a title given a storyline, and client."""

  # Combine the prompt and storyline as a helpful generation prefix
  titles_prefix = prefixes['TITLES_PROMPT'] + storyline + ' ' + TITLE_ELEMENT
  title_text = generate_text_no_loop(
      generation_prompt=titles_prefix,
      client=client,
      filter=filter,
      sample_length=SAMPLE_LENGTH_TITLE,
      seed=seed,
      num_samples=num_samples)
  title = Title.from_string(TITLE_ELEMENT + title_text)
  return (title, titles_prefix)


def generate_characters(
    storyline: str,
    prefixes: Dict[str, str],
    client: LanguageAPI,
    filter: Optional[FilterAPI] = None,
    seed: Optional[int] = None,
    max_paragraph_length: int = (MAX_PARAGRAPH_LENGTH_CHARACTERS),
    num_samples: int = 1):
  """Generate characters given a storyline, prompt, and client."""

  # Combine the prompt and storyline as a helpful generation prefix
  characters_prefix = prefixes['CHARACTERS_PROMPT'] + storyline
  characters_text = generate_text(
      generation_prompt=characters_prefix,
      client=client,
      filter=filter,
      seed=seed,
      max_paragraph_length=max_paragraph_length,
      num_samples=num_samples)
  characters = Characters.from_string(characters_text)

  return (characters, characters_prefix)


def generate_scenes(storyline: str,
                    character_descriptions: Dict[str, str],
                    prefixes: Dict[str, str],
                    client: LanguageAPI,
                    filter: Optional[FilterAPI] = None,
                    seed: Optional[int] = None,
                    max_paragraph_length: int = (MAX_PARAGRAPH_LENGTH_SCENES),
                    num_samples: int = 1):
  """Generate scenes given storyline, prompt, main characters, and client."""

  scenes_prefix = prefixes['SCENE_PROMPT'] + storyline + '\n'
  for name in character_descriptions:
    scenes_prefix += character_descriptions[name] + '\n'
  scenes_prefix += '\n' + SCENES_MARKER
  scenes_text = generate_text(
      generation_prompt=scenes_prefix,
      client=client,
      filter=filter,
      seed=seed,
      max_paragraph_length=max_paragraph_length,
      num_samples=num_samples)
  scenes = Scenes.from_string(scenes_text)

  return (scenes, scenes_prefix)


def generate_place_descriptions(storyline: str,
                                scenes: Scenes,
                                prefixes: Dict[str, str],
                                client: LanguageAPI,
                                filter: Optional[FilterAPI] = None,
                                seed: Optional[int] = None,
                                num_samples: int = 1):
  """Generate a place description given a scene object and a client."""

  place_descriptions = {}

  # Get unique place names from the scenes.
  unique_place_names = set([scene.place for scene in scenes.scenes])

  # Build a unique place prefix prompt.
  place_prefix = prefixes['SETTING_PROMPT'] + storyline + '\n'

  # Build a list of place descriptions for each place
  place_prefixes = []
  for place_name in unique_place_names:
    place_suffix = Place.format_prefix(place_name)
    place_text = generate_text(
        generation_prompt=place_prefix + place_suffix,
        client=client,
        filter=filter,
        sample_length=SAMPLE_LENGTH_PLACE,
        seed=seed,
        num_samples=num_samples)
    place_text = place_suffix + place_text
    place_descriptions[place_name] = Place.from_string(place_name, place_text)
    place_prefixes.append(place_prefix + place_suffix)

  return (place_descriptions, place_prefixes)


def prefix_summary(storyline: str,
                   scenes: List[Scene],
                   concatenate_scenes_in_summary: bool = False) -> str:
  """Assemble the summary part of the dialog prefix."""

  summary = SUMMARY_ELEMENT + storyline + '\n'
  if len(scenes) > 1:
    summary += PREVIOUS_ELEMENT + scenes[len(scenes) - 2].beat + '\n'
  return summary


def detect_loop(text: str, max_num_repetitions: int = MAX_NUM_REPETITIONS):
  """Detect loops in generated text."""

  blocks = text.split('\n\n')
  num_unique_blocks = collections.Counter(blocks)
  for block in blocks:
    num_repetitions = num_unique_blocks[block]
    if num_repetitions > max_num_repetitions:
      print(f'Detected {num_repetitions} repetitions of block:\n{block}')
      return True
  return False


def generate_dialog(storyline: str,
                    scenes: List[Scene],
                    character_descriptions: Dict[str, str],
                    place_descriptions: Dict[str, Place],
                    prefixes: Dict[str, str],
                    max_paragraph_length: int,
                    client: LanguageAPI,
                    filter: Optional[FilterAPI] = None,
                    max_num_repetitions: Optional[int] = None,
                    seed: Optional[int] = None,
                    num_samples: int = 1):
  """Generate dialog given a scene object and a client."""

  scene = scenes[-1]

  place_t = PLACE_ELEMENT + scene.place + '\n'
  if scene.place in place_descriptions:
    place_description = place_descriptions[scene.place]
    if place_description:
      place_t += DESCRIPTION_ELEMENT + place_description.description
      place_t += '\n'

  # Build the characters information for the scene
  characters_t = ''
  if character_descriptions:
    characters_t += CHARACTERS_ELEMENT
    for name in character_descriptions:
      if name in scene.beat:
        characters_t += character_descriptions[name] + '\n'

  plot_element_t = PLOT_ELEMENT + scene.plot_element + '\n'

  summary_t = prefix_summary(
      storyline, scenes, concatenate_scenes_in_summary=False)

  beat_t = BEAT_ELEMENT + scene.beat + '\n'

  dialog_prefix = (
      prefixes['DIALOG_PROMPT'] + place_t + characters_t + plot_element_t +
      summary_t + beat_t)
  dialog_prefix += '\n' + DIALOG_MARKER + '\n'

  dialog = generate_text(
      generation_prompt=dialog_prefix,
      client=client,
      filter=filter,
      seed=seed,
      max_paragraph_length=max_paragraph_length,
      max_num_repetitions=max_num_repetitions,
      num_samples=num_samples)

  return (dialog, dialog_prefix)


def diff_prompt_change_str(prompt_before: str, prompt_after: str) -> str:
  """Return a text diff on prompt sets `prompt_before` and `prompt_after`."""

  # For the current element, compare prompts line by line.
  res = difflib.unified_diff(
      prompt_before.split('\n'), prompt_after.split('\n'))
  diff = ''
  for line in res:
    line = line.strip()
    if line != '---' and line != '+++' and not line.startswith('@@'):
      if len(line) > 1 and (line.startswith('+') or line.startswith('-')):
        diff += line + '\n'
  if diff.endswith('\n'):
    diff = diff[:-1]
  return diff


def diff_prompt_change_list(prompt_before: List[str],
                            prompt_after: List[str]) -> str:
  """Return a text diff on prompt sets `prompt_before` and `prompt_after`."""

  # Handle deletions and insertions.
  len_before = len(prompt_before)
  len_after = len(prompt_after)
  if len_before > len_after:
    return 'Deleted element'
  if len_before < len_after:
    return 'Added new element'

  diffs = [
      diff_prompt_change_str(a, b)
      for (a, b) in zip(prompt_before, prompt_after)
  ]
  return '\n'.join([diff for diff in diffs if len(diff) > 0])


def diff_prompt_change_scenes(prompt_before: List[Scene],
                              prompt_after: List[Scene]) -> str:
  """Return a text diff on prompt sets `prompt_before` and `prompt_after`."""

  # Handle deletions and insertions.
  len_before = len(prompt_before)
  len_after = len(prompt_after)
  if len_before > len_after:
    return 'Deleted element'
  if len_before < len_after:
    return 'Added new element'

  diffs = [
      diff_prompt_change_list([a.place, a.plot_element, a.beat],
                              [b.place, b.plot_element, b.beat])
      for (a, b) in zip(prompt_before, prompt_after)
  ]
  return '\n'.join([diff for diff in diffs if len(diff) > 0])


def diff_prompt_change_dict(prompt_before: Dict[str, str],
                            prompt_after: Dict[str, str]) -> str:
  """Return a text diff on prompt sets `prompt_before` and `prompt_after`."""

  # Loop over the keys in the prompts to compare them one by one.
  keys_before = sorted(prompt_before.keys())
  keys_after = sorted(prompt_after.keys())
  diffs = [
      diff_prompt_change_str(a, b) for (a, b) in zip(keys_before, keys_after)
  ]
  diff_keys = '\n'.join([diff for diff in diffs if len(diff) > 0])
  # Loop over the values in the prompts to compare them one by one.
  values_before = sorted(prompt_before.values())
  values_after = sorted(prompt_after.values())
  diffs = [
      diff_prompt_change_str(a, b)
      for (a, b) in zip(values_before, values_after)
  ]
  diff_values = '\n'.join([diff for diff in diffs if len(diff) > 0])
  return diff_keys + diff_values


class StoryGenerator:
  """Generate a story from the provided storyline, using the client provided."""

  level_names = ('storyline', 'title', 'characters', 'scenes', 'places',
                 'dialogs')

  def __init__(
      self,
      storyline: str,
      prefixes: Dict[str, str],
      max_paragraph_length: int = 1024,
      max_paragraph_length_characters: int = (MAX_PARAGRAPH_LENGTH_CHARACTERS),
      max_paragraph_length_scenes: int = (MAX_PARAGRAPH_LENGTH_SCENES),
      num_samples: int = 1,
      client: Optional[LanguageAPI] = None,
      filter: Optional[FilterAPI] = None):
    self._prefixes = prefixes
    self._max_paragraph_length = max_paragraph_length
    self._max_paragraph_length_characters = max_paragraph_length_characters
    self._max_paragraph_length_scenes = max_paragraph_length_scenes
    self._num_samples = num_samples
    self._client = client
    self._filter = filter

    # Prompts and outputs of the hierarchical generator are organised in levels.
    self.prompts = {
        'title': '',
        'characters': '',
        'scenes': '',
        'places': {
            '': ''
        },
        'dialogs': ['']
    }
    self._title = Title('')
    self._characters = Characters({'': ''})
    self._scenes = Scenes([Scene('', '', '')])
    self._places = {'': Place('', '')}
    self._dialogs = ['']

    # History of interventions.
    self.interventions = {}
    self._set_storyline(storyline)

  def _set_storyline(self, storyline: str):
    """Set storyline and initialise the outputs of the generator."""
    self._level = 0

    # Add period to the end of the storyline, unless there is already one there.
    if storyline.find('.') == -1:
      storyline = storyline + '.'
    self._storyline = storyline

    # Keep track of each storyline intervention.
    timestamp = time.time()
    self.interventions[timestamp] = 'STORYLINE\n' + storyline

  @property
  def seed(self):
    return self._client.seed

  @property
  def title(self) -> Title:
    """Return the title."""
    return self._title

  @property
  def characters(self) -> Characters:
    """Return the characters."""
    return self._characters

  @property
  def scenes(self) -> Scenes:
    """Return the title."""
    return self._scenes

  @property
  def places(self) -> Dict[str, Place]:
    """Return the places."""
    return self._places

  @property
  def dialogs(self) -> List[str]:
    """Return the dialogs."""
    return self._dialogs

  def title_str(self) -> str:
    """Return the title as a string."""
    return self._title.title

  def num_scenes(self) -> int:
    """Return the number of scenes."""
    return self._scenes.num_scenes()

  def step(self,
           level: Optional[int] = None,
           seed: Optional[int] = None,
           idx: Optional[int] = None) -> bool:
    """Step down a level in the hierarchical generation of a story."""

    # Move to the next level of hierarchical generation.
    if level is None:
      level = self._level
    if level < 0 or level >= len(self.level_names):
      raise ValueError('Invalid level encountered on step.')
    level += 1
    self._level = level

    # Keep track of each step intervention.
    timestamp = time.time()
    self.interventions[timestamp] = 'STEP ' + str(level) + '\n'

    if level == 1:
      # Step 1: Generate title given a storyline.
      (title, titles_prefix) = generate_title(
          storyline=self._storyline,
          prefixes=self._prefixes,
          client=self._client,
          filter=self._filter,
          num_samples=self._num_samples,
          seed=seed)
      self._title = title
      self.prompts['title'] = titles_prefix
      self.interventions[timestamp] += title.to_string()
      success = len(title.title) > 0
      return success

    if level == 2:
      # Step 2: Generate characters given a storyline.
      (characters, character_prompts) = generate_characters(
          storyline=self._storyline,
          prefixes=self._prefixes,
          client=self._client,
          filter=self._filter,
          num_samples=self._num_samples,
          max_paragraph_length=self._max_paragraph_length_characters,
          seed=seed)
      self._characters = characters
      self.prompts['characters'] = character_prompts
      self.interventions[timestamp] += characters.to_string()
      success = len(characters.character_descriptions) > 0
      return success

    if level == 3:
      # Step 3: Generate sequence of scenes given a storyline and characters.
      characters = self._characters
      (scenes, scene_prompts) = generate_scenes(
          storyline=self._storyline,
          character_descriptions=get_character_descriptions(characters),
          prefixes=self._prefixes,
          client=self._client,
          filter=self._filter,
          num_samples=self._num_samples,
          max_paragraph_length=self._max_paragraph_length_scenes,
          seed=seed)
      self._scenes = scenes
      self.prompts['scenes'] = scene_prompts
      self.interventions[timestamp] += scenes.to_string()
      success = len(scenes.scenes) > 0
      return success

    if level == 4:
      # Step 4: For each scene, generate place descriptions given place name.
      scenes = self._scenes
      (place_descriptions, place_prompts) = generate_place_descriptions(
          storyline=self._storyline,
          scenes=scenes,
          prefixes=self._prefixes,
          client=self._client,
          filter=self._filter,
          num_samples=self._num_samples,
          seed=seed)
      self._places = place_descriptions
      self.prompts['places'] = place_prompts
      for place_name in place_descriptions:
        place = place_descriptions[place_name]
        if place:
          self.interventions[timestamp] += place.to_string()
      num_places = scenes.num_places()
      success = (len(place_descriptions) == num_places) and num_places > 0
      return success

    if level == 5:
      # Step 5: For each scene, generate dialog from scene information.
      title = self._title
      characters = self._characters
      scenes = self._scenes
      place_descriptions = self._places
      if idx is None:
        (dialogs, dialog_prompts) = zip(*[
            generate_dialog(
                storyline=self._storyline,
                scenes=scenes.scenes[:(k + 1)],
                character_descriptions=(characters.character_descriptions),
                place_descriptions=place_descriptions,
                prefixes=self._prefixes,
                max_paragraph_length=self._max_paragraph_length,
                max_num_repetitions=MAX_NUM_REPETITIONS,
                client=self._client,
                filter=self._filter,
                num_samples=self._num_samples,
                seed=seed) for k in range(len(scenes.scenes))
        ])
      else:
        num_scenes = self._scenes.num_scenes()
        while len(self._dialogs) < num_scenes:
          self._dialogs.append('')
        while len(self.prompts['dialogs']) < num_scenes:
          self.prompts['dialogs'].append('')
        if idx >= num_scenes or idx < 0:
          raise ValueError('Invalid scene index.')
        dialogs = self._dialogs
        dialog_prompts = self.prompts['dialogs']
        dialogs[idx], dialog_prompts[idx] = generate_dialog(
            storyline=self._storyline,
            scenes=scenes.scenes[:(idx + 1)],
            character_descriptions=(characters.character_descriptions),
            place_descriptions=place_descriptions,
            prefixes=self._prefixes,
            max_paragraph_length=self._max_paragraph_length,
            max_num_repetitions=MAX_NUM_REPETITIONS,
            client=self._client,
            filter=self._filter,
            num_samples=self._num_samples,
            seed=seed)
      self._dialogs = dialogs
      self.prompts['dialogs'] = dialog_prompts
      for dialog in dialogs:
        self.interventions[timestamp] += str(dialog)
      return True

  def get_story(self):
    if self._characters is not None:
      character_descriptions = get_character_descriptions(self._characters)
    else:
      character_descriptions = None
    return Story(
        storyline=self._storyline,
        title=self._title.title,
        character_descriptions=character_descriptions,
        place_descriptions=self._places,
        scenes=self._scenes,
        dialogs=self._dialogs)

  def rewrite(self, text, level=0, entity=None):
    if level < 0 or level >= len(self.level_names):
      raise ValueError('Invalid level encountered on step.')
    prompt_diff = None

    if level == 0:
      # Step 0: Rewrite the storyline and begin new story.
      prompt_diff = diff_prompt_change_str(self._storyline, text)
      self._set_storyline(text)

    if level == 1:
      # Step 1: Rewrite the title.
      title = Title.from_string(text)
      prompt_diff = diff_prompt_change_str(self._title.title, title.title)
      self._title = title

    if level == 2:
      # Step 2: Rewrite the characters.
      characters = Characters.from_string(text)
      prompt_diff = diff_prompt_change_dict(
          self._characters.character_descriptions,
          characters.character_descriptions)
      self._characters = characters

    if level == 3:
      # Step 3: Rewrite the sequence of scenes.
      scenes = Scenes.from_string(text)
      prompt_diff = diff_prompt_change_scenes(self._scenes.scenes,
                                              scenes.scenes)
      self._scenes = scenes

    if level == 4:
      # Step 4: For a given place, rewrite its place description.
      place_descriptions = self._places
      if entity in place_descriptions:
        place_prefix = Place.format_prefix(entity)
        text = place_prefix + text
        place = Place.from_string(entity, text)
        prompt_diff = diff_prompt_change_str(self._places[entity].name,
                                             place.name)
        prompt_diff += '\n' + diff_prompt_change_str(
            self._places[entity].description, place.description)

        self._places[entity] = place

    if level == 5:
      # Step 5: Rewrite the dialog of a given scene.
      dialogs = self._dialogs
      num_scenes = len(self._scenes.scenes)
      if entity >= 0 and entity < num_scenes:
        prompt_diff = diff_prompt_change_str(self._dialogs[entity], text)
        self._dialogs[entity] = text

    # Keep track of each rewrite intervention.
    if prompt_diff is not None and len(prompt_diff) > 0:
      timestamp = time.time()
      self.interventions[timestamp] = 'REWRITE ' + self.level_names[level]
      if entity:
        self.interventions[timestamp] += ' ' + str(entity)
      self.interventions[timestamp] += prompt_diff

  def complete(self,
               level=0,
               seed=None,
               entity=None,
               sample_length=SAMPLE_LENGTH):
    if level < 0 or level >= len(self.level_names):
      raise ValueError('Invalid level encountered on step.')
    prompt_diff = None

    if level == 2:
      # Step 2: Complete the characters.
      text_characters = self._characters.to_string()
      text_characters = strip_remove_end(text_characters)
      prompt = self.prompts['characters'] + text_characters
      text = generate_text(
          generation_prompt=prompt,
          client=self._client,
          filter=self._filter,
          sample_length=sample_length,
          max_paragraph_length=sample_length,
          seed=seed,
          num_samples=1)
      new_characters = Characters.from_string(text_characters + text)
      prompt_diff = diff_prompt_change_dict(
          self._characters.character_descriptions,
          new_characters.character_descriptions)
      self._characters = new_characters

    if level == 3:
      # Step 3: Complete the sequence of scenes.
      text_scenes = self._scenes.to_string()
      text_scenes = strip_remove_end(text_scenes)
      prompt = self.prompts['scenes'] + text_scenes
      text = generate_text(
          generation_prompt=prompt,
          client=self._client,
          filter=self._filter,
          sample_length=sample_length,
          max_paragraph_length=sample_length,
          seed=seed,
          num_samples=1)
      new_scenes = Scenes.from_string(text_scenes + text)
      prompt_diff = diff_prompt_change_scenes(self._scenes.scenes,
                                              new_scenes.scenes)
      self._scenes = new_scenes

    if level == 5:
      # Step 5: Complete the dialog of a given scene.
      dialogs = self._dialogs
      num_scenes = len(self._scenes.scenes)
      while len(self._dialogs) < num_scenes:
        self._dialogs.append('')
      while len(self.prompts['dialogs']) < num_scenes:
        self.prompts['dialogs'].append('')
      if entity >= 0 and entity < num_scenes:
        prompt = (self.prompts['dialogs'][entity] + self._dialogs[entity])
        text = generate_text(
            generation_prompt=prompt,
            client=self._client,
            filter=self._filter,
            sample_length=sample_length,
            max_paragraph_length=sample_length,
            seed=seed,
            num_samples=1)
        new_dialog = self._dialogs[entity] + text
        prompt_diff = diff_prompt_change_str(self._dialogs[entity], new_dialog)
        self._dialogs[entity] = new_dialog

    # Keep track of each rewrite intervention.
    if prompt_diff is not None and len(prompt_diff) > 0:
      timestamp = time.time()
      self.interventions[timestamp] = 'COMPLETE ' + self.level_names[level]
      if entity:
        self.interventions[timestamp] += ' ' + str(entity)
      self.interventions[timestamp] += prompt_diff


# ------------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------------


class GenerationAction:
  NEW = 1
  CONTINUE = 2
  REWRITE = 3


class GenerationHistory:
  """Custom data structure to handle the history of GenerationAction edits:

  NEW, CONTINUE or REWRITE. Consecutive REWRITE edits do not add to history.
  """

  def __init__(self):
    self._items = []
    self._actions = []
    self._idx = -1
    self._locked = False

  def _plain_add(self, item, action: GenerationAction):
    self._items.append(item)
    self._actions.append(action)
    self._idx = len(self._items) - 1
    return self._idx

  def add(self, item, action: GenerationAction):
    if len(self._items) == 0 or action != GenerationAction.REWRITE:
      return self._plain_add(item, action)
    last_action = self._actions[-1]
    if last_action != GenerationAction.REWRITE:
      return self._plain_add(item, action)
    self._items[self._idx] = item
    return self._idx

  def previous(self):
    if len(self._items) == 0:
      return None
    self._idx = max(self._idx - 1, 0)
    return self._items[self._idx]

  def next(self):
    if len(self._items) == 0:
      return None
    self._idx = min(self._idx + 1, len(self._items) - 1)
    return self._items[self._idx]

filter = None

print('Dramatron set-up complete.')

GPT3_API_KEY = 'gpt3_api_key'

import openai

class GPT3API(LanguageAPI):
  """A class wrapping the GPT-3 model API from OpenAI."""

  def __init__(self,
               sample_length: int,
               model: Optional[str] = None,
               model_param: Optional[str] = None,
               config_sampling: Optional[dict] = None,
               seed: Optional[int] = None,
               max_retries: int = _MAX_RETRIES,
               timeout: float = _TIMEOUT):
    """Initializer.

    Args:
      sample_length: Length of text to sample from model.
      model: The model name to correct to. An error will be raised if it does
        not exist.
      model_param: GPT-3 API key.
      config_sampling: ConfigDict with parameters.
      seed: Random seed for sampling.
      max_retries: Maximum number of retries for the remote API.
      timeout: Maximum waiting timeout
    """
    super().__init__(sample_length=sample_length,
                     model=model,
                     model_param=model_param,
                     config_sampling=config_sampling,
                     seed=seed,
                     max_retries=max_retries,
                     timeout=timeout)

    # Set the OpenAI key.
    openai.api_key = self._model_param

  @property
  def client(self):
    return self._client

  @property
  def model_metadata(self):
    return {'engine': self._model,
            'api_key': self._model_param,
            'max_tokens': self._sample_length}

  def sample(self,
             prompt: str,
             sample_length: Optional[int] = None,
             seed: Optional[int] = None,
             num_samples: int = 1):
    """Sample model with provided prompt and optional sample_length and seed."""
    if sample_length is None:
      sample_length = self._sample_length
    if seed is None:
      seed = self._seed

    for attempt in range(self._max_retries):
      try:
        result_gpt3 = openai.Completion.create(
            engine=self._model,
            prompt=prompt,
            max_tokens=sample_length,
            temperature=self._config_sampling['temp'],
            top_p=self._config_sampling['prob'],
            frequency_penalty=self._config_sampling['frequency_penalty'],
            presence_penalty=self._config_sampling['presence_penalty'],
            n=num_samples)
        results = []
        for k in range(len(result_gpt3["choices"])):
          results.append(LanguageResponse(
              text=result_gpt3["choices"][k]["text"],
              text_length=len(result_gpt3["choices"][k]["text"]),
              prompt=prompt,
              prompt_length=result_gpt3["usage"]["prompt_tokens"]))
        return results
      except Exception as ex:
        print(f'Attempt {attempt + 1} of {self._max_retries}: API call failed.')
        last_exception = ex
    raise last_exception


# Create the config.
config = {}
config['language_api_name'] = 'GPT3API'
config['model_api_key'] = GPT3_API_KEY
config['model_name'] = GPT3_ENGINE
config['max_retries'] = MAX_RETRIES
config['sample_length'] = SAMPLE_LENGTH
config['max_paragraph_length'] = MAX_PARAGRAPH_LENGTH
config['max_paragraph_length_characters'] = MAX_PARAGRAPH_LENGTH_CHARACTERS
config['max_paragraph_length_scenes'] = MAX_PARAGRAPH_LENGTH_SCENES
config['sampling'] = {}
config['sampling']['prob'] = SAMPLING_PROB
config['sampling']['temp'] = SAMPLING_TEMP
config['sampling']['frequency_penalty'] = 0.23
config['sampling']['presence_penalty'] = 0.23
config['prefixes'] = {}
config['file_dir'] = None

print('Config:')
for key, value in config.items():
  if key != 'prefixes':
    print(f'{key}: {value}')

client = GPT3API(
    model_param=config['model_api_key'],
    model=config['model_name'],
    seed=DEFAULT_SEED,
    sample_length=config['sample_length'],
    max_retries=config['max_retries'],
    config_sampling=config['sampling'])

print(f'Client model metadata: {client.model_metadata}')

medea_prefixes = {}
medea_prefixes['CHARACTERS_PROMPT'] = """
Example 1. Ancient Greek tragedy based upon the myth of Jason and Medea. Medea, a former princess and the wife of Jason, finds her position in the Greek world threatened as Jason leaves Medea for a Greek princess of Corinth. Medea takes vengeance on Jason by murdering his new wife as well as Medea's own two sons, after which she escapes to Athens.
""" + CHARACTER_MARKER + """Medea """ + DESCRIPTION_MARKER + """ Medea is the protagonist of the play. A sorceress and a princess, she fled her country and family to live with Jason in Corinth, where they established a family of two children and gained a favorable reputation. Jason has divorced Medea and taken up with a new family.""" + STOP_MARKER + """
""" + CHARACTER_MARKER + """Jason """ + DESCRIPTION_MARKER + """ Jason is considered the play's villain, though his evil stems more from weakness than strength. A former adventurer, Jason abandons his wife, Medea, in order to marry the beautiful young daughter of Creon, King of Corinth, and fuels Medea to a revenge.""" + STOP_MARKER + """
""" + CHARACTER_MARKER + """Women of Corinth """ + DESCRIPTION_MARKER + """ The Women of Corinth are a commentator to the action. They fully sympathizes with Medea's plight, excepting her decision to murder her own children.""" + STOP_MARKER + """
""" + CHARACTER_MARKER + """Creon """ + DESCRIPTION_MARKER + """ Creon is the King of Corinth, banishes Medea from the city.""" + STOP_MARKER + """
""" + CHARACTER_MARKER + """The Nurse """ + DESCRIPTION_MARKER + """ The Nurse is the caretaker of the house and of the children and serves as Medea's confidant.""" + STOP_MARKER + """
""" + END_MARKER + """
Example 2. """


medea_prefixes['SCENE_PROMPT'] = """
Example 1. Ancient Greek tragedy based upon the myth of Jason and Medea. Medea, a former princess and the wife of Jason, finds her position in the Greek world threatened as Jason leaves Medea for a Greek princess of Corinth. Medea takes vengeance on Jason by murdering his new wife as well as Medea's own two sons, after which she escapes to Athens.
Medea is the protagonist of the play. A sorceress and a princess, she fled her country and family to live with Jason in Corinth, where they established a family of two children and gained a favorable reputation. Jason has divorced Medea and taken up with a new family.
Jason can be considered the play's villain, though his evil stems more from weakness than strength. A former adventurer, Jason abandons his wife, Medea, in order to marry the beautiful young daughter of Creon, King of Corinth, and fuels Medea to a revenge.
The Women of Corinth serve as a commentator to the action. They fully sympathizes with Medea's plight, excepting her decision to murder her own children.
The King of Corinth Creon banishes Medea from the city.
The Messenger appears only once in the play to bear tragical news.
The Nurse is the caretaker of the house and of the children and serves as Medea's confidant.
The Tutor of the children is a very minor character and mainly acts as a messenger.

""" + SCENES_MARKER + """

""" + PLACE_ELEMENT + """Medea's modest home.
""" + PLOT_ELEMENT + """Exposition.
""" + BEAT_ELEMENT + """The Nurse recounts the chain of events that have turned Medea's world to enmity. The Nurse laments how Jason has abandoned Medea and his own children in order to remarry with the daughter of Creon.

""" + PLACE_ELEMENT + """Medea's modest home.
""" + PLOT_ELEMENT + """Inciting Incident.
""" + BEAT_ELEMENT + """The Nurse confides in the Tutor amd testifies to the emotional shock Jason's betrayal has sparked in Medea. The Tutor shares the Nurse's sympathy for Medea's plight. Medea's first words are cries of helplessness. Medea wishes for her own death.

""" + PLACE_ELEMENT + """Medea's modest home.
""" + PLOT_ELEMENT + """Conflict.
""" + BEAT_ELEMENT + """The Women of Corinth address Medea and try to reason with Medea and convince her that suicide would be an overreaction. The Nurse recognizes the gravity of Medea's threat.

""" + PLACE_ELEMENT + """Outside the Royal Palace.
""" + PLOT_ELEMENT + """Rising Action.
""" + BEAT_ELEMENT + """Medea pleads to the Nurse that Jason be made to suffer for the suffering he has inflicted upon her. Creon approaches the house and banishes Medea and her children from Corinth. Medea plans on killing her three antagonists, Creon, his daughter and Jason.

""" + PLACE_ELEMENT + """Outside the Royal Palace.
""" + PLOT_ELEMENT + """Dilemma.
""" + BEAT_ELEMENT + """Jason rebuke Medea for publicly expressing her murderous intentions. Jason defends his choice to remarry. Medea refuses Jason's offers and sends him away to his new bride.

""" + PLACE_ELEMENT + """Outside the Royal Palace.
""" + PLOT_ELEMENT + """Climax.
""" + BEAT_ELEMENT + """When Jason returns, Medea begins to carry out her ruse. Medea fakes regret and break down in false tears of remorse. Determined, Medea sends her children to offer poisoned gifts to Creon's daughter. Medea's children face impending doom.

""" + PLACE_ELEMENT + """Outside the Royal Palace.
""" + PLOT_ELEMENT + """Falling Action.
""" + BEAT_ELEMENT + """The Messenger frantically runs towards Medea and warns Medea to escape the city as soon as possible. The Messenger reveals that Medea has been identified as the murderer.

""" + PLACE_ELEMENT + """Outside the Royal Palace.
""" + PLOT_ELEMENT + """Resolution.
""" + BEAT_ELEMENT + """Medea and her two dead children are seated in a chariot drawn by dragons. Jason watches in horror and curses himself for having wed Medea and mourns his tragic losses.

""" + PLACE_ELEMENT + """On a winged chariot.
""" + PLOT_ELEMENT + """Dénouement.
""" + BEAT_ELEMENT + """Medea denies Jason the right to a proper burial of his children. She flees to Athens and divines an unheroic death for Jason.

""" + END_MARKER + """
Example 2. """


medea_prefixes['SETTING_PROMPT'] = """
Example 1. Ella, a waitress, falls in love with her best friend, Allen, a teacher. The two drift apart when Allen makes new friends from a different social class. Ella turns to food to become a famous chef.
""" + PLACE_ELEMENT + """The bar.
""" + DESCRIPTION_ELEMENT + """The bar is dirty, more than a little run down, with most tables empty. The odor of last night's beer and crushed pretzels on the floor permeates the bar.""" + END_MARKER + """
Example 2. Grandma Phyllis’ family reunion with her two grandchildren is crashed by two bikers.
""" + PLACE_ELEMENT + """The Lawn in Front of Grandma Phyllis's House.
""" + DESCRIPTION_ELEMENT + """A big oak tree dominates the yard. There is an old swing set on the lawn, and a bright white fence all around the grass.""" + END_MARKER + """
Example 3. Ancient Greek tragedy based upon the myth of Jason and Medea. Medea, a former princess and the wife of Jason, finds her position in the Greek world threatened as Jason leaves Medea for a Greek princess of Corinth. Medea takes vengeance on Jason by murdering his new wife as well as Medea's own two sons, after which she escapes to Athens.
""" + PLACE_ELEMENT + """Outside the Royal Palace.
""" + DESCRIPTION_ELEMENT + """In mythological Ancient Greece, in front of a modest house in Corinth, on the outskirts of a lavish royal palace where wedding preparations are under way.""" + END_MARKER + """
Example 4. """


medea_prefixes['TITLES_PROMPT'] = """
Examples of alternative, original and descriptive titles for known play and film scripts.

Example 1. Ancient Greek tragedy based upon the myth of Jason and Medea. Medea, a former princess of the kingdom of Colchis, and the wife of Jason, finds her position in the Greek world threatened as Jason leaves her for a Greek princess of Corinth. Medea takes vengeance on Jason by murdering his new wife as well as her own two sons, after which she escapes to Athens. Title: A Feminist Tale""" + END_MARKER + """
Example 2. Ancient Greek tragedy that deals with Antigone’s burial of her brother Polynices, in defiance of the laws of Creon and the state, and the tragic repercussions of her act of civil disobedience. Title: In My Brother's Name""" + END_MARKER + """
Example 3. Greek comedy that tells the story of the god Dionysus (also known to the Greeks as Bacchus) who, despairing of the current state of Athens’ tragedians, travels to Hades with his slave Xanthias to bring Euripides back from the dead. Title: Dionysus in Hades""" + END_MARKER + """
Example 4. """


medea_prefixes['DIALOG_PROMPT'] = """
Example 1.
""" + PLACE_ELEMENT + """Outside the Royal Palace.
""" + DESCRIPTION_ELEMENT + """Before Medea's house in Corinth, near the royal palace of Creon.
""" + CHARACTERS_ELEMENT + """Medea is the protagonist of the play. A sorceress and a princess, she fled her country and family to live with Jason in Corinth, where they established a family of two children and gained a favorable reputation. Jason has divorced Medea and taken up with a new family. Jason can be considered the play's villain, though his evil stems more from weakness than strength. A former adventurer, Jason abandons his wife, Medea, in order to marry the beautiful young daughter of Creon, King of Corinth, and fuels Medea to a revenge. The Messenger appears only once in the play to bear tragical news.
""" + PLOT_ELEMENT + """Resolution.
""" + SUMMARY_ELEMENT + """Ancient Greek tragedy based upon the myth of Jason and Medea. Medea, a former princess and the wife of Jason, finds her position in the Greek world threatened as Jason leaves Medea for a Greek princess of Corinth. Medea takes vengeance on Jason by murdering his new wife as well as Medea's own two sons, after which she escapes to Athens.
""" + PREVIOUS_ELEMENT + """The Messenger frantically warns Medea to escape the city as soon as possible. The Messenger reveals that Medea has been identified as the murderer.
""" + BEAT_ELEMENT + """The palace opens its doors, revealing Medea and the two dead children seated in a chariot drawn by dragons. Jason curses himself for having wed Medea and mourns his tragic losses. Medea denies Jason the right to a proper burial of his children. Medea flees to Athens and divines an unheroic death for Jason.

""" + DIALOG_MARKER + """

WOMEN OF CORINTH
Throw wide the doors and see thy children's murdered corpses.

JASON
Haste, ye slaves, loose the bolts, undo the fastenings, that
I may see the sight of twofold woe, my murdered sons and her, whose
blood in vengeance I will shed.  (MEDEA appears above the house, on
a chariot drawn by dragons; the children's corpses are beside her.)

MEDEA
Why shake those doors and attempt to loose their bolts, in
quest of the dead and me their murderess? From such toil desist. If
thou wouldst aught with me, say on, if so thou wilt; but never shalt
thou lay hand on me, so swift the steeds the sun, my father's sire,
to me doth give to save me from the hand of my foes.

JASON
Accursed woman! by gods, by me and all mankind abhorred as
never woman was, who hadst the heart to stab thy babes, thou their
mother, leaving me undone and childless; this hast thou done and still
dost gaze upon the sun and earth after this deed most impious. Curses
on thee! now perceive what then I missed in the day I brought thee,
fraught with doom, from thy home in a barbarian land to dwell in Hellas,
traitress to thy sire and to the land that nurtured thee.
Perish, vile sorceress, murderess of
thy babes! Whilst I must mourn my luckless fate, for I shall ne'er
enjoy my new-found bride, nor shall I have the children, whom I bred
and reared, alive to say the last farewell to me; nay, I have lost
them.

MEDEA
To this thy speech I could have made a long reply, but Father
Zeus knows well all I have done for thee, and the treatment thou hast
given me. Yet thou wert not ordained to scorn my love and lead a life
of joy in mockery of me, nor was thy royal bride nor Creon, who gave
thee a second wife, to thrust me from this land and rue it not. Wherefore,
if thou wilt, call me e'en a lioness, and Scylla, whose home is in
the Tyrrhene land; for I in turn have wrung thy heart, as well I might.

JASON
Thou, too, art grieved thyself, and sharest in my sorrow.

MEDEA
Be well assured I am; but it relieves my pain to know thou
canst not mock at me.

JASON
O my children, how vile a mother ye have found!

MEDEA
My sons, your father's feeble lust has been your ruin!

JASON
'Twas not my hand, at any rate, that slew them.

MEDEA
No, but thy foul treatment of me, and thy new marriage.

JASON
Didst think that marriage cause enough to murder them?

MEDEA
Dost think a woman counts this a trifling injury?

JASON
So she be self-restrained; but in thy eyes all is evil.

MEDEA
Thy sons are dead and gone. That will stab thy heart.
""" + END_MARKER + """

Example 2.
"""

prefix_set = 'medea_prefixes' #@param ['medea_prefixes', 'scifi_prefixes', 'custom_prefixes']
prefixes = eval(prefix_set)
config['prefixes'] = prefixes

print(f'Loaded {prefix_set}.')

# test
logline = "Folk tale about a rabbit, a fox and a crow living in an enchanted forest. The cunning animals safeguard the golden apple tree from a greedy lumberjack and conspire to hide the lumberjack's axe." #@param {type:"string"}
# print(logline)

generator = StoryGenerator(
    storyline=logline,
    prefixes=prefixes,
    max_paragraph_length=config['max_paragraph_length'],
    client=client,
    filter=filter)

print(f'New Dramatron generator created.')

story = None

####################
### Create title ###
####################

# data_title = {"text": "", "text_area": None, "seed": generator.seed - 1}

def fun_generate_title():
  title_seed = generator.seed - 1
  seed = title_seed + 1
  # seed = 5
  print(f"Generating {seed}...")
  generator.step(0, seed=seed)
  title = generator.title_str().strip()
  return title

# def fun_rewrite_title(text):
#   text_to_parse = TITLE_ELEMENT + text + END_MARKER
#   generator.rewrite(text_to_parse, level=1)
#   return text

# generated_title = fun_generate_title()
# print(f'Generated title: {generated_title}.')

########################
### Create character ###
########################

# data_chars = {"text": "", "text_area": None, "seed": generator.seed - 1,
#               "history": GenerationHistory(), "lock": False}

def fun_generate_characters():
  new_character = ""
  characters_seed = generator.seed - 1
  seed = characters_seed
  # data_chars["lock"] = True
  while True:
    generator.step(1, seed=seed)
    new_character = strip_remove_end(generator.characters.to_string())
    # Test if characters were actually generated.
    if len(new_character) == 0:
      seed += 1
    else:
      break
  characters_seed = seed
  # data_chars["history"].add(data_chars["text"], GenerationAction.NEW)

  # if data_chars["text_area"] is not None:
  #   data_chars["text_area"].value = data_chars["text"]
  # data_chars["lock"] = False
  return new_character

# def fun_continue_characters(_):
#   data_chars["seed"] += 1
#   seed = data_chars["seed"]
#   data_chars["lock"] = True

#   generator.complete(level=2, seed=seed, sample_length=256)
#   data_chars["text"] = strip_remove_end(generator.characters.to_string())
#   data_chars["history"].add(data_chars["text"], GenerationAction.CONTINUE)

#   if data_chars["text_area"] is not None:
#     data_chars["text_area"].value = data_chars["text"]
#   data_chars["lock"] = False

# def fun_back_forward(data, history: GenerationHistory, delta: int):
#   data["lock"] = True
#   if delta > 0:
#     data["text"] = history.next()
#   if delta < 0:
#     data["text"] = history.previous()
#   if data["text"] is not None and data["text_area"] is not None:
#       data["text_area"].value = data["text"]
#   data["lock"] = False

# def fun_back_characters(_):
#   fun_back_forward(data_chars, data_chars["history"], -1)

# def fun_forward_characters(_):
#   fun_back_forward(data_chars, data_chars["history"], 1)

# def fun_rewrite_characters(text):
#   data_chars["text"] = text
#   text_to_parse = text + END_MARKER
#   generator.rewrite(text_to_parse, level=2)
#   if data_chars["lock"] is False:
#     data_chars["history"].add(text, GenerationAction.REWRITE)
#   return text

# textarea_chars = widgets.interactive(
#     fun_rewrite_characters, text=data_chars["text_area"])
# display(textarea_chars)

# Trigger generation for first seed.
# generated_character = fun_generate_characters()
# print(f'Generated character: {generated_character}.')

#####################
### Create scenes ###
#####################

# data_scenes = {"text": None, "text_area": None, "seed": generator.seed - 1,
#                "history": GenerationHistory(), "lock": False}

def fun_generate_scenes():
  scenes_seed = generator.seed - 1
  seed = scenes_seed
  # data_scenes["lock"] = True
  generator.step(2, seed=seed)
  scene_text = strip_remove_end(generator.scenes.to_string())
  # data_scenes["history"].add(data_scenes["text"], GenerationAction.NEW)
  # data_scenes["lock"] = False
  return scene_text

# def fun_continue_scenes(_):
#   data_scenes["seed"] += 1
#   seed = data_scenes["seed"]
#   data_scenes["lock"] = True
#   scenes_continue_button.description = f"Generating {seed}..."
#   generator.complete(level=3, seed=seed, sample_length=256)
#   data_scenes["text"] = strip_remove_end(generator.scenes.to_string())
#   data_scenes["history"].add(data_scenes["text"], GenerationAction.CONTINUE)
#   scenes_continue_button.description = "Continue generation"
#   if data_scenes["text_area"] is not None:
#     data_scenes["text_area"].value = data_scenes["text"]
#   data_scenes["lock"] = False

# def fun_back_scenes(_):
#   fun_back_forward(data_scenes, data_scenes["history"], -1)

# def fun_forward_scenes(_):
#   fun_back_forward(data_scenes, data_scenes["history"], 1)

# def fun_rewrite_scenes(text):
#   generator.rewrite(text, level=3)
#   if data_scenes["lock"] is False:
#     data_scenes["history"].add(text, GenerationAction.REWRITE)
#   return text

# Trigger generation for first seed.
# generated_scenes = fun_generate_scenes()

#####################
### Create places ###
#####################

place_names = list(set([scene.place for scene in generator.scenes[0]]))
place_descriptions = {place_name: Place(place_name, '')
                      for place_name in place_names}
# data_places = {"descriptions": place_descriptions, "text_area": {},
#                "seed": generator.seed - 1}
data_places = []

def fun_generate_places():
  places_seed = generator.seed - 1
  seed = places_seed
  # Update existing text areas with a waiting message.
  # Generate all the places.
  generator.step(3, seed=seed)
  generator_places = generator.places
  # missing_places = {k: True for k in data_places["text_area"].keys()}
  # for place_name, place_description in generator_places.items():
  #   if place_name in data_places:
  #     description = place_description.description
  #     data_places["text_area"][place_name].value = description
      # del missing_places[place_name]
    # else:
    #   print(f"\nWarning: [{place_name}] was added to the plot synopsis.")
      # print(f"Make a copy of the outputs and re-run the cell.")
  # for place_name in missing_places:
  #   data_places["text_area"][place_name].value = (
  #       f"Warning: [{place_name}] was removed from the plot synopsis. "
  #       "Make a copy of the outputs and re-run the cell.")

  return generator_places

def fun_rewrite_places(place_name, text):
  generator.rewrite(text, level=4, entity=place_name)
  return text

# Widget to generate new scenes.
# new_places_button = widgets.Button(button_style='', icon='check',
#     description='Generate new', tooltip='Generate new', disabled=False)
# new_places_button.on_click(fun_generate_places)
# display(new_places_button)

# Render each place using widgets.
# for place_name, place_description in data_places["descriptions"].items():
#   text_place = place_description.description
#   layout = widgets.Layout(height='90px', min_height='100px', width='auto')
#   display(widgets.Label(place_name))
#   data_places["text_area"][place_name] = widgets.Textarea(
#       value=text_place, layout=layout, description=' ',
#       style={'description_width': 'initial'})
#   textarea_place = widgets.interactive(
#       fun_rewrite_places, place_name=widgets.fixed(place_name),
#       text=data_places["text_area"][place_name])
#   display(textarea_place)

# Trigger generation for first seed.
generated_places = fun_generate_places()
print(f'Generated places: {generated_places}.')

##############
### Render ###
##############

def render_story():
  # Render the story.
  story = generator.get_story()
  script_text = render_story(story)
  print(script_text)

  # Render the prompts.
  # prefix_text = render_prompts(generator.prompts)

  # Render the interventions.
  # edits_text = ''
  # for timestamp in sorted(generator.interventions):
  #   edits_text += 'EDIT @ ' + str(timestamp) + '\n'
  #   edits_text += generator.interventions[timestamp] + '\n\n\n'

  # # Prepare the filenames for saving the story and prompts.
  # timestamp_generation = datetime.datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')
  # title_ascii = re.sub('[^0-9a-zA-Z]+', '_',
  #                     generator.title_str().strip()).lower()
  # filename_script = f'{title_ascii}_{timestamp_generation}_script.txt'
  # filename_prefix = f'{title_ascii}_{timestamp_generation}_prefix.txt'
  # filename_edits = f'{title_ascii}_{timestamp_generation}_edits.txt'
  # filename_config = f'{title_ascii}_{timestamp_generation}_config.json'

  return script_text


import os.path
import json
import openai
from jinja2 import Environment, FileSystemLoader

from chat.gpt_agent import GptAgent
from constants.sentence_type_enum import SentenceType

"""
   1. Start with Hiragana and Katakana words
   2. Reading examples for any of them
   3. Once all have been seen at least 20 times and have > 90% accuracy, introduce new Kanji
   4. Introduce Kanji in order of most frequently used.
   5. Allow AI to pick from at most N previously seen Kanji as well
   6. Follow steps 2 and 3
   """

class LessonManager:
    _WORD_STATS_PATH = './data/word_stats.json'
    _KANJI_PATH = './data/kanji_freq_list.txt'

    ACCURACY_THRESHOLD = 0.8
    MIN_SEEN_THRESHOLD = 5
    MIN_WORDS = 20  # Always introduce new words if below
    REQ_HIRA_KATA_WORDS = 100  # Only introduce Kanji above this
    WORD_WINDOW = 5

    FOCUS_INSTRUCTIONS = {
        SentenceType.NEW_HIRAKATA: "consists of up to 5 Hiragana or Katakana words of your choosing. Do not use Kanji at all.",
        SentenceType.MORE_HIRAKATA: "uses any of the these words {{words}}, and one more Hiragana or Katakana word of your choosing. Do not use Kanji at all.",
        SentenceType.EXISTING_WORD: "uses one or more of these words: {{words}}. Do not use Kanji characters besides those listed here.",
        SentenceType.NEW_KANJI: "uses one or more of these words: {{words}}, as well as one additional Kanji or your choosing. Do not use Kanji characters besides those listed here or the new one you introduce.",
    }

    def __init__(self):
        self.word_stats = self._load_stats(self._WORD_STATS_PATH)

        with open("openai_key", 'r') as f:
            openai.api_key = f.read().strip()

        self.jinja_env = Environment(loader=FileSystemLoader("prompts/"))
        self.tutor = self._create_tutor()

    def _create_sentence_generator(self):
        # Get focus words
        sentence_type, focus_words = self.get_focus_words()
        focus_instructions = self.jinja_env.from_string(self.FOCUS_INSTRUCTIONS[sentence_type]).render(words=', '.join(focus_words))

        # Load main prompt
        gen_prompt = self.jinja_env.get_template("proposer_prompt.txt").render(focus_instructions=focus_instructions)
        print(f"Loaded sentence generator prompt:\n{gen_prompt}")
        return GptAgent(instruction_prompt=gen_prompt)

    def _create_tutor(self):
        prompt = self.jinja_env.get_template("tutor_prompt.txt").render()
        print(f"Loaded tutor prompt:\n{prompt}")
        tutor = GptAgent(instruction_prompt=prompt)
        return tutor

    def _create_marker(self, student_answer:str, ):
        prompt = self.jinja_env.get_template("marker_prompt.txt").render(exercise=self.cur_exercise, translation=student_answer)
        print(f"Loaded marker prompt:\n{prompt}")
        return GptAgent(instruction_prompt=prompt)

    def check_translation(self, translation):
        marker = self._create_marker(translation)
        marker.tell("What is your response to teacher A?", role="system")
        resp = marker.listen()
        print(f"Marker response:\n{resp}")
        self.tutor.tell(resp, role="user", speaker_name="Answer_checker")
        self.tutor.tell("With this information, how do you respond to the student?", role="system")
        return self.tutor.listen()

    def get_focus_words(self, count=1) -> (SentenceType, list[str]):
        """
        Picks the count words/kanji to focus on in the next exercise
        TODO Rewrite to avoid duplicate data and repeated calculations
        """
        # If not words shown yet ask for a new one
        if len(self.word_stats) < self.MIN_WORDS:
            print("Less that min words: Introduce new Hiragana or Katakana")
            return SentenceType.NEW_HIRAKATA, []

        # Calculate accuracies
        accuracies = [[c, correct / seen] if seen else [c, 0.0] for c, (seen, correct) in self.word_stats.items()]

        # Sort from lowest to highest
        accuracies = sorted(accuracies, key=lambda x: x[1])

        # If at least one is below 90% accurate, practice it
        if accuracies[0][1] < self.ACCURACY_THRESHOLD:
            print("Below accuracy threshold: Practice old words")
            return SentenceType.EXISTING_WORD, [c for c, _ in accuracies[:count]]

        # Otherwise check seen count
        seen_counts = sorted([[c, seen] for c, (seen, correct) in self.word_stats.items()], key=lambda x: x[1])

        # If at least one is below the seen threshold, practice it
        if seen_counts[0][1] < self.MIN_SEEN_THRESHOLD:
            print("Below seen threshold: Practice old words")
            return SentenceType.EXISTING_WORD, [c for c, _ in seen_counts[:count]]

        # If all else is fine, introduce a new character
        if len(self.word_stats) < self.REQ_HIRA_KATA_WORDS:
            # Hiragana and Katakana only
            print("Introduce more hiragana or katakana")
            return SentenceType.MORE_HIRAKATA, list(self.word_stats.keys())[:count - 1]
        else:
            # Kanji
            print("Introduce new kanji")
            return SentenceType.NEW_KANJI, [c for c, _ in seen_counts[:count - 1]]


    def _load_stats(self, stats_path: str):
        """
        Loads the stats for the specified dataset, creating them if necessary.
        stats are in the form {word: (seen count, correct count), ...}
        """

        if not os.path.exists(stats_path):
            stats = {}
            with open(stats_path, 'w') as f:
                json.dump(stats, f)
        else:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
        return stats

    def _save_stats(self):
        with open(self._WORD_STATS_PATH, 'w') as f:
            json.dump(self.word_stats, f)

    def _load_kanji(self, idx):
        with open(self._KANJI_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if idx >= len(lines):
                return None

            return lines[idx].strip()

    def inc_stats(self, exercise: list, stat_idx: int):
        for word, *_ in exercise:
            if word in self.word_stats:
                self.word_stats[word][stat_idx] += 1
            else:
                self.word_stats[word] = [1, 0]
        self._save_stats()

    def get_next_sentence(self):
        # Create agent
        exercise_gen = self._create_sentence_generator()

        # Prompt
        exercise_gen.tell("Let's do this step by step to make sure we get the correct format:", role="assistant")

        self.cur_exercise = ""
        for _ in range(5):
            resp = exercise_gen.listen()
            self.cur_exercise += f"\n{resp}"

            if "[STOP]" in resp:
                break

        # TODO validate output format
        assert "[STOP]" in self.cur_exercise
        self.cur_exercise = self.cur_exercise.replace("[STOP]", "")

        split = self.cur_exercise.strip().split("\n")
        sentence = split[0]
        sentence_disp = json.loads(split[-1])
        self.inc_stats(sentence_disp, 0)

        return sentence, sentence_disp



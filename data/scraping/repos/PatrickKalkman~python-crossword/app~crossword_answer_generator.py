from app.openai_response_processor import OpenAIResponseProcessor


class CrosswordAnswerGenerator(OpenAIResponseProcessor):
    def _create_prompt(self, category, word_length, num_words):
        return (f"Generate a list of {num_words} {word_length}-syllable "
                f"English words related to the category '{category}', suitable"
                "for a crossword puzzle.")

    def _parse_response(self, response_text):
        lines = response_text.strip().split('\n')
        words = [line.split(' ', 1)[1] for line in lines if len(line.split(' ', 1)) > 1]
        words = [word.lower().strip() for word in words]
        return words

    def _create_cache_filename(self, category, word_length, num_words):
        return f"cache/{category}_{word_length}_{num_words}.json"

    def get_additional_words(self):
        with open("cache/additional_words.txt", "r") as f:
            words = f.read().splitlines()
        return [word.lower().strip() for word in words]

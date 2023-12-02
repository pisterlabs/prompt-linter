import openai

from scripts.config import g_analyser_options


class OpenAIController:

    @staticmethod
    def hint():
        global g_openai_hints
        return \
            f"""
            Provide the data, formatted in an XML-fashion: <LOCATION>Country, City</LOCATION> <DESCRIPTION>...</DESCRIPTION>.
            In the description mark each landmark, town, facilities or other specific locations and establishments enclosed
            in a XML-like tag <LNDMARK> and each vacation activity verb phrase with <ACTIVITY>. Make sure to put corresponding
            closing tags. Also wrap each <LOCATION>-<DESCRIPTION> pair in a <DESTINATION> tag.
            """

    @staticmethod
    def ask(question: str):
        # Hint OpenAI in order to omit unnecessary data processing of the response.
        question += OpenAIController.hint()

        fmt_msg = lambda q: {'role': 'user', 'content': q}

        resp = openai.ChatCompletion.create(
            model=g_analyser_options['openai_model'],
            temperature=g_analyser_options['openai_temperature'],
            n=g_analyser_options['openai_responses_at_once'],
            messages=[fmt_msg(question)],
        )

        accumulated = ''
        for r in resp['choices']:
            accumulated += f"{r['message']['content']}\n"
        print(accumulated)
        return accumulated

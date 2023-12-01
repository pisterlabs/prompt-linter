import openai


class GrammarChecker:
    # Init with language
    def __init__(self, language: str):
        if language == "German":
            self.correction_string = (
                "Behebe die Rechtschreibung (falls es Fehler gibt)."
            )
        elif language == "English":
            self.correction_string = (
                "Fix the spelling in the given text if there are any mistakes."
            )
        elif language == "Spanish":
            self.correction_string = (
                "¡Corrige la ortografía y la gramática (si hay errores)!"
            )
        elif language == "Czech":
            self.correction_string = "Opravte pravopisné chyby!"

    # Check if the input is correct
    def check(self, input: str) -> str:
        """Returns the correct string"""
        completion = openai.Edit.create(
            model="text-davinci-edit-001",
            input=input,
            instruction=self.correction_string,
            temperature=0,
        )

        corrected_text: str = completion.choices[0].text
        # Can be buggy this way
        corrected_text = corrected_text.replace(self.correction_string, "")

        # Check if the corrected text is more than 20 % longer than the original text
        # Test example: "What do you love to do?"
        # This means that the correction model probably added some extra text :(
        if len(corrected_text) > len(input) * 1.2 and any([c in [".", ",", ";", ":", "!", "?"] for c in corrected_text]):
            # Something fishy is going on, the script generated garbage
            # Calculate the indices of separators
            separator_indices = [
                i
                for i, c in enumerate(corrected_text)
                if c in [".", ",", ";", ":", "!", "?"]
            ]
            try:
                last_separator_input_text = [
                    i for i, c in enumerate(input) if c in [".", ",", ";", ":", "!", "?"]
                ][-1]


                # Find the separator_index that is closest to the last separator_index of the input
                closest_separator_index = min(
                    separator_indices, key=lambda x: abs(x - last_separator_input_text)
                )
                # Return the text up to the closest separator_index
                return corrected_text[: closest_separator_index + 1]
            except IndexError:
                # I don't know the bug
                return corrected_text

        return corrected_text

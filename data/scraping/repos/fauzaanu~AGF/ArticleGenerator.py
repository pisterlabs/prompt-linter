import os
from pprint import pprint

import openai
from dotenv import load_dotenv


class ArticleGenerator:
    def __init__(self,text,api_key,filename=None):
        self.original_text = text
        self.filename = filename
        self.openai_key = api_key


    def generate_prompt(self, prompt, system, temperature=2.0, model="gpt-3.5-turbo"):
        """
        Generate a prompt using the OpenAI API.
        """

        # Set the OpenAI API key
        openai.api_key = self.openai_key
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": f"{system}"},
                {"role": "user", "content": f"{prompt}"},
            ],
            temperature=temperature,
        )

        # Return the generated output
        response_dict = response.to_dict()
        generated_output = response_dict["choices"][0]["message"]["content"]
        return generated_output

    def text_check(self,text):
        if text is None:
            return self.original_text
        else:
            return text


    def automate(self) -> str:
        """
        This function will automate the whole process of writing an article.
        Do not use this method if you want modularity. (Example: If you dont need to maintain clarity but go through the whole process, then dont use this method)
        :return:
        :rtype:
        """
        text = self.original_text
        pprint("Progress: Summarizing (1/13)")
        summary = self.summarize(text)
        pprint("Progress: Extracting keypoints (2/13)")
        keypoints = self.keypoints(summary)
        pprint("Progress: Paraphrasing (3/13)")
        paraphrase = self.paraphrase(keypoints)
        pprint("Progress: Eliminating redundancy (4/13)")
        eliminate_redundancy = self.eliminate_redundancy(paraphrase)
        pprint("Progress: Maintaining clarity (5/13)")
        maintain_clarity = self.maintain_clarity(eliminate_redundancy)
        pprint("Progress: Using own words (6/13)")
        use_own_words = self.use_own_words(maintain_clarity)
        pprint("Progress: Shortening lengthy explainations (7/13)")
        shorten_lenghty_explainations = self.shorten_lenghty_explainations(use_own_words)
        pprint("Progress: Organizing text by importance (8/13)")
        organize_text_by_importance = self.organize_text_by_importance(shorten_lenghty_explainations)
        pprint("Progress: Expanding (9/13)")
        expand = self.expand(organize_text_by_importance)
        pprint("Progress: Maintaining coherence (10/13)")
        maintain_coherence = self.maintain_coherence(expand)
        pprint("Progress: Adding transitions (11/13)")
        transitions = self.transitions(maintain_coherence)
        pprint("Progress: Adding conclusion (12/13)")
        conclusion = self.conclusion(transitions)
        pprint("Progress: Proofreading and finalizing (13/13)")
        proofread_and_finalize = self.proofread_and_finalize(conclusion)
        pprint("Progress: Reviewing for consistency (14/13)")
        review_for_consistency = self.review_for_consistency(proofread_and_finalize)
        pprint("Progress: Done - Writing to file")

        if self.filename is not None:
            return self.save(review_for_consistency)
        else:
            return review_for_consistency



    def summarize(self,text=None)-> str:
        text = self.text_check(text)

        system = "You are an expert in summarization given a text"
        prompt =  "Summarize the following text to re-write it as an article:" + text
        return self.generate_prompt(prompt, system, temperature=0.5)

    def keypoints(self,text=None)-> str:
        text = self.text_check(text)

        system = "You are an expert in figuring out the keypoints given a text"
        prompt =  "Extract the keypoints from the following text to re-write the following text as an article:" + text
        return self.generate_prompt(prompt, system, temperature=0.5)

    def paraphrase(self,text=None)-> str:
        text = self.text_check(text)

        system = "You are an expert in paraphrasing given a text"
        prompt =  "Paraphrase the following text to re-write the following text as an article:" + text
        return self.generate_prompt(prompt, system, temperature=0.5)

    def eliminate_redundancy(self,text=None)-> str:
        text = self.text_check(text)

        system = "You are an expert in eliminating redundancy given a text"
        prompt =  "Eliminate the redundancy in the following text to re-write the following text as an article:" + text
        return self.generate_prompt(prompt, system, temperature=0.5)

    def maintain_clarity(self,text=None)-> str:
        text = self.text_check(text)

        system = "You are an expert in maintaining clarity given a text"
        prompt =  "Maintain the clarity in the following text to re-write the following text as an article:" + text
        return self.generate_prompt(prompt, system, temperature=0.5)

    def use_own_words(self,text=None)-> str:
        text = self.text_check(text)

        system = "You are an expert in using your own words given a text"
        prompt =  "Use your own words to re-write the following text as an article:" + text
        return self.generate_prompt(prompt, system, temperature=0.5)

    def shorten_lenghty_explainations(self,text=None)-> str:
        text = self.text_check(text)

        system = "You are an expert in shortening lengthy explainations given a text"
        prompt =  "Shorten the following text to re-write the following text as an article:" + text
        return self.generate_prompt(prompt, system, temperature=0.5)

    def organize_text_by_importance(self,text=None)-> str:
        text = self.text_check(text)

        system = "You are an expert in organizing text by importance given a text"
        prompt =  "Organize the following text by importance to re-write the following text as an article:" + text
        return self.generate_prompt(prompt, system, temperature=0.5)

    def expand(self,text=None)-> str:
        text = self.text_check(text)

        system = "You are an expert in expanding a text to form an article"
        prompt =  "Expand the following text to re-write it as an article:" + text
        return self.generate_prompt(prompt, system, temperature=0.5)

    def maintain_coherence(self,text=None)-> str:
        text = self.text_check(text)

        system = "You are an expert in maintaining coherence given a text"
        prompt =  "Maintain the coherence in the following text to re-write the following text as an article:" + text
        return self.generate_prompt(prompt, system, temperature=0.5)

    def transitions(self,text=None)-> str:
        text = self.text_check(text)

        system = "You are an expert in transitions given a text"
        prompt =  "Add transitions to the following text to re-write the following text as an article:" + text
        return self.generate_prompt(prompt, system, temperature=0.5)

    def conclusion(self,text=None)-> str:
        text = self.text_check(text)

        system = "You are an expert in conclusion given a text"
        prompt =  "Add a conclusion to the following text to re-write the following text as an article:" + text
        return self.generate_prompt(prompt, system, temperature=0.5)

    def proofread_and_finalize(self,text=None)-> str:
        text = self.text_check(text)

        system = "You are an expert in proofreading and finalizing given a text"
        prompt =  "Proofread and finalize the following text to re-write the following text as an article:" + text
        return self.generate_prompt(prompt, system, temperature=0.5)

    def review_for_consistency(self,text=None)-> str:
        text = self.text_check(text)

        system = "You are an expert in reviewing for consistency given a text"
        prompt =  "Review the following text for consistency to re-write the following text as an article:" + text
        return self.generate_prompt(prompt, system, temperature=0.5)


    def save(self,text):
        with open(self.filename, "w") as f:
            f.write(text)
        print(f"Saved to {self.filename}")
        return text

if __name__ == '__main__':
    with open("../article.txt") as f:
        text = f.read()

    load_dotenv()
    openai_key = os.environ.get("OPENAI_API_KEY")

    article = ArticleGenerator(text, openai_key, "../re_written_article.txt")
    result = article.automate()
    print(result)

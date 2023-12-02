'''
This baseline implements the direct assessment prompt by
Kocmi and Federmann, Large Language Models Are State-of-the-Art Evaluators of Translation Quality. ArXiv: 2302.14520
for open source LLMs with MT and summarization
'''

import guidance, torch

from model_dict import load_from_catalogue


class DirectAssessment:
    def __init__(self, model, tokenizer, **kwargs):
        self.model = guidance.llms.Transformers(
            model, tokenizer=tokenizer, trust_remote_code=True, **kwargs
        )

    def set_model(self, model):
        self.model = model
        guidance.llms.Transformers.cache.clear()


    def direct_assessment_mt_block(
        self,
        hyp,
        gt,
        prompt_placeholder="",
        response_placeholder="",
        source_lang="en",
        target_lang="de",
    ):
        return "\n".join(
            [
                prompt_placeholder,
                f"Score the following translation from {source_lang} to {target_lang} with respect to",
                "the source sentence on a continuous scale from 0 to 100, where a score of zero means",
                '"no meaning preserved" and score of one hundred means "perfect meaning and grammar".',
                f'{source_lang} source: "{gt}"',
                f'{target_lang} translation: "{hyp}"',
                response_placeholder,
                "Score: {{gen 'score' pattern='(100|[1-9]?[0-9])'}}",
            ]
        )

    def direct_assessment_summ_block(
        self,
        hyp,
        gt,
        prompt_placeholder="",
        response_placeholder="",
    ):
        return "\n".join(
            [
                prompt_placeholder,
                f"Score the summarization with respect to the summarized document",
                "on a continuous scale from 0 to 100, where a score of zero means",
                '"irrelevant, factually incorrect and not readable" and score of one hundred means',
                '"relevant, factually correct, good readability".',
                f'Source text: "{gt}"',
                f'Summary: "{hyp}"',
                response_placeholder,
                "Score: {{gen 'score' pattern='(100|[1-9]?[0-9])'}}",
            ]
        )

    def prompt_model(
        self,
        gt,
        hyp,
        mt = True,
        prompt_placeholder=None,
        response_placeholder=None,
        target_lang="German",
        source_lang="English",
        verbose=False
    ):
        if mt:
            prompt = self.direct_assessment_mt_block(
            gt=gt,
            hyp=hyp,
            response_placeholder=response_placeholder,
            prompt_placeholder=prompt_placeholder,
            target_lang=target_lang,
            source_lang=source_lang
        )
        else:
            prompt = self.direct_assessment_summ_block(
            gt=gt,
            hyp=hyp,
            response_placeholder=response_placeholder,
            prompt_placeholder=prompt_placeholder
        )

        if verbose:
            print(prompt)

        guidance_prompt = guidance(prompt, llm=self.model)
        res = guidance_prompt()

        torch.cuda.empty_cache()
        return res.text, res["score"]


if __name__ == "__main__":
    #modelname = "NousResearch/Nous-Hermes-13b"
    modelname = "TheBloke/guanaco-65B-GPTQ"
    #modelname = "TheBloke/WizardLM-13B-V1.1-GPTQ"
    model, tokenizer, u_prompt, a_prompt = load_from_catalogue(modelname)
    BPG = DirectAssessment(model=model, tokenizer=tokenizer)

    _, score = BPG.prompt_model(
        gt="I have a small cat",
        hyp="Ich habe eine große Katze",
        prompt_placeholder=u_prompt,
        response_placeholder=a_prompt
    )

    print(score)

    _, score = BPG.prompt_model(
        gt="I like to eat fish. Therefore, I like to go to the restaurant. There, I often eat snails, which I like to eat, too",
        hyp="I like to eat fish and snails at the restaurant",
        mt=False,
        prompt_placeholder=u_prompt,
        response_placeholder=a_prompt
    )

    print(score)
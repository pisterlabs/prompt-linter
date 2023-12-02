import openai
from typing import Optional

from meeting_summarizer.utils import add_chunks_of_transcripts_to_prompt
from openai_api_interaction import OpenAICompletionAPI


def summarize_transcription(
        transcriptions: str,
        config: OpenAICompletionAPI,
        prompt_template: Optional[str] = None,
) -> str:
    """
    Summarizes the meeting transcription using OpenAI's GPT-4 model.

    Parameters
    ----------
    transcriptions : str
        The meeting transcription.
    config : OpenAICompletionAPI
        The configuration for the OpenAI Completion API.
    prompt_template : str, optional
        The template for creating the GPT-4 prompt, by default
                    "Please summarize the following meeting points:\n{points}\n".

    Returns
    -------
    str
        The generated meeting summary.
    """
    # Set up the OpenAI API client
    openai.api_key = config.api_key

    if prompt_template is None:
        prompt_template = "Write a concise summary of the following:" \
                          "\n\n<<<CHUNK>>>\n\n" \
                          "CONCISE SUMMARY:\n\n"

    # Create the prompts
    prompts = add_chunks_of_transcripts_to_prompt(
        transcriptions=transcriptions,
        model=config.model,
        prompt_template=prompt_template,
        num_token_completion=config.max_tokens
    )

    if len(prompts) < 20:
        response = openai.Completion.create(
            model=config.model,
            prompt=prompts,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            n=config.n,
            echo=config.echo,
            presence_penalty=config.presence_penalty,
            frequency_penalty=config.frequency_penalty,
            best_of=config.best_of,
        )
        summary = [choice["text"].strip() for choice in response.choices]
        summary = "".join(summary)
        return summary

    else:
        responses = []
        for i in range(0, len(prompts), 20):
            response = openai.Completion.create(
                model=config.model,
                prompt=prompts[i:i + 20],
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                n=config.n,
                echo=config.echo,
                stop=config.stop,
                presence_penalty=config.presence_penalty,
                frequency_penalty=config.frequency_penalty,
                best_of=config.best_of,
            )
            summary = [choice["text"].strip() for choice in response.choices]
            responses += summary
        summary = "".join(responses)
        return summary

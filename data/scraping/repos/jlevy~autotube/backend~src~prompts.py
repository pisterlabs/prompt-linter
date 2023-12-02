from anthropic import HUMAN_PROMPT, AI_PROMPT


def make_prompt(transcript) -> str:
    return f"""

    {HUMAN_PROMPT}

    The following is a transcript of a podcast.

    <transcript>
    {transcript}
    </transcript>

    Read the transcript and answer the following questions:

    1. What were the main high-level topics discussed in the podcast?
    2. For each logical topic discussed in the podcast, what are the key takeaways? Summarize as a short sentence or phrase.

    After thinking about, format your answers as XML, like so:

    ```xml
    <?xml version="1.0" encoding="UTF-8"?>
    <video>
        <topics>
            <topic>
                <content>A short word or phrase describing the topic</content>
                <instances>
                    <substring_index></substring_index>
            </topic>
            <topic>
                <content>Another short word or phrase, describing a different topic discussed</content>
                <instances>
                    <substring_index>A substring from the transcript to reference a section where the topic was discussed</substring_index>
                    <substring_index>Another substring from the transcript to reference a different section where the same topic was discussed</substring_index>
            </topic>
        </topics>

        <takeaways>
            <takeaway>
                <content>A short sentence summarizing the content discussed in this segment</content>
                <substring_index>A substring from the transcript to reference the section being summarized in this takeaway</substring_index>
            </takeaway>
            <takeaway>
                <content>Another short sentence summarizing the content discussed in this segment</content>
                <substring_index>A substring from the transcript to reference the section being summarized in this takeaway</substring_index>
            </takeaway>
        </takeaways>
    </video>
    ```

    **IMPORTANT:** Please return _only_ the correctly formatted XML, without any additional information or text.

    {AI_PROMPT}"""
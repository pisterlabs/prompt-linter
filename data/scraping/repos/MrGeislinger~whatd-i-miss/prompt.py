import anthropic

TRANSCRIPT_SEPERATOR = '###'

def transcript_prompt_piece(
    transcript: str,
    series_name: str | None = None,
    transcript_tag: str = 'transcript', 
    verbose: int = 0,
) -> str:
    episode_str = f'"{series_name}" episode' if series_name else 'episode'
    transcript_prompt = (
        f'{anthropic.HUMAN_PROMPT}:\n'
        f'Here is the full transcript of the {episode_str}:\n'
        f'<{transcript_tag}>\n'
        f'{transcript}\n'
        f'\n</{transcript_tag}>\n'
        f'\n'
        f'You are an expert at writing factual summaries.\n'
        '--------------------\n'
    )

    return transcript_prompt

def create_prompt(
    user_input: str,
    transcript: str,
    series_name: str | None = None,
    verbose: int = 0,
) -> str:
    transcript_piece = transcript_prompt_piece(transcript, series_name)
    prompt = (
        f'{transcript_piece}'
        f'Based on the transcript above, address the following:\n'
        f'"""\n{user_input}\n"""\n'
        'Carefully consider the key points from the transcript in relation to the statement/question above.'
        'Think carefully what snippets are related to the key points. '
        'Reference specific quotes/snippets from the transcript for evidence for each key point. '
        'The snippets do not have to be sequential and can occur anywhere in the transcript. '
        'Only snippets that are relevant to the key point should be considered as evidence. '
        'The more relevant quotes/snippets, the better. There should be at least two for each key point. '
        'Response should only be in a JSON with 3 to 5 key_points and overall_summary similar to this example from a different transcript:\n'
        '###Response###\n'
        '{\n'
        '''    "key_points": [\n'''
        '''       {\n'''
        '''           "text": "SPEAKER 1 does not like how the text on Kindle books is fully justified instead of left aligned. They find it aesthetically displeasing."\n'''
        '''           "evidence": [\n'''
        '''               "I don't know if it's just me but when text is fully justified I find myself reading more slowly by just how bad it looks.",\n'''
        '''               "Even Kindle's promotional material shows the books as left aligned; clearly they think it looks better too."\n'''
        '''           ]\n'''
        '''       }, \n'''
        '''       {\n'''
        '''           "text":"SPEAKER 2 says that the ability to left align text on older Kindle models but not the newer paper white model indicates that Amazon is being 'irrational or vindictive' in not including the option on newer Kindles."\n'''
        '''           "evidence": [\n'''
        '''               "The thing is that Amazon used to use left alignment on the old Kindles but for some reason the paper white model doesn't have that option.",\n'''
        '''               "It's too simple of something not to include on the new models so Amazon must irrational or vindictive. It's as simple as that.'"\n'''
        '''               "Now I sound like a crazy person wondering why they took out that feature because it only makes sense for them to just be laughing at us all having to look at this subpar format!"\n'''
        '''           ]\n'''
        '''       },\n'''
        '''       {\n'''
        '''           "text":"SPEAKER 1 really likes have the Kindle overall even with they minor concerns."\n'''
        '''           "evidence": [\n'''
        '''               "I understand your gripes with the Kindle but it is pretty amazing that we can use these pieces of technology",\n'''
        '''           ]\n'''
        '''       }\n'''
        '''    ],\n'''
        '''    "overall_summary": "SPEAKER 1 expresses frustration and disappointment with Amazon's decision to fully justify text on newer Kindle models. They feel that they could easily include an option to left align text but have chosen not to for unclear reasons. This decision makes reading on the Kindle an unpleasant experience for both SPEAKER 1 and SPEAKER 2."\n'''
        '''}\n\n'''
        '###End-of-Response###\n'
        'If the statement/question cannot be addressed with the given transcript, return an empty JSON object {}.\n'
        'Do not attempt to create a different output.\n'
        f'{anthropic.AI_PROMPT}: Here is the JSON below:\n'
    )
    if verbose:
        print(
            'Prompt:\n',
            f'{transcript_prompt_piece("FAKE_TRANSCRIPT", series_name)}'
            f'Based on the transcript above, address the following:\n'
            f'"""\n{user_input}\n"""\n'
            f'{anthropic.AI_PROMPT}:'
        )
    return prompt
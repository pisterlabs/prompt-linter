import loguru
from typing import List, Optional
from anthropic_api import AnthropicChatBot, ClaudeOptions
from message_templates import COMPRESSION_TEMPLATE, REVIEW_TEMPLATE

logger = loguru.logger

claude = AnthropicChatBot()


async def compression(
    transcripts: List[str],
    output_path: Optional[str],
    model: Optional[ClaudeOptions],
    max_tokens: Optional[int],
    stream: Optional[bool],
) -> str:
    if not transcripts:
        raise ValueError(f"{logger.error('No transcripts')}")
    if not model:
        model = ClaudeOptions.CLAUDE_2
    if not max_tokens:
        max_tokens = 90000
    if not stream:
        stream = True
    if not output_path:
        output_path = ".src/out/output.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(transcripts))

    comp_template = str(COMPRESSION_TEMPLATE)
    review_template = str(REVIEW_TEMPLATE)
    comp_template.join(transcripts)
    try:
        prompt = claude.converter.create_human_prompt(comp_template)
        compressed_transcripts = await claude.async_create(
            prompt, model, max_tokens, True
        )
        comp_transcript_string = str(compressed_transcripts)
        comp_transcript_string.join(review_template)
        review_prompt = claude.converter.create_human_prompt(comp_transcript_string)
        result = await claude.async_create(review_prompt, model, max_tokens, True)
        if result:
            result_string = "".join(str(result))
            logger.info(result_string)
            return result_string
    except BufferError as error:
        logger.exception(f"Caught BufferError with response body: {error.__cause__}")
    return "Error: No response"

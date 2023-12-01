import asyncio
import anthropic
import os


EOS_CHARACTERS = ["\n", ".", "!", "?", "\n•"]

TOKENIZER = anthropic.get_tokenizer()

async def stream_claude_sentences(client: anthropic.Client, prompt: str):
    """ Spit back individual sentences that Claude generates so we can start on the TTS ASAP. """

    processed_sentence_tokens = []  # Tokens of sentences we have sent away to be processed.
    
    response = await client.acompletion_stream(
        prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
        stop_sequences=[anthropic.HUMAN_PROMPT],
        max_tokens_to_sample=200,
        model="claude-v1",
        stream=True,
    )

    async for data in response:
        for eos_char in EOS_CHARACTERS:
            if data['completion'].endswith(eos_char):
                # We have a new sentence we can start TTS on.
                all_tokens = TOKENIZER.encode(data['completion']).ids
                # Strip away all the sentences we have already processed.
                new_sentence_tokens = all_tokens[len(processed_sentence_tokens):]
                
                new_sentence = TOKENIZER.decode(new_sentence_tokens).rstrip("•").strip()
                if new_sentence != "":
                    yield new_sentence

                processed_sentence_tokens.extend(new_sentence_tokens)

                break


async def main():
    client = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])
    async for sentence in stream_claude_sentences(client, "Tell me three jokes."):
        print(sentence)


if __name__ == "__main__":
    asyncio.run(main())
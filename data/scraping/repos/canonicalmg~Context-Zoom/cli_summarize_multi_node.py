import openai
from openai.error import RateLimitError
import time
import curses
import json
from transformers import GPT2Tokenizer

OPENAI_API_KEY=""
openai.api_key = OPENAI_API_KEY  

def gpt_call(prompt, temperature=0.9, max_retries=3):
    """
    Call GPT-3.5-turbo to generate a response based on the given prompt.

    Args:
        prompt (str): The input prompt for GPT-3.5-turbo.
        temperature (float, optional): Controls the randomness of the response. Defaults to 0.9.
        max_retries (int, optional): Number of retries in case of rate limit errors. Defaults to 3.

    Returns:
        str: The generated response from GPT-3.5-turbo.
    """
    # Prepare the messages for the GPT-3.5-turbo API call
    messages = [
        {"role": "system", "content": "You are summarizeGPT. Specializing in summarizing text."},
        {"role": "user", "content": prompt}
    ]
    retries = 0
    while retries < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=temperature,
                max_tokens=2000,
                n=1,
                # Do not stop the response generation at any specific token
                stop=None,
            )
            message_content = response.choices[0].message.content.strip()
            return message_content
        except RateLimitError as e:
            retries += 1
            # Wait for a short duration before retrying
            time.sleep(2)
    # Retry limit reached, return an empty string
    return ""

def summarize_into_blocks(text, num_blocks, block_size):
    prompt = f"""
    Please summarize this text into {block_size} words or less.


    {text}
    """
    response = gpt_call(prompt)
    return response


def chunk_text(text, max_tokens=1000):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk))
        
    return chunks

def count_tokens(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens = tokenizer.encode(text)
    return len(tokens)

def count_words(text):
    # Split the text into words and count them
    words = text.split()
    return len(words)


class TextBlock:
    def __init__(self, text, children=None):
        self.text = text
        self.child_blocks = children if children else []

    def summarize(self):
        num_words = count_words(self.text)
        print(f"Summarizing {self.text} of size {num_words}")
        # Stop summarizing if the text is 5 tokens or less
        if num_words <= 5:
            return
        # Summarize the text into a single block of half the size
        summary = summarize_into_blocks(self.text, 1, num_words // 2)
        if summary:
            # Take the summary as the child block
            summarized_block = TextBlock(summary)
            # Add the new child block to the list of children
            self.child_blocks.append(summarized_block)
            # Recursively summarize the child block
            summarized_block.summarize()


def recursive_summarization(text):
    root = TextBlock(text)

    chunks = chunk_text(text)
    for chunk in chunks:
        root.child_blocks.append(TextBlock(chunk))
    for child in root.child_blocks:
        child.summarize()
    return root.child_blocks
    # root_blocks = [TextBlock(chunk) for chunk in chunks]
    # for root_block in root_blocks:
    #     root_block.summarize()
    # return root_blocks


def get_test_data():
    root_blocks = [
        TextBlock(
            "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”",
            children=[
                TextBlock("Alice was tired sitting by her sister, with nothing to do.", children=[
                    TextBlock("Alice was tired and bored."),
                    TextBlock("She had nothing to do."),
                ]),
                TextBlock("She had looked at her sister's book, but found it uninteresting.", children=[
                    TextBlock("She peeped into her sister's book."),
                    TextBlock("The book had no pictures or conversations."),
                ]),
            ]
        ),
        TextBlock(
            "So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.",
            children=[
                TextBlock("Alice was considering making a daisy-chain, but was feeling too hot and sleepy.", children=[
                    TextBlock("It was a hot day."),
                    TextBlock("She was contemplating the effort of making a daisy-chain."),
                ]),
                TextBlock("Suddenly, a White Rabbit ran close by her.", children=[
                    TextBlock("A White Rabbit appeared suddenly."),
                    TextBlock("The rabbit ran close to Alice."),
                ]),
            ]
        ),
        TextBlock(
            "There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, “Oh dear! Oh dear! I shall be late!” (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.",
            children=[
                TextBlock("Alice didn't find it particularly strange when the Rabbit spoke and expressed concern about being late. Though she later realized she should have been more surprised, at the time it felt quite natural. The Rabbit then hurriedly took out a watch from its waistcoat-pocket.", children=[
                    TextBlock("Alice found it normal when the Rabbit expressed worry about being late.", children=[
                        TextBlock("Alice wasn't surprised by the Rabbit's worry."),
                        TextBlock("The Rabbit checked its waistcoat-pocket watch in haste.")
                    ]),
                    TextBlock("The Rabbit, in a hurry, checked a watch from its waistcoat-pocket.", children=[
                        TextBlock("Alice was intrigued by a uniquely accessorized rabbit."),
                        TextBlock("The rabbit's rush prompts Alice's pursuit.")
                    ]),
                ]),
                TextBlock("Seeing the Rabbit with a watch and a waistcoat-pocket sparked Alice's curiosity. She had never seen such a sight before. Compelled by her curiosity, Alice chased after the Rabbit across the field, and was fortunate enough to see it disappear down a large rabbit-hole under the hedge.", children=[
                    TextBlock("A rabbit with a waistcoat-pocket and a watch intrigued Alice, a sight she had never seen before", children=[
                        TextBlock("Alice was captivated by a rabbit with human accessories."),
                        TextBlock("Alice had never before seen such a rabbit.")
                    ]),
                    TextBlock("Driven by curiosity, Alice ran after the rabbit, witnessing it vanish down a large rabbit-hole under a hedge.", children=[
                        TextBlock("Alice's curiosity led her to chase the rabbit."),
                        TextBlock("She saw the rabbit disappear down a large hole.")
                    ]),
                ]),
            ]
        ),
        # Add more root blocks as needed...
    ]
    return root_blocks

def main(stdscr):
    # Turn off cursor blinking
    curses.curs_set(0)

    # # Color setup
    curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)

    text = """
Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so very remarkable in that; nor did Alice think it so very much out of the way to hear the Rabbit say to itself, “Oh dear! Oh dear! I shall be late!” (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually took a watch out of its waistcoat-pocket, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the world she was to get out again.

Suddenly she came upon a little three-legged table, all made of solid glass; there was nothing on it except a tiny golden key, and Alice’s first thought was that it might belong to one of the doors of the hall; but, alas! either the locks were too large, or the key was too small, but at any rate it would not open any of them. However, on the second time round, she came upon a low curtain she had not noticed before, and behind it was a little door about fifteen inches high: she tried the little golden key in the lock, and to her great delight it fitted!

Alice opened the door and found that it led into a small passage, not much larger than a rat-hole: she knelt down and looked along the passage into the loveliest garden you ever saw. How she longed to get out of that dark hall, and wander about among those beds of bright flowers and those cool fountains, but she could not even get her head through the doorway; “and even if my head would go through,” thought poor Alice, “it would be of very little use without my shoulders. Oh, how I wish I could shut up like a telescope! I think I could, if I only knew how to begin.” For, you see, so many out-of-the-way things had happened lately, that Alice had begun to think that very few things indeed were really impossible.

There seemed to be no use in waiting by the little door, so she went back to the table, half hoping she might find another key on it, or at any rate a book of rules for shutting people up like telescopes: this time she found a little bottle on it, (“which certainly was not here before,” said Alice,) and round the neck of the bottle was a paper label, with the words “DRINK ME,” beautifully printed on it in large letters.

It was all very well to say “Drink me,” but the wise little Alice was not going to do that in a hurry. “No, I’ll look first,” she said, “and see whether it’s marked ‘poison’ or not”; for she had read several nice little histories about children who had got burnt, and eaten up by wild beasts and other unpleasant things, all because they would not remember the simple rules their friends had taught them: such as, that a red-hot poker will burn you if you hold it too long; and that if you cut your finger very deeply with a knife, it usually bleeds; and she had never forgotten that, if you drink much from a bottle marked “poison,” it is almost certain to disagree with you, sooner or later.

However, this bottle was not marked “poison,” so Alice ventured to taste it, and finding it very nice, (it had, in fact, a sort of mixed flavour of cherry-tart, custard, pine-apple, roast turkey, toffee, and hot buttered toast,) she very soon finished it off.

What a curious feeling!” said Alice; “I must be shutting up like a telescope.”

And so it was indeed: she was now only ten inches high, and her face brightened up at the thought that she was now the right size for going through the little door into that lovely garden. First, however, she waited for a few minutes to see if she was going to shrink any further: she felt a little nervous about this; “for it might end, you know,” said Alice to herself, “in my going out altogether, like a candle. I wonder what I should be like then?” And she tried to fancy what the flame of a candle is like after the candle is blown out, for she could not remember ever having seen such a thing.

After a while, finding that nothing more happened, she decided on going into the garden at once; but, alas for poor Alice! when she got to the door, she found she had forgotten the little golden key, and when she went back to the table for it, she found she could not possibly reach it: she could see it quite plainly through the glass, and she tried her best to climb up one of the legs of the table, but it was too slippery; and when she had tired herself out with trying, the poor little thing sat down and cried.

“Come, there’s no use in crying like that!” said Alice to herself, rather sharply; “I advise you to leave off this minute!” She generally gave herself very good advice, (though she very seldom followed it), and sometimes she scolded herself so severely as to bring tears into her eyes; and once she remembered trying to box her own ears for having cheated herself in a game of croquet she was playing against herself, for this curious child was very fond of pretending to be two people. “But it’s no use now,” thought poor Alice, “to pretend to be two people! Why, there’s hardly enough of me left to make one respectable person!”

Soon her eye fell on a little glass box that was lying under the table: she opened it, and found in it a very small cake, on which the words “EAT ME” were beautifully marked in currants. “Well, I’ll eat it,” said Alice, “and if it makes me grow larger, I can reach the key; and if it makes me grow smaller, I can creep under the door; so either way I’ll get into the garden, and I don’t care which happens!”

She ate a little bit, and said anxiously to herself, “Which way? Which way?”, holding her hand on the top of her head to feel which way it was growing, and she was quite surprised to find that she remained the same size: to be sure, this generally happens when one eats cake, but Alice had got so much into the way of expecting nothing but out-of-the-way things to happen, that it seemed quite dull and stupid for life to go on in the common way.

So she set to work, and very soon finished off the cake.
    """.lstrip()
    root_blocks = recursive_summarization(text)

    current_blocks = [root_blocks]
    current_indices = [0]

    column_width = 45
    space_between_columns = 3
    scroll_offset = 0
    line_offset = [0] * 100  # initialize with a large enough number, this holds the line offset for each column
    top_line = [0]

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()  # get the height and width of the window

        for col, blocks in enumerate(current_blocks):
            adjusted_col = col - scroll_offset
            if adjusted_col < 0:
                continue

            line_no = 0
            for row, block in enumerate(blocks):
                if row == current_indices[col]:
                    stdscr.attron(curses.color_pair(1))

                block_lines = block.text.splitlines()
                for i in range(line_offset[col], min(line_offset[col] + height, len(block_lines))):
                    line = block_lines[i]
                    for j in range(0, len(line), column_width):
                        if line_no < height:  # don't print lines beyond the window's height
                            stdscr.addstr(line_no, adjusted_col * (column_width + space_between_columns), line[j:j+column_width])
                            line_no += 1
                        else:
                            break

                line_no += 1
                stdscr.attroff(curses.color_pair(1))

        k = stdscr.getch()

        # Navigate between blocks with up and down keys
        if k == curses.KEY_UP:
            current_indices[-1] = max(0, current_indices[-1] - 1)
            if current_indices[-1] < line_offset[-1]:
                line_offset[-1] = max(0, line_offset[-1] - 1)
        elif k == curses.KEY_DOWN:
            current_indices[-1] = min(len(current_blocks[-1]) - 1, current_indices[-1] + 1)
            if current_indices[-1] - line_offset[-1] > height - 2:
                line_offset[-1] = min(len(current_blocks[-1]) - height + 1, line_offset[-1] + 1)
        elif k == curses.KEY_RIGHT:
            if current_blocks[-1][current_indices[-1]].child_blocks:
                current_blocks.append(current_blocks[-1][current_indices[-1]].child_blocks)
                current_indices.append(0)
                line_offset.append(0)
                if len(current_blocks) - scroll_offset > width // (column_width + space_between_columns):
                    scroll_offset += 1
        elif k == curses.KEY_LEFT:
            if len(current_blocks) > 1:
                current_blocks.pop()
                current_indices.pop()
                line_offset.pop()
                if scroll_offset > 0:
                    scroll_offset -= 1

if __name__ == "__main__":
    curses.wrapper(main)



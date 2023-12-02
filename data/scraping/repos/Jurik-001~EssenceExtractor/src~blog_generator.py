"""Generate a blog post from a text file."""

import os

from openai import OpenAI

from src import utils

OUTPUT_TOKEN_LENGTH_BUFFER = 1500


class BlogGenerator:
    """Generate a blog post from a text file."""

    def __init__(
            self,
            output_path="blogs",
            model_name=utils.DEFAULT_MODEL_NAME,
            cost_manager=None,
    ):
        self.model_name = model_name
        self.output_path = output_path
        self.client = OpenAI()
        self.token_counter = utils.TokenCounter(self.model_name)
        self.cost_manager = cost_manager
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def _read_text_file(self, file_path):
        """Read a text file.

        Args:
            file_path (str): The path to the text file.

        Returns:
            str: The text in the file.
        """
        with open(file_path, "r") as f:
            text = f.read()
        return text

    def _generate_answer(self, system_prompt, user_prompt):
        """Generate an answer from a prompt.

        Args:
            system_prompt (str): The system prompt.
            user_prompt (str): The user prompt.

        Returns:
            str: The generated answer.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def _split_into_first_chunk(self, text, chunk_size, separator=" "):
        """Split text into the first chunk of a given size.

        Args:
            text (str): The text to split.
            chunk_size (int): The size of the chunk.
            separator (str, optional): The separator to use. Defaults to " ".

        Returns:
            str: The first chunk of the given size.
        """
        words = text.split(separator)
        current_chunk = []
        current_count = 0

        for word in words:
            current_chunk.append(word)
            current_count = self.token_counter.count_tokens(separator.join(current_chunk))

            if current_count >= chunk_size:
                return separator.join(current_chunk)

        return separator.join(current_chunk)

    def _create_refine_prompt(self, existing_answer, text):
        """Create a prompt for refining an existing answer.

        Args:
            existing_answer (str): The existing answer.
            text (str): The text to refine the answer with.

        Returns:
            str: The prompt for refining the answer.
        """
        return (
            f"We have provided an existing blog article "
            f"up to a certain point: {existing_answer}\n"
            "We have the opportunity to refine the existing blog article"
            "(only if needed) with some more context below.\n"
            "------------\n"
            f"{text}\n"
            "------------\n"
            "Given the new context, refine the original blog article"
            "If the context isn't useful, return the original blog article."
        )

    def generate_article_content(self, text_file_path):
        """Generate a blog post from a text file.

        Args:
            text_file_path (str): The path to the text file.

        Returns:
            str: The path to the generated blog post.
        """
        input_text = self._read_text_file(text_file_path)
        system_message = ("Your role is creating a finalized version "
                          "of a ready to publish article based on given transcript. "
                          "Write the article with a focus on educating the reader and"
                          "a captivating introduction, body, and a concise conclusion, "
                          "use markdown, than "
                          "place each timestamp, formatted as [MM:SS], to "
                          "the end of its relevant section using [MM:SS - MM:SS]."
                          ".\n")
        system_msg_length = self.token_counter.count_tokens(system_message)

        user_msg_length = self.token_counter.count_tokens(
            self._create_refine_prompt("", ""),
        )
        chunk_size = (
                self.token_counter.model_token_length -
                system_msg_length -
                user_msg_length -
                OUTPUT_TOKEN_LENGTH_BUFFER
        )

        blog_post = ""
        while input_text:
            chunk = self._split_into_first_chunk(input_text, chunk_size)
            user_message = self._create_refine_prompt(blog_post, chunk)

            if self.cost_manager:
                self.cost_manager.calculate_cost_text(user_message, is_input=True)

            blog_post = self._generate_answer(system_message, user_message)
            blog_post_length = self.token_counter.count_tokens(blog_post)
            if self.cost_manager:
                self.cost_manager.calculate_cost_token(blog_post_length, is_input=False)

            chunk_size = (
                    self.token_counter.model_token_length -
                    system_msg_length -
                    user_msg_length -
                    blog_post_length -
                    OUTPUT_TOKEN_LENGTH_BUFFER
            )

            input_text = input_text.replace(chunk, "")

        return blog_post

    def add_image_placeholder(self, blog_content):
        """Adds image placeholder to the blog content.

        Args:
            blog_content (str): The blog content.

        Returns:
            str: The blog content with image placeholders.
        """
        system_message = ("Your role is to NOT changing the following article "
                          "and if it adds value place images before or after "
                          "the section using ![...](path_to_image) "
                          "with a meaningful alt text.\n")
        blog_content = self._generate_answer(system_message, blog_content)
        return blog_content
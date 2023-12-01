from itertools import islice
import os
import zlib

import openai
import tiktoken

openai.api_key = "REPLACE ME"


class Generator:
    LOGPROBS = 5

    def __init__(self, model, window, attempts, tokens_per_query):
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.window = window
        self.attempts = attempts
        self.tokens_per_query = tokens_per_query
        self.clear()

    def clear(self):
        self.prompt = []
        self._next_choices = []

    def add_token(self, token):
        self.prompt.append(token)
        if (
            self._next_choices
            and self._next_choices[0]
            and self._next_choices[0][0] == token
        ):
            self._next_choices = self._next_choices[1:]
        else:
            self._next_choices = []

    def encode_tokens(self, texts):
        tokens = []
        for text in texts:
            try:
                tokens.append(self.encoding.encode_single_token(text))
            except KeyError:
                pass
        return tokens

    def _query(self, tokens_per_query, logit_bias):
        choices = openai.Completion.create(
            model=self.model,
            prompt=self.prompt[-self.window :] or "",
            max_tokens=tokens_per_query,
            top_p=0,
            logprobs=Generator.LOGPROBS,
            logit_bias=logit_bias,
        ).choices
        if not choices:
            self._next_choices = [[]]
        else:
            self._next_choices = [
                self.encode_tokens(
                    sorted(logprobs, key=lambda token: logprobs[token], reverse=True)
                )
                for logprobs in choices[0].logprobs.top_logprobs
            ]

    def generate(self, tokens_per_query=None):
        if tokens_per_query is None:
            tokens_per_query = self.tokens_per_query
        else:
            tokens_per_query = min(tokens_per_query, self.tokens_per_query)
        logit_bias = {self.encoding.encode_single_token("<|endoftext|>"): -100}
        for i in range(self.attempts):
            if not self._next_choices:
                self._query(tokens_per_query, logit_bias)
            ans = self._next_choices[0]
            yield from ans
            if len(ans) < Generator.LOGPROBS:
                return
            self._next_choices = []
            tokens_per_query = 1
            for token in ans:
                logit_bias[token] = -100


class Codec:
    def __init__(self, generator, zip_level=0):
        self.generator = generator
        self.zip_level = zip_level

    def compress(self, text: str) -> tuple[bytes, bytes]:
        result = self.generator.encoding.encode_ordinary(text)
        result = self._compress_tokens(result)
        result = self._to_7bit(result)
        if self.zip_level:
            return result, zlib.compress(result, self.zip_level)
        else:
            return result, result

    def decompress(self, compressed_blob: bytes, medium_blob=None) -> str:
        result = compressed_blob
        if self.zip_level:
            result = zlib.decompress(result)
        if medium_blob is not None:
            assert medium_blob == result
        result = self._from_7bit(result)
        result = self._decompress_tokens(result)
        result = self.generator.encoding.decode(result)
        return result

    def _compress_tokens(self, tokens):
        result = []
        self.generator.clear()
        while tokens:
            completion_tokens = self.generator.generate(len(tokens))
            for index, token in enumerate(completion_tokens):
                if tokens[0] == token:
                    result.append(index)
                    break
            else:
                result.append(tokens[0] + Generator.LOGPROBS * self.generator.attempts)
            self.generator.add_token(tokens[0])
            tokens = tokens[1:]
        return result

    @staticmethod
    def _starting_zeros(lst):
        for i in range(len(lst)):
            if lst[i]:
                return i
        return len(lst)

    def _decompress_tokens(self, nums):
        self.generator.clear()
        while nums:
            if nums[0] >= Generator.LOGPROBS * self.generator.attempts:
                self.generator.add_token(
                    nums[0] - Generator.LOGPROBS * self.generator.attempts
                )
            else:
                tokens = self.generator.generate(
                    min(len(nums), Codec._starting_zeros(nums) + 1)
                )
                token = next(islice(tokens, nums[0], None))
                self.generator.add_token(token)
            nums = nums[1:]
        return self.generator.prompt

    def _to_7bit(self, nums: list[int]) -> bytes:
        buf = []
        for num in nums:
            while True:
                byte = num & 127
                num >>= 7
                if num:
                    byte |= 128
                buf.append(byte)
                if not num:
                    break
        return bytes(buf)

    def _from_7bit(self, buf: bytes) -> list[int]:
        nums = []
        num = 0
        ind = 0
        for byte in buf:
            num |= (byte & 127) << (ind * 7)
            ind += 1
            if not (byte & 128):
                nums.append(num)
                num = 0
                ind = 0
        assert num == 0
        return nums


def parse_extension(compressed_name: str):
    zip_level = 0
    for zl in range(1, 10):
        ext = f".zip{zl}"
        if compressed_name.endswith(ext):
            zip_level = zl
            compressed_name = compressed_name.removesuffix(ext)
            break
    raw_name, ext = compressed_name.rsplit(".", 1)
    model, window, attempts, tokens_per_query = ext.split(",")
    assert window.startswith("w")
    window = int(window[1:])
    assert attempts.startswith("a")
    attempts = int(attempts[1:])
    assert tokens_per_query.startswith("t")
    tokens_per_query = int(tokens_per_query[1:])
    return (
        Codec(Generator(model, window, attempts, tokens_per_query), zip_level),
        raw_name,
        compressed_name,
    )


def check_write(file_name, contents):
    if isinstance(contents, bytes):
        mode = "b"
        encoding = None
    else:
        mode = ""
        encoding = "utf-8"
    if os.path.exists(file_name):
        with open(file_name, "r" + mode, encoding=encoding) as file:
            existing_contents = file.read()
            assert existing_contents == contents, (
                f"File '{file_name}' contents do not match provided contents\n"
                f"expected: {contents!r}\n"
                f"found   : {existing_contents!r}"
            )
    else:
        with open(file_name, "w" + mode, encoding=encoding) as file:
            file.write(contents)

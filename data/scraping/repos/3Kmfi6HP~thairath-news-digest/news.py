import json
import logging
import os
import re
import time
from enum import Enum

import openai
from slugify import slugify
from summarizer import Summarizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import config
from thairath_news import summary_cache, translation
from thairath_news_page import parser_factory, get_news_detail

logger = logging.getLogger(__name__)

# google t5 transformer
model, tokenizer, bert_model = None, None, None
if not config.disable_transformer:
    MAX_TOKEN = 4096
    # github runner only has 7 GB of RAM, https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners
    MODEL_NAME = config.transformer_model
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=MAX_TOKEN)
    bert_model = Summarizer()


class SummaryModel(Enum):
    PREFIX = 'Prefix'
    FULL = 'Full'
    EMBED = 'Embed'
    OPENAI = 'OpenAI'
    TRANSFORMER = 'GoogleT5'


class News:

    def __init__(self, rank=-1, title='', url='', comhead='', score='', author='',
                 author_link='', submit_time='', comment_cnt='', comment_url='', full_path=''):
        self.rank = rank
        self.title = title.strip()
        self.url = url
        self.comhead = comhead
        self.score = score
        self.author = author
        self.author_link = author_link
        self.submit_time = submit_time
        self.comment_cnt = comment_cnt
        self.comment_url = comment_url
        self.content = ''
        self.summary = ''
        self.summarized_by = SummaryModel.FULL
        self.favicon = ''
        self.image = None
        self.img_id = None
        self.full_path = full_path

    def get_image_url(self):
        if self.image and self.image.url:
            detail = get_news_detail(self.full_path)
            self.image.url = detail["thumbnail"]["image"]
            return self.image.url
        return ''

    def pull_content(self):
        try:
            logger.info("#%d, fetching %s", self.rank, self.url)
            parser = parser_factory(self.url)
            # if not self.title:
                # self.title = parser.title.strip()
            self.favicon = parser.get_favicon_url()
            # Replace consecutive spaces with a single space
            # self.content = re.sub(r'\s+', ' ', parser.get_content(config.max_content_size))
            self.content = re.sub(r'\s+', ' ', get_news_detail(self.full_path)["content"])
            # remove any html tag in content
            self.content = re.sub(r'<[^>]+>', '', self.content)
            self.summary = self.summarize()
            self.image = self.get_image_url()
            tm = parser.get_illustration()
            if tm:
                fname = tm.uniq_name()
                tm.save(os.path.join(config.output_dir, "image", fname))
                self.image = tm
                self.img_id = fname
        except Exception as e:
            logger.exception('Failed to fetch %s, %s', self.url, e)
        if not self.summary:  # last resort, in case remote server is down
            self.summary = summary_cache.get(self.url)

    def get_score(self):
        if isinstance(self.score, int):
            return self.score
        try:
            return int(self.score.strip())
        except:
            return 0

    def slug(self):
        return slugify(self.title or 'no title')

    def summarize(self):
        if not self.content:
            return ''
        if self.content.startswith('<iframe '):
            self.summarized_by = SummaryModel.EMBED
            return self.content
        if len(self.content) <= config.summary_size:
            logger.info(
                f'No need to summarize since we have a small text of size {len(self.content)}')
            return self.content

        summary = self.summarize_by_openai(self.content.strip())
        if summary:
            self.summarized_by = SummaryModel.OPENAI
            return summary
        summary = self.summarize_by_transformer(self.content.strip())
        if summary:
            self.summarized_by = SummaryModel.TRANSFORMER
            return summary
        else:
            self.summarized_by = SummaryModel.PREFIX
            return self.content

    def summarize_by_openai(self, content):
        summary = summary_cache.get(self.url, SummaryModel.OPENAI)
        if summary:
            logger.info("Cache hit for %s", self.url)
            return summary
        if not openai.api_key:
            logger.info("OpenAI API key is not set")
            return ''
        if self.get_score() <= config.openai_score_threshold:  # Avoid expensive openai
            logger.info("Score %d is too small, ignore openai", self.get_score())
            return ''

        if len(content) > 4096 * 2:
            # one token generally corresponds to ~4 characters, from https://platform.openai.com/tokenizer
            content = content[:4096 * 2]

        content = content.replace('```', ' ').strip()  # in case of prompt injection
        title = self.title.replace('"', "'").replace('\n', ' ').strip() or 'no title'
        start_time = time.time()
        # Hope one day this model will be clever enough to output correct json
        # Note: sentence should end with ".", "third person" - https://news.ycombinator.com/item?id=36262670
        prompt = f'Output only answers to following 3 steps, prefix each answer with step number.\n' \
                 f'1 - Summarize the article delimited by triple backticks in 2 sentences and in the third person.\n' \
                 f'2 - Translate the summary into Chinese.\n' \
                 f'3 - Provide a Chinese translation of sentence: "{title}".\n' \
                 f'```{content.strip(".")}.```'
        kwargs = {'model': config.openai_model,
                  # one token generally corresponds to ~4 characters
                  # 'max_tokens': int(config.summary_size / 4),
                  'stream': False,
                  'temperature': 0,
                  'n': 1,  # only one choice
                  'timeout': 30}
        try:
            if config.openai_model.startswith('text-'):
                resp = openai.Completion.create(
                    prompt=prompt,
                    **kwargs
                )
                answer = resp['choices'][0]['text'].strip()
            else:
                resp = openai.ChatCompletion.create(
                    messages=[
                        {'role': 'user', 'content': prompt},
                    ],
                    **kwargs)
                answer = resp['choices'][0]['message']['content'].strip()
            logger.info(f'prompt: {prompt}')
            logger.info(f'took {time.time() - start_time}s to generate: '
                        # Default str(resp) prints \u516c
                        f'{json.dumps(resp.to_dict_recursive(), sort_keys=True, indent=2, ensure_ascii=False)}')
            return self.parse_step_answer(answer).strip()
        except Exception as e:
            logger.warning('Failed to summarize using openai, %s', e)
            return ''

    def summarize_by_transformer(self, content):
        if config.disable_transformer:
            logger.warning("Transformer is disabled by env DISABLE_TRANSFORMER=1")
            return ''
        summary = summary_cache.get(self.url, SummaryModel.TRANSFORMER)
        if summary:
            logger.info("Cache hit for %s", self.url)
            return summary
        if self.get_score() <= 10:  # Avoid slow transformer
            logger.info("Score %d is too small, ignore transformer", self.get_score())
            return ''

        start_time = time.time()
        if len(content) > tokenizer.model_max_length:
            content = bert_model(content, use_first=True,
                                 ratio=tokenizer.model_max_length / len(content))
        tokens_input = tokenizer.encode("summarize: " + content, return_tensors='pt',
                                        max_length=tokenizer.model_max_length,
                                        truncation=True)
        summary_ids = model.generate(tokens_input, min_length=80,
                                     max_length=int(config.summary_size / 4),  # tokens
                                     length_penalty=20,
                                     no_repeat_ngram_size=2,
                                     temperature=0,
                                     num_beams=2)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True,
                                   clean_up_tokenization_spaces=True).capitalize()
        logger.info(f'took {time.time() - start_time}s to generate: {summary}')
        return summary

    def parse_step_answer(self, answer):
        lines = re.split(r'\n+', answer)
        # Hard to tolerate all kinds of formats, so just handle one
        pattern = r'^(\d+)\s*-\s*'
        for i, line in enumerate(lines):
            match = re.match(pattern, line)
            if not match:
                logger.warning(f'Answer line: {line} has no step number')
                return ''
            if str(i + 1) != match.group(1):
                logger.warning(f'Answer line {line} does not match step: {i + 1}')
                return ''
            lines[i] = re.sub(pattern, '', line)
        if len(lines) < 3:
            return lines[0]  # only get the summary
        translation.add(lines[0], lines[1], 'zh')
        translation.add(self.title, self.parse_title_translation(lines[2]), 'zh')
        return lines[0]

    def parse_title_translation(self, title):
        # Somehow, openai always return the original title
        title_cn = title.removesuffix('。').removesuffix('.')
        pattern = r'^"[^"]+"[^"]+“([^”]+)”'
        match = re.search(pattern, title_cn)
        if match:
            title_cn = match.group(1).strip()
            return title_cn.strip() # clean path

        parts = re.split(r'的中文翻译(?:为)?(?:：)?', title_cn, maxsplit=1)
        if len(parts) > 1 and parts[1].strip():
            title_cn = parts[1].strip().strip(':').strip('：').strip()
        else:
            title_cn = parts[0].strip()
        quote = ('"', '“', '”', '《', '》') # they are used interchangeably
        while title_cn and title_cn[0] in quote and title_cn[-1] in quote:
            title_cn = title_cn[1:-1].strip()
        return title_cn.removesuffix('。').removesuffix('.')
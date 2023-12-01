"""Implements tales generator on Cohere service."""

import logging
import re
import os
import cohere
import pandas as pd
from fastapi import HTTPException

from app.services.prompts import heroes as heroes_prompt
from app.services.prompts import plots as plots_prompt
from app.services.prompts import tale as tale_prompt
from app.services.prompts import title as title_prompt
from app.core.config import settings

logger = logging.getLogger('uvicorn')

API_KEY = settings.COHERE_KEY


class TalePrompt:
    """Generates prompts for tale structure and text."""

    HEAD = ('Examples of breakdowns of story into Vladimir Propp\'s '
            '"Morphology of a fairy tale" structutre.\n')

    SAMPLE = 'Example {counter}.\n{text}\n\n<scenes>\n{predict}\n\n<end>\n'
    SAMPLE_PREDICT = 'Example {counter}. \nSummary: {text}\n\n<scenes>'

    HEROES = ('Example {counter}.\n{text}\nCharacters and descriptions:'
              '\n{heroes}\n<end>')
    HEROES_PREDICT = 'Example {counter}.\n{text}\nCharacters and descriptions:'

    def __init__(self, line, **kwargs):
        self.client = None
        self.heroes = {}
        self.structures = {}
        self.line = line

    async def close(self):
        """Closes connection."""
        await self.client.close_connection()

    @classmethod
    async def create(cls, line, **kwargs):
        """Initializes async client."""
        self = cls(line, **kwargs)
        self.client = await cohere.AsyncClient.create(API_KEY)
        return self

    async def generate(
        self, prompt: str, model: str = 'xlarge',
        stop_sequences: list[str] = None, **kwargs
    ) -> str:
        """Generates sequence from given prompt."""
        max_tokens = kwargs.get('max_tokens', 500)
        params = {
            'return_likelihoods': 'GENERATION',
            'stop_sequences': stop_sequences or ['<end>'],
            'num_generations': kwargs.get('num_generations', 3),
            'temperature': kwargs.get('temperature', 0.555),
            'max_tokens': min([max_tokens, max(200, 8788 - len(prompt))]),
            'presence_penalty': 0.33,
        }
        logger.info(
            'GENERATING for len(prompt)=%s PARAMS:=%s\n', len(prompt), params)
        try:
            prediction = await self.client.generate(
                model=model, prompt=prompt, **params)
        except cohere.CohereError as error:
            await self.close()
            raise HTTPException(status_code=400, detail=str(error)) from error
        gens = []
        likelihoods = []
        for gen in prediction.generations:
            gens.append(gen.text)

            sum_likelihood = 0
            for token in gen.token_likelihoods:
                sum_likelihood += token.likelihood
            likelihoods.append(sum_likelihood)

        data = pd.DataFrame({'generation': gens, 'likelihood': likelihoods})
        data = data.drop_duplicates(subset=['generation'])
        data = data.sort_values('likelihood', ascending=False, ignore_index=True)
        return data

    def prompt_plots(self, text) -> str:
        """Generates prompt to get tale plots."""
        output = [self.HEAD]
        tales = [plots_prompt.RED_HOOD, plots_prompt.MERMAID]
        for counter, tale in enumerate(tales, 1):
            sample = self.SAMPLE.format(
                counter=counter, text=tale['summary'], predict=tale['plots'])
            output.append(sample)
        predict = self.SAMPLE_PREDICT.format(
            counter=len(tales) + 1, text=text)
        output.append(predict)
        return '\n'.join(output)

    def prompt_text_intro(self, structure: int, heroes: int) -> str:
        """Generates prompt to get tale full text."""
        parts = ''
        data = {
            'name': self.line,
            'heroes': '',
            'heroes_descriptions': '',
            'audience': 'children',
        }
        if heroes is not None:
            logger.info(self.heroes)
            data['heroes'] = ', '.join(self.heroes[heroes]['names'])
            data['heroes_desc'] = '\n'.join(self.heroes[heroes]['descriptions'])
        if structure is not None:
            parts = '\n'.join(self.structures[structure])
        return tale_prompt.INTRO.format(parts, **data)

    def prompt_heroes(self, text: str) -> str:
        """Builds prompt to get heroes descriptions."""
        output = []
        tales = [heroes_prompt.RED_HOOD, heroes_prompt.MERMAID]
        for counter, tale in enumerate(tales, 1):
            output.append(self.HEROES.format(
                counter=counter, text=tale['SUMMARY'], heroes=tale['HEROES']))
        predict = self.HEROES_PREDICT.format(counter=len(tales) + 1, text=text)
        output.append(predict)
        return '\n'.join(output)

    async def get_title(self, **kwargs):
        """Generates tale title."""
        prompt = title_prompt.TITLE.format(self.line)
        logger.info('Prompt Request:%s', prompt)
        result = await self.generate(prompt, **kwargs)
        for title in result['generation'].values:
            return title

    async def get_heroes(self, **kwargs):
        """Generates heroes names and descriptions."""
        prompt = self.prompt_heroes(f'Summary: {self.line}')
        logger.info('Prompt Request:%s', prompt)
        result = await self.generate(prompt, **kwargs)
        output = []
        for idx, gen in enumerate(result['generation'].values):
            heroes = []
            descriptions = re.findall(
                r'\<description\>\s(.*?)\s<stop>', gen, re.DOTALL)
            names = re.findall(
                r'\<character\>\s(.*?)\s<description>', gen, re.DOTALL)
            self.heroes[idx] = {'names': names, 'descriptions': descriptions}
            for hero_id, (name, description) in enumerate(zip(names, descriptions), start=1):
                if description:
                    heroes.append({
                        'id': hero_id,
                        'name': name,
                        'description': description})
            if heroes:
                output.append(heroes)
        return output

    async def get_structure(self, heroes: int = None, **kwargs):
        """Generates tale structure gor given heroes."""
        pattern = re.compile(r"\d\) (.*?)\s\((.*?)\)", re.DOTALL)
        text = [self.line]
        if heroes is not None:
            text.append('\n'.join(self.heroes[heroes]['descriptions']))
        prompt = self.prompt_plots('\n'.join(text))
        logger.info('Prompt Request:%s', prompt)
        output = await self.generate(prompt, **kwargs)
        for idx, gen in enumerate(output['generation'].values):
            matched = re.search(r'(1\).*?)\n\n', gen, re.DOTALL)
            if matched:
                parts = pattern.findall(matched.group())
                parts = [dict(id=part_id, name=name, text=text)
                    for part_id, (name, text) in enumerate(parts, start=1)]
                self.structures[idx] = parts
        return self.structures

    async def get_tale(
        self, structure: int = None, heroes: int = None, **kwargs):
        """Generates final tale text for given heroes and structure."""
        prompt = self.prompt_text_intro(structure, heroes)
        logger.info('Prompt Request:%s', prompt)
        if not prompt:
            return ''
        result = await self.generate(prompt, **kwargs)
        stories = []
        for idx, story in enumerate(result['generation'].values):
            stories.append(story.strip())
        return stories

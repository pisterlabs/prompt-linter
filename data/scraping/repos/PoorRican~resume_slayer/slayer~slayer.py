import asyncio
from langchain.schema import Document
from typing import List

from .history import generate_star_chain
from .summary import generate_snippets, generate_summary_chain
from .util import cut_sections, chunk_markdown, job_requirement_chain, format_resume_chain


class Slayer(object):
    summary: str
    experiences: List[Document]
    description: str
    title: str
    response: str

    def __init__(self, resume: str, description: str, title: str):
        super().__init__()

        sections = cut_sections(resume)
        self.summary = sections['summary']
        self.experiences = chunk_markdown(sections['history'])

        self.description = description
        self.title = title

    async def _process_star(self, requirements: List[str]) -> List[Document]:
        improve_summary = generate_star_chain()

        _experiences = self.experiences.copy()
        _tasks = [improve_summary.arun({"section": _experience.page_content,
                                        "requirements": requirements}) for _experience in _experiences]
        improved = await asyncio.gather(*_tasks)
        for _experience, new_text in zip(_experiences, improved):
            _experience.page_content = new_text
        return _experiences

    async def _process_summary(self, requirements: List[str]) -> str:
        snippets = await generate_snippets(self.experiences, requirements, self.description)
        return await generate_summary_chain().arun({'snippets': snippets, 'desc': self.description})

    async def process(self) -> str:
        # chains should be executed asynchronously

        requirements_chain = job_requirement_chain()
        requirements = requirements_chain.run({'desc'}).skills

        # improve resume sections
        overview, experiences = await asyncio.gather(self._process_summary(requirements),
                                                     self._process_star(requirements))

        result = ''
        result += '# Overview\n' + str(overview) + '\n'

        result += '# Experience\n'
        for experience in experiences:
            result += str(experience) + '\n'

        return format_resume_chain()({'section': result})['text']

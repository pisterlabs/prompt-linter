import asyncio
from dataclasses import dataclass
from pathlib import Path

from config import OPENAI_API_KEY
from domain.essay import Essay
from domain.essay_report import CompetencyReport, EssayReport
from langchain.chains.openai_functions import create_structured_output_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel, Field


@dataclass
class Competency:
    name: str
    path: Path


PROMPTS_FOLDER = Path(__file__).parent / "prompts"
COMPETENCIES = [
    Competency("Competência 1", PROMPTS_FOLDER / "competencia_1.txt"),
    Competency("Competência 2", PROMPTS_FOLDER / "competencia_2.txt"),
    Competency("Competência 3", PROMPTS_FOLDER / "competencia_3.txt"),
    Competency("Competência 4", PROMPTS_FOLDER / "competencia_4.txt"),
    Competency("Competência 5", PROMPTS_FOLDER / "competencia_5.txt"),
]


class GPTEssayReportCompetency(BaseModel):
    score: int = Field(..., description="A nota do aluno na competência")
    text: str = Field(
        ...,
        description="A descrição da nota do aluno na competência, incluindo o que ele fez de certo e errado. É necessário que o motivo pela subtração de pontos seja explicado.",
    )


async def get_grade_from_gpt_for_competency(
    essay: Essay, competency: Competency
) -> CompetencyReport:
    """Returns a report with the grade for a single competency.

    Uses the GPT-4 model to grade the essay with the prompt located in the apropriate file.

    Args:
        essay (Essay): The essay to be graded
        competency (Competency): The competency to be graded

    Returns:
        CompetencyReport: A report with the grade for the competency
    """
    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0.2,
        openai_api_key=OPENAI_API_KEY,
    )

    with open(competency.path, "r") as f:
        prompt_text = f.read()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_text),
            ("human", "Tema: {theme}\n\n{text}"),
        ]
    )

    chain = create_structured_output_chain(
        GPTEssayReportCompetency, llm, prompt, verbose=True
    )

    result: GPTEssayReportCompetency = await chain.arun(
        theme=essay.theme, text=essay.text
    )
    return CompetencyReport(
        competency_name=competency.name,
        score=result.score,
        text=result.text,
    )


async def get_grade_from_gpt(essay: Essay) -> EssayReport:
    """Returns a report with the grade for each competency.

    Uses the GPT-4 model to grade the essay with the prompts located in the prompts folder.

    Args:
        essay (Essay): The essay to be graded

    Returns:
        EssayReport: A report with the grade for each competency
    """
    competency_reports = await asyncio.gather(
        *[
            get_grade_from_gpt_for_competency(essay, competency)
            for competency in COMPETENCIES
        ]
    )

    return EssayReport(
        competency_reports=competency_reports,
    )

from anthropic import Anthropic
from dotenv import load_dotenv
from typarse import BaseParser
from haystack.nodes import PDFToTextConverter, PreProcessor
import os
import json

from prompts import CLAUDE_FIX, COMBINATORICS_PROBLEM, COMBINATORICS_MODEL, SPRING_PROBLEM, SPRING_MODEL
from tutor import get_tag_value

from tqdm import tqdm

load_dotenv()

class Parser(BaseParser):
    path: str
    save_dir: str = './extracted/combinatorics'
    fix: bool

    _abbrev = {
        'path': 'p',
        'save_dir': 's',
        'fix': 'f',
    }

anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_KEY"))

def claude_fix(text: str) -> str:
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=1000,
        prompt=CLAUDE_FIX % text,
        temperature=0
    )

    return completion.completion

def doc_parser():
    problems = []
    solutions = []
    problem_setmode = 0
    solution_mode = 0
    problem_statement = ''
    problem_solution = ''

    for each in docs_default:

        if 'Problem' in each.content:
            problem_setmode = 1
            solution_mode = 0
            solutions.append(problem_solution)
            problem_statement = ''
            problem_solution = ''

            problem_statement += each.content

        elif 'Solution' in each.content:
            solution_mode = 1
            problem_setmode = 0
            problems.append(problem_statement)
            problem_statement = ''
            problem_solution += each.content

        elif problem_setmode:
            problem_statement += each.content
        elif solution_mode:
            problem_solution += each.content
    return problems, solutions[1:]


if __name__ == "__main__":
    args = Parser()

    converter = PDFToTextConverter(remove_numeric_tables=True, valid_languages=["en"])
    doc_pdf = converter.convert(file_path=args.path, meta=None)[0]

    preprocessor = PreProcessor(
        split_by="sentence",
        split_length=1,
        split_respect_sentence_boundary=False)
    docs_default = preprocessor.process([doc_pdf])

    problems, solutions = doc_parser()

    os.makedirs(args.save_dir, exist_ok=True)

    num_problems = len(problems)


    all_data = {
        "Problem 1": {
            "problem": COMBINATORICS_PROBLEM,
            "solution": COMBINATORICS_MODEL,
        },
        "Problem 2": {
            "problem": SPRING_PROBLEM,
            "solution": SPRING_MODEL,
        },
    }

    for i, (problem, solution) in enumerate(tqdm(zip(problems, solutions), total=num_problems)):
        problem_name = f"Problem {len(all_data) + 1}"
        with open(os.path.join(args.save_dir, f'problem_{i}.txt'), 'w') as f:
            if args.fix:
                problem = claude_fix(problem)
                problem = get_tag_value(problem, "result")
            f.write(problem)
        with open(os.path.join(args.save_dir, f'solution_{i}.txt'), 'w') as f:
            if args.fix:
                solution = claude_fix(solution)
                solution = get_tag_value(solution, "result")
            f.write(solution.strip())
        all_data[problem_name] = {
            "problem": problem,
            "solution": solution,
        }
    
    with open(os.path.join(args.save_dir, "problem_base.json"), 'w') as f:
        json.dump(all_data, f)
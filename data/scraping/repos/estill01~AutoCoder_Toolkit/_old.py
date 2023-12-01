import openai
import dotenv
from typing import List
from autocoder_toolkit.utils import generate_task


# TODO Add RAIL output formatters (?)


# ========================
# META-FUNCTIONS                # TODO Move to dedicated file
# ========================


def analyze_code(code: str, analysis_type: str, temp: int = 0) -> List[str]:
    analysis_output = generate_task(
        task=f"{analysis_type} analysis",
        action=f"Analyze the following code and identify {analysis_type}:",
        user_input=code,
        temp=temp
    )
    return split_recommendations(analysis_output)


def make_plan(item: str, plan_type: str, temp: int = 0) -> str:
    return generate_task(
        task=f"{plan_type} planning",
        action=f"Create a plan to address the following {plan_type}:",
        user_input=item,
        temp=temp
    )

def implement_task(code: str, task: str, temp: int = 0) -> str:
    return generate_task(
        task="task implementation",
        action="Implement the following task on the code:",
        user_input=code,
        extra_info=task,
        temp=temp
    )

def implement_plan(code: str, plans: List[str], temp: int = 0) -> str:
    updated_code = code
    for plan in plans:
        updated_code = implement_task(updated_code, plan, temp)
    return updated_code


def generate_code(component_type: str, requirements: str, temp: int = 0) -> str:
    return generate_task(
        task=f"{component_type} generation",
        action=f"Generate {component_type} code based on the following requirements:",
        user_input=requirements,
        temp=temp
    )

def compare_code(code1: str, code2: str, comparison_type: str, temp: int = 0) -> str:
    return generate_task(
        task=f"{comparison_type} comparison",
        action=f"Compare the following two code snippets based on {comparison_type}:",
        user_input=f"Code 1:\n{code1}\n\nCode 2:\n{code2}",
        temp=temp
    )

def evaluate_technology(technology: str, evaluation_criteria: str, temp: int = 0) -> str:
    return generate_task(
        task=f"{technology} evaluation",
        action=f"Evaluate the following technology based on the given criteria:",
        user_input=evaluation_criteria,
        extra_info=technology,
        temp=temp
    )

# TODO Switch implementation to deterministic tech
def search_code(keyword: str, codebase: str, temp: int = 0) -> List[str]:
    search_results = generate_task(
        task="code search",
        action=f"Search the following codebase for occurrences of the keyword '{keyword}':",
        user_input=codebase,
        temp=temp
    )
    return split_recommendations(search_results)

def transform_code(code: str, transformation_type: str, temp: int = 0) -> str:
    return generate_task(
        task=f"{transformation_type} transformation",
        action=f"Apply {transformation_type} transformation to the following code:",
        user_input=code,
        temp=temp
    )



# ========================
# PIPELINES                     # TODO Move to dedicaed file
# ========================

def refactor_and_optimize_code(code: str, analysis_type: str, plan_type: str, temp: int = 0) -> str:
    identified_items = analyze_code(code, analysis_type, temp)
    updated_code = code
    for item in identified_items:
        plan = make_plan(item, plan_type, temp)
        updated_code = implement_task(updated_code, plan, temp)
    return updated_code


def split_recommendations(review_output: str) -> List[str]:
    recommendations = review_output.split('\n')
    return [rec.strip() for rec in recommendations if rec.strip()]



# ========================
# TASKS
# ========================

# ------------------------
# ENHANCE
# ------------------------

def review_code(code: str, temp: int = 0) -> str:
    return generate_task(
        task="code review",
        action="Review the following code and provide feedback and suggestions for improvement:",
        user_input=code,
        temp=temp
    )


# TODO Swap this to multiple calls
def update_code(code: str, recommendations: Union[str, List[str]], temp: int = 0) -> str:
    if isinstance(recommendations, list):
        recommendations = '\n'.join(recommendations)
    return generate_task(
        task="code update",
        action="Update or enhance the following code using the provided recommendations:",
        user_input=code,
        extra_info=recommendations,
        temp=temp
    )

# ------------------------
# REFACTOR
# ------------------------

def refactor_code(code: str, reason: str, temp: int = 0) -> str:
    return generate_task(
        task="code refactoring",
        action="Refactor the following code:",
        user_input=code,
        extra_info=f"to {reason}.",
        temp=temp
    )

# ------------------------
# TESTING
# ------------------------

def generate_unit_tests(code: str, language: str, temp: int = 0) -> str:
    return generate_task(
        task="unit test generation",
        action="Generate unit tests for the following code:",
        user_input=code,
        extra_info=f"in {language}.",
        temp=temp
    )

# ------------------------
# BUGS
# ------------------------
def identify_bugs(code: str, temp: int = 0) -> str:
    return generate_task(
        task="bug detection",
        action="Identify any bugs in the following code and suggest potential fixes:",
        user_input=code,
        temp=temp
    )

def explain_bug_fixes(bug_report: str, temp: int = 0) -> str:
    return generate_task(
        task="bug fix explanation",
        action="Explain how to fix the bugs identified in the following report:",
        user_input=bug_report,
        temp=temp
    )

def fix_bugs(code: str, bug_fix_explanation: str, temp: int = 0) -> str:
    return generate_task(
        task="bug fixing",
        action="Fix the bugs in the following code using the provided bug fix explanation:",
        user_input=code,
        extra_info=bug_fix_explanation,
        temp=temp
    )

# -> check work

# ------------------------
# IMPLEMENT
# ------------------------

def implement_feature(feature_description: str, language: str, temp: int = 0) -> str:
    return generate_task(
        task="feature implementation",
        action="Implement the feature described in the user input:",
        user_input=feature_description,
        extra_info=f"in {language} code.",
        temp=temp
    )


# ------------------------
# CORRECTNESS CHECK
# ------------------------


def check_syntax(code: str, language: str, temp: int = 0) -> str:
    return generate_task(
        task="syntax check",
        action="Check the syntax of the following code and point out any errors:",
        user_input=code,
        extra_info=f"in {language}.",
        temp=temp
    )


# ------------------------
# TRANSLATE
# ------------------------

def translate_code(code: str, source_language: str, target_language: str, temp: int = 0) -> str:
    return generate_task(
        task="code translation",
        action="Translate the following code:",
        user_input=code,
        extra_info=f"from {source_language} to {target_language}.",
        temp=temp
    )


# ------------------------


# -> Extract import statements from file 
def find_imports(file_path, source_language):
    with open(file_path, "r") as f:
        code = f.read()
    content = llm(
        _code_prompt(
            f"Extract all import statements from the following {source_language} code. put each extracted import statement on a newline. Include the whole import statement including language syntax, imported files, classes, functions, and other code artifacts."
        ),
        code
    )
    # TODO Use an output formatter to get a list of imports
    imports = content.split('\n')
    return [import_statement for import_statement in imports if import_statement.strip()]



## REFACTOR -- PICK BACK UP HERE: CONVERTING TO `ChatCompletions` ##

def is_external_import(import_statement, source_language):
    prompt = f"Is the following {source_language} import statement an external package or a reference to another file in the project?\n\n{import_statement}\n\nAnswer:"

    response = openai.Completion.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=50,
        temperature=0.7,
    )

    return response.choices[0].text.strip().lower() == "external"

def find_alternative_imports(source_imports, source_language, target_language, browser_compatible=False):
    alternative_imports = []

    for source_import in source_imports:
        prompt = f"Find a suitable {target_language} package that can replace the {source_language} package '{source_import}'."
        if browser_compatible and target_language.lower() in ['javascript', 'typescript']:
            prompt += " The {target_language} package should be compatible with running in a browser environment."
        prompt += f"\n\n{source_language} import statement:\n" + source_import

        response = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            max_tokens=1024,
            temperature=0.7,
        )

        alternative_import = response.choices[0].text.strip()

        if not alternative_import:
            if browser_compatible and target_language.lower() in ['javascript', 'typescript']:
                raise ValueError(f"No browser-compatible {target_language} alternative found for the {source_language} import: {source_import}")
            else:
                raise ValueError(f"No {target_language} alternative found for the {source_language} import: {source_import}")

        alternative_imports.append(alternative_import)

    return alternative_imports

def process_directory(source_dir, output_dir, source_language, target_language, browser_compatible=False):
    source_extension = source_language.lower()[:2]  # Example: ".py" for Python
    target_extension = target_language.lower()[:2]  # Example: ".js" for JavaScript

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(f".{source_extension}"):
                file_path = os.path.join(root, file)
                output_file = os.path.join(output_dir, os.path.relpath(file_path, source_dir)).replace(f".{source_extension}", f".{target_extension}")

                os.makedirs(os.path.dirname(output_file), exist_ok=True)

                with open(file_path, "r") as f:
                    source_code = f.read()

                source_imports = find_imports(file_path, source_language)
                alternative_imports = find_alternative_imports(source_imports, source_language, target_language, browser_compatible)

                translated_code = translate_code(source_language, target_language, source_code)
                translated_code_with_imports = "\n".join(alternative_imports) + "\n\n" + translated_code
                with open(output_file, "w") as f:
                    f.write(translated_code_with_imports)



def update_code(code, error_message, target_language):
    prompt = f"Update the following {target_language} code to fix the error described:\n\nError message:\n{error_message}\n\n{target_language} code:\n{code}\n\nUpdated {target_language} code:"

    response = openai.Completion.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=1024,
        temperature=0.7,
    )

    return response.choices[0].text.strip()


def fix_errors(file_path, error_output, target_language):
    with open(file_path, "r") as f:
        code = f.read()

    fixed_code = update_code(code, error_output, target_language)

    with open(file_path, "w") as f:
        f.write(fixed_code)


def execute_code(file_path):
    # This function should be implemented based on the specific target language and execution environment
    # TODO make sure to collect any error output and feed that into `fix_errors`, `refactor_code`, and `update_code`
    pass


def refactor_code(file_path, target_language):
    with open(file_path, "r") as f:
        code = f.read()

    prompt = f"Refactor the following {target_language} code to ensure a high level of encapsulation and abstraction, as if implemented by an expert software engineer:\n\n{target_language} code:\n{code}\n\nRefactored {target_language} code:"

    response = openai.Completion.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=1024,
        temperature=0.7,
    )

    refactored_code = response.choices[0].text.strip()

    with open(file_path, "w") as f:
        f.write(refactored_code)

# TODO This should be in a prompt post-processor ; extract to a prompt/ or utils/ dir
def remove_extraneous_text(file_path, target_language):
    with open(file_path, "r") as f:
        code = f.read()

    prompt = f"Remove any extraneous text from the following {target_language} code, leaving only code and relevant comments:\n\n{target_language} code:\n{code}\n\nCleaned {target_language} code:"

    response = openai.Completion.create(
        model="gpt-4",
        prompt=prompt,
        max_tokens=1024,
        temperature=0.7,
    )

    cleaned_code = response.choices[0].text.strip()

    with open(file_path, "w") as f:
        f.write(cleaned_code)


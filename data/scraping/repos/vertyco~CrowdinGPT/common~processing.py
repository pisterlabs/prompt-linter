import json
import typing as t
from datetime import datetime
from time import sleep

import openai
from aiocache import cached
from openai.error import (
    APIConnectionError,
    APIError,
    RateLimitError,
    ServiceUnavailableError,
)

from common.constants import PRICES, TRANSLATE, cyan, green, red, yellow
from common.crowdin_api import CrowdinAPI
from common.models import QA, Language, Project, String, Translation
from common.translate_api import TranslateManager

from . import (
    AUTO,
    BACKTICK_MISMATCH,
    CROWDIN_KEY,
    DEEPL_KEY,
    ENDPOINT_OVERRIDE,
    MODEL,
    OPENAI_KEY,
    PLACEHOLDER_MISMATCH,
    PRE_TRANSLATE,
    PROCESS_QA,
    messages_dir,
    processed_json,
    processed_qa_json,
    system_prompt_path,
    tokens_json,
)

ADDON = "\nRevise your translation and return only the updated version"


def static_processing(source: str, dest: str) -> str:
    """Help GPT a bit with some common static fixes"""
    # Maintain ending punctuation from source
    if source.endswith(".") and not dest.endswith("."):
        dest += "."
    elif not source.endswith(".") and dest.endswith("."):
        dest = dest.rstrip(".")
    if source.endswith("!") and not dest.endswith("!"):
        dest += "!"

    # Maintain placeholder at the start of text
    if source.startswith("{}\n") and not dest.startswith("{}\n"):
        dest = "{}\n" + dest

    # Count the trailing newlines in both source and destination
    source_trailing_newlines = len(source) - len(source.rstrip("\n"))
    dest_trailing_newlines = len(dest) - len(dest.rstrip("\n"))

    # Balance the count of newlines in the destination
    if source_trailing_newlines != dest_trailing_newlines:
        dest = dest.rstrip("\n")  # Remove all trailing newlines
        dest += "\n" * source_trailing_newlines  # Add the correct number of newlines

    # Maintain space characters at the start and end of text
    for idx in range(20, 1, -1):
        space_seq = " " * idx

        if source.endswith(space_seq) and not dest.endswith(space_seq):
            dest += space_seq
        if source.startswith(space_seq) and not dest.startswith(space_seq):
            dest = space_seq + dest

    return dest


def update_tokens(response: dict):
    usage = json.loads(tokens_json.read_text())
    usage["total"] += response["usage"].get("total_tokens", 0)
    usage["prompt"] += response["usage"].get("prompt_tokens", 0)
    usage["completion"] += response["usage"].get("completion_tokens", 0)
    tokens_json.write_text(json.dumps(usage))


def get_cost() -> float:
    usage = json.loads(tokens_json.read_text())
    input_price, output_price = PRICES[MODEL]
    input_cost = (usage["prompt"] / 1000) * input_price
    output_cost = (usage["completion"] / 1000) * output_price
    return round(input_cost + output_cost, 3)


@cached(ttl=120)
async def call_openai(
    messages: t.List[dict],
    use_functions: bool,
    temperature: float = 0.0,
    presence_penalty: float = -0.3,
    frequency_penalty: float = -0.3,
):
    kwargs = {
        "api_key": OPENAI_KEY,
        "api_base": ENDPOINT_OVERRIDE,
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
    }
    if use_functions:
        kwargs["functions"] = [TRANSLATE]
    return await openai.ChatCompletion.acreate(**kwargs)


async def process_translations():
    processed = json.loads(processed_json.read_text())
    processed_qa = json.loads(processed_qa_json.read_text())

    client = CrowdinAPI(api_key=CROWDIN_KEY)
    projects = await client.get_projects()
    if not projects:
        print(red("There are no projects to process!!!"))
        return
    for project in projects:
        strings = await client.get_strings(project.id)
        issues = await client.get_qa_issues(project.id) if PROCESS_QA else []
        mapped_strings = {string.id: string for string in strings}
        mapped_langs = {lang.id: lang for lang in project.targetLanguages}
        if PROCESS_QA:
            for issue in issues:
                key = f"{project.id}-{issue.id}"
                string = mapped_strings.get(issue.stringId)
                if not string:
                    processed_qa.append(key)
                    processed_qa_json.write_text(json.dumps(processed_qa))
                    print(yellow(f"Added {key} to processed QA for no key"))
                    continue
                translation = await client.get_translation(project.id, string.id, issue.languageId)
                if not translation:
                    processed_qa.append(key)
                    processed_qa_json.write_text(json.dumps(processed_qa))
                    print(yellow(f"Added {key} to processed QA for no translation"))
                    continue
                lang = mapped_langs[issue.languageId]
                success = await process_revision(client, project, lang, string, translation)
                if not success:
                    continue
                processed_qa.append(key)
                processed_qa_json.write_text(json.dumps(processed_qa))
                cost = get_cost()
                print(f"{yellow('-')}-" * 22 + f" Usage: ${cost} " + f"{yellow('-')}-" * 22)
        else:
            print(yellow(f"Found {len(strings)} strings for project '{project.name}'"))
            for lang in project.targetLanguages:
                for string in strings:
                    key = f"{project.id}-{string.id}-{lang.id}"
                    if key in processed:
                        continue
                    if await client.get_translation(project.id, string.id, lang.id):
                        processed.append(key)
                        processed_json.write_text(json.dumps(processed))
                        print(yellow(f"Added {key} to processed"))
                        continue
                    print(cyan(f"Processing {key}"))
                    success = await process_translation(client, project, lang, string)
                    if not success:
                        continue
                    processed.append(key)
                    processed_json.write_text(json.dumps(processed))
                    cost = get_cost()
                    print(f"{yellow('-')}-" * 22 + f" Usage: ${cost} " + f"{yellow('-')}-" * 22)


async def process_revision(
    client: CrowdinAPI,
    project: Project,
    language: Language,
    string: String,
    translation: Translation,
    issue: QA,
):
    messages = [
        {"role": "user", "content": f"Translate the following text to {language.name}"},
        {"role": "user", "content": string.text},
        {"role": "assistant", "content": translation.text},
        {"role": "user", "content": issue.text + ADDON},
    ]

    corrections = 0
    success = False

    openai_fails = 0
    translation_fails = 0

    while True:
        if openai_fails > 2 or translation_fails > 3:
            print("Failed to revise, skipping")
            return
        try:
            response = await call_openai(messages, use_functions=False)
            update_tokens(response)
        except ServiceUnavailableError as e:
            openai_fails += 1
            print(red(f"ServiceUnavailableError, waiting 5 seconds before trying again: {e}"))
            sleep(5)
            print("Trying again...")
            continue
        except (APIConnectionError, APIError) as e:
            openai_fails += 1
            print(red(f"APIConnectionError/APIError, waiting 5 seconds before trying again: {e}"))
            sleep(5)
            print("Trying again...")
            continue
        except RateLimitError as e:
            openai_fails += 1
            print(red(f"Rate limted! Waiting 1 minute before retrying: {e}"))
            sleep(60)
            continue
        except Exception:
            openai_fails += 1
            print(red(f"EXCEPTION\n{json.dumps(messages, indent=2)}"))
            sleep(60)
            continue

        message = response["choices"][0]["message"]
        reply = message["content"]
        reply = static_processing(string.text, reply)

        print(yellow("Uploading..."))
        status, data = await client.upload_translation(project.id, string.id, language.id, reply)
        if status == 201:
            success = True
            print(green("Translation upload successful"))
            break

        print(red(f"Translation upload unsuccessful (status {status})"))
        if not data:
            print("Skipping")
            break
        errors = data.get("errors")
        if not errors:
            print("Skipping")
            break
        error = errors[0]["error"]["errors"][0]["message"]
        if "An identical translation" in error:
            print("Skipping, identical translation exists")
            break
        print(f"Upload Error: {red(error)}")
        messages.append({"role": "user", "content": error + ADDON})
        corrections += 1
        file = messages_dir / f"dump_{round(datetime.now().timestamp())}.json"
        file.write_text(json.dumps(messages, indent=4))

    files = sorted(messages_dir.iterdir(), key=lambda f: f.stat().st_mtime)
    for f in files[:-9]:
        f.unlink(missing_ok=True)

    return success


async def process_translation(
    client: CrowdinAPI, project: Project, language: Language, string: String
) -> bool:
    """Return True if successfully translated"""
    translator = TranslateManager(deepl_key=DEEPL_KEY)
    system_prompt_raw = system_prompt_path.read_text().strip()
    system_prompt = system_prompt_raw.replace("{target_language}", language.name)

    source_text = string.text
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "¿Hola, cómo estás?"},
        {"role": "user", "content": "{}\nCog Version: {}\nAuthor: {}"},
        {"role": "assistant", "content": "{}\nVersión de Cog: {}\nAutor: {}"},
        {"role": "user", "content": "Invalid schema!\n**Missing**\n{}"},
        {"role": "assistant", "content": "Geçersiz şema!\n**Eksik**\n{}"},
        {"role": "user", "content": source_text},
    ]

    if PRE_TRANSLATE:
        if translation := await translator.translate(source_text, language.name):
            if translation.text.strip() != source_text.strip():
                name = "get_translation"
                dump = json.dumps({"message": source_text, "to_language": language.name})
                call = {"name": name, "arguments": dump}
                messages.append({"role": "assistant", "content": None, "function_call": call})
                messages.append({"role": "function", "name": name, "content": translation.text})

    functions_called = 0
    corrections = 0
    use_functions = True
    success = False

    openai_fails = 0
    translation_fails = 0

    while True:
        if openai_fails > 2 or translation_fails > 3 or corrections > 4:
            print("Failed to translate, skipping")
            return
        try:
            if functions_called > 6:
                use_functions = False
            response = await call_openai(messages, use_functions)
            update_tokens(response)
        except ServiceUnavailableError as e:
            openai_fails += 1
            print(red(f"ServiceUnavailableError, waiting 5 seconds before trying again: {e}"))
            sleep(5)
            print("Trying again...")
            continue
        except (APIConnectionError, APIError) as e:
            openai_fails += 1
            print(red(f"APIConnectionError/APIError, waiting 5 seconds before trying again: {e}"))
            sleep(5)
            print("Trying again...")
            continue
        except RateLimitError as e:
            openai_fails += 1
            print(red(f"Rate limted! Waiting 1 minute before retrying: {e}"))
            sleep(60)
            continue
        except Exception:
            openai_fails += 1
            print(red(f"EXCEPTION\n{json.dumps(messages, indent=2)}"))
            sleep(60)
            continue

        message = response["choices"][0]["message"]

        reply: t.Optional[str] = message["content"]
        if reply:
            reply = reply.replace(r"\n", "\n")
            reply = static_processing(string.text, reply)
            message["content"] = reply
            messages.append(message)

            print()
            print(f"Called {functions_called} functions")
            print("-" * 45 + " Source " + "-" * 45)
            print(f"{cyan(string.text)}\n")
            print("-" * 45 + f" {language.name} " + "-" * 45)
            print(f"{green(reply)}\n")
            print("-" * 100)

            txt = (
                "Does this look okay?\n"
                "- Type 'y' to continue\n"
                "- Type 'n' or press ENTER to skip\n"
                "Enter your response: "
            )
            review = False
            if string.text.count("{") != reply.count("{"):
                print("Bracket mismatch")
                if corrections > 3:
                    review = True
                else:
                    messages.append({"role": "system", "content": PLACEHOLDER_MISMATCH})
                    corrections += 1
                    continue
            if string.text.count("`") != reply.count("`"):
                print("Backtick mismatch")
                if corrections > 3:
                    review = True
                else:
                    messages.append({"role": "system", "content": BACKTICK_MISMATCH})
                    corrections += 1
                    continue

            if not AUTO:
                review = True

            if review:
                if AUTO == 2:
                    print(red("Auto skipping..."))
                    break
                confirmation = input(yellow(txt))
                if "y" not in confirmation.lower():
                    print("Skipping...")
                    break

            print(yellow("Uploading..."))
            status, data = await client.upload_translation(
                project.id, string.id, language.id, reply
            )
            if status == 201:
                success = True
                print(green("Translation upload successful"))
                break

            print(red(f"Translation upload unsuccessful (status {status})"))
            if not data:
                print("Skipping")
                break
            errors = data.get("errors")
            if not errors:
                print("Skipping")
                break
            error = errors[0]["error"]["errors"][0]["message"]
            if "An identical translation" in error:
                print("Skipping, identical translation exists")
                break
            print(f"Upload Error: {red(error)}")
            messages.append({"role": "system", "content": error + ADDON})
            corrections += 1
            continue

        messages.append(message)
        function_call = message["function_call"]
        function_name = function_call["name"]

        if function_name not in ("get_translation",):
            messages.append(
                {"role": "system", "content": f"{function_name} is not a valid function"}
            )
            continue

        args = function_call.get("arguments", "{}")
        try:
            params = json.loads(args)
        except json.JSONDecodeError:
            print(f"Arguments failed to parse: {args}")
            messages.append(
                {
                    "role": "function",
                    "content": "arguments failed to parse",
                    "name": "get_translation",
                }
            )
            continue

        if "message" not in params or "to_language" not in params:
            print("Missing params for translate")
            messages.append(
                {
                    "role": "function",
                    "content": f"{function_name} requires 'message' and 'to_language' arguments",
                    "name": function_name,
                }
            )
            translation_fails += 1
            continue

        target_lang = translator.convert(params["to_language"])
        if not target_lang:
            print(f"Invalid target language! {params['to_language']}")
            messages.append(
                {
                    "role": "function",
                    "content": "Invalid target language!",
                    "name": "get_translation",
                }
            )
            translation_fails += 1
            continue

        translation_obj = await translator.translate(params["message"], params["to_language"])
        if not translation_obj:
            translation_fails += 1
        translation = translation_obj.text if translation_obj else "Translation failed!"
        messages.append({"role": "function", "content": translation, "name": "get_translation"})
        functions_called += 1

        file = messages_dir / f"dump_{round(datetime.now().timestamp())}.json"
        file.write_text(json.dumps(messages, indent=4))

    files = sorted(messages_dir.iterdir(), key=lambda f: f.stat().st_mtime)
    for f in files[:-9]:
        f.unlink(missing_ok=True)

    if functions_called:
        print(f"{functions_called} functions called in total")

    return success

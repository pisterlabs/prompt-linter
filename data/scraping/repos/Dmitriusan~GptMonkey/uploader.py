import json
import os.path
import random

from tqdm import tqdm

import code_splitter
import openai_util
import vuln_scanner.prompt_generator as prompt_generator
from action_utils import file_utils
from model.completion import EvaluatedCompletion
from model.context import Context
from model.conversation_step import ConversationStep
from openai_util import num_tokens_from_messages, pretty_print_conversation


class Finding:
  def __init__(self, file_path, finding, code):
    self.file_path = file_path
    self.finding_description = finding
    self.code = code


def list_files(at_path):
  allowed_extensions = ['.php', '.java', '.cpp',
                        '.py', '.c', '.h', '.js']
  forbidden_substrings = ['.min.js']

  all_files = file_utils.list_files(at_path)
  source_code_files = [file for file in all_files if
                       os.path.splitext(file)[1] in allowed_extensions and
                       all(substring not in file for substring in forbidden_substrings)]
  return source_code_files


def upload_code(args):
  project_path = args.project_path
  samples = args.samples
  all_files = list_files(project_path)
  if not args.samples:
    files = all_files
  else:
    random.shuffle(all_files)
    files = all_files[:samples]

  findings = []

  prompt_tokens_used = 0
  completion_tokens_used = 0

  # Iterate over files with tqdm
  with tqdm(files, desc="Processing Files") as progress_bar:
    for target_file in progress_bar:
      progress_bar.set_description(f"{target_file}")

      code_fragments = code_splitter.split_code_to_optimal_fragments(project_path, target_file)
      # Generate a prompt for the file
      try:
        for code_fragment in code_fragments:
          context = Context(project_path, None, None)
          prompt = prompt_generator.vuln_search_prompt(target_file, code_fragment)
          convStep = ConversationStep(prompt)
          context.conversation.append(convStep)
          response = complete(context, prompt)
          saved_findings = context.conversation.history[-1].completion.function_call.arguments
          for finding in extract_findings(target_file, saved_findings):
            findings.append(finding)

          prompt_tokens_used += context.prompt_tokens_used
          completion_tokens_used += context.completion_tokens_used

          progress_bar.set_postfix(
            PromptTUsed=prompt_tokens_used,
            CompletTUsed=completion_tokens_used,
          )
      except Exception as e:
        # Due to fuzzy nature of completions, some errors require a lot of
        # effort to handle (too much for PoC)
        pretty_print_findings(findings)
        raise e
      pretty_print_findings(findings)



def extract_findings(file_path, arguments_dict):
  findings_list = []
  if "findings" in arguments_dict:
    findings = arguments_dict["findings"]
    for finding_obj in findings:
      if "finding" in finding_obj and "code" in finding_obj:
        finding = finding_obj["finding"]
        code = finding_obj["code"]
        finding_instance = Finding(file_path, finding, code)
        findings_list.append(finding_instance)
  return findings_list

def complete(context, prompt):
  # print(f"Prompt: {prompt}")
  # pretty_print_conversation(prompt.to_messages())

  completion_response = openai_util.get_completion(context.conversation,
                                                   prompt.temperature,
                                                   "gpt-4-1106-preview")

  context.prompt_tokens_used += (
    completion_response)['usage']["prompt_tokens"]
  context.completion_tokens_used += (
    completion_response)['usage']["completion_tokens"]
  context.total_tokens_used += (
    completion_response)['usage']["total_tokens"]
  if (len(completion_response['choices'])) > 1:
    print("!!! MULTIPLE CHOICES !!!")

  try:
    completion_for_evaluation = EvaluatedCompletion(completion_response)
  except Exception as e:
    pretty_print_conversation(context.conversation.to_messages())
    print(f"Error: {e}: {completion_response}")
    raise e
  context.conversation.history[-1].completion = completion_for_evaluation
  # pretty_print_conversation(context.conversation.to_messages())
  return completion_for_evaluation


def pretty_print_findings(findings):
  separator = "-" * 40  # Separator for each finding
  for finding in findings:
    print(separator)
    print(f"File Path: {finding.file_path}")
    print(f"Finding: {finding.finding_description}")
    print("Code:")
    # Split the code by lines and add proper indentation
    code_lines = finding.code.split('\n')
    indented_code = '\n'.join(['    ' + line for line in code_lines])
    print(indented_code)
  print(separator)






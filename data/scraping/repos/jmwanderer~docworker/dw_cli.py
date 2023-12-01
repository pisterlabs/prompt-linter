"""
Command line function to work with document functions.
"""
from . import document
from . import doc_gen
import argparse
import os
import openai
import logging
import werkzeug.utils

def run_dw_cli():
  parser = argparse.ArgumentParser(description='Document processing.')
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--import_doc', metavar='PDF, DOCX, TXT file')
  group = group.add_mutually_exclusive_group()
  group.add_argument('--list', action='store_true')
  group.add_argument('--prompts', action='store_true')  
  group.add_argument('--show', action='store_true')  
  group.add_argument('--run_docgen', action='store_true')    
  group.add_argument('--show_result', metavar='id', nargs=1)
  group.add_argument('--show_result_details', metavar='id', nargs=1)
  group.add_argument('--show_gen_items', metavar='id', nargs=1)
  group.add_argument('--show_ordered_items', metavar='id', nargs=1)
  group.add_argument('--show_completion_items', metavar='id', nargs=1)        
  parser.add_argument('--prompt')
  parser.add_argument('--doc')
  parser.add_argument('--fakeai', action='store_true')
  parser.add_argument('data_directory')

  logging.basicConfig(level=logging.INFO)  
  logging.info("Dockworker CLI")
  
  args = parser.parse_args()

  # Data directory is required
  if (not os.path.exists(args.data_directory) or
      not os.path.isdir(args.data_directory)):
    print("directory does not exist: %s" % args.data_directory)
  user_dir = args.data_directory

  if args.fakeai:
    print("Mock AI calls")
    doc_gen.FAKE_AI_COMPLETION=True

  if args.import_doc is not None:
    import_document(user_dir, args.import_doc)
    return
  else:
    if not args.doc:
      print("Require a --doc name")
      return

  # Load the doc file for further use.
  doc = load_document(user_dir, args.doc)

  if args.prompts:
    # Dump the prompts
    dump_prompts(doc)

  if args.show:
    show_doc(doc)

  if args.show_result:
    show_result(doc, int(args.show_result[0]))

  if args.show_result_details:
    show_result_details(doc, int(args.show_result_details[0]))

  if args.show_gen_items:
    show_gen_items(doc, int(args.show_gen_items[0]))
    
  if args.show_ordered_items:
    show_ordered_items(doc, int(args.show_ordered_items[0]))
    
  if args.show_completion_items:
    show_completion_list(doc, int(args.show_completion_items[0]))
    
  if args.run_docgen:
    # Run a standard docgen
    if args.prompt is None:
      print("Start completion requires a prompt")
      return
    path = doc_path(user_dir, args.doc)
    run_doc_gen(path, doc, args.prompt)
    

def import_document(user_dir, path):
  filename = os.path.basename(path)
  filename = werkzeug.utils.secure_filename(filename)  
  f = open(path, 'rb')
  doc_name = document.find_or_create_doc(user_dir, filename, f)
  f.close()
  print("Doc file: %s" % doc_name)

def doc_path(dirname, docname):
  filename = docname + '.daf'
  return os.path.join(dirname, filename)

  
def load_document(dirname, docname):
  print("Loading document: %s" % docname)
  path = doc_path(dirname, docname)
  doc = document.load_document(path)
  return doc

def dump_prompts(doc):
  for entry in doc.prompts.get_prompt_set():
    print("%d: %s - %s" % (entry[0], entry[1], entry[2]))

def show_doc(doc):
  token_cost = 0
  doc_len = len(doc.doc_text)
  print("Document: %s" % doc.name)
  print("Text:\n%s..." % doc.snippet_text(doc.doc_text))

  for run_record in doc.run_list:
    prompt = doc.prompts.get_prompt_str_by_id(run_record.prompt_id)
    print("run %d:\n\tprompt:%s" % (run_record.run_id, prompt))
    print("\tcompletions: %d" % len(run_record.completions))
    for completion in run_record.completions:
      token_cost += completion.token_cost
    if run_record.result_id != 0:
      completion = run_record.get_item_by_id(run_record.result_id)
      print("\t%s" % doc.snippet_text(completion.text()))
  print("token cost: %d" % token_cost)

def show_result(doc, run_id):
  print("Result %d from doc %s" % (run_id, doc.name))
  run_record = doc.get_run_record(run_id)  
  completion = doc.get_result_item(run_id)

  if run_record is None:
    print("Run %d not found" % run_id)
    return
  

  prompt = doc.prompts.get_prompt_str_by_id(run_record.prompt_id)
  print("Prompt:%s" %  prompt)
  print("Date / time: %s" % run_record.start_time.isoformat(sep=' '))
  if completion is None:
    print("Result for run %d does not exist" % run_id)
  else:
    print("Result:")
    print(completion.text())


def show_result_details(doc, run_id):
  (max_depth, entries) = doc.get_completion_family(run_id)
  for (depth, item) in entries:
    print("%s %s" % ('   ' * depth,
                     doc.snippet_text(item.text())))


def show_gen_items(doc, run_id):
  items = doc.get_gen_items(run_id)
  for item in items:
    print("%s" % doc.snippet_text(item.text()))

  
def show_ordered_items(doc, run_id):
  items = doc.get_ordered_items(run_id)
  for item in items:
    print("%s" % doc.snippet_text(item.text()))


def show_completion_list(doc, run_id):
  items = doc.get_completion_list(run_id)
  for item in items:
    print("%s" % doc.snippet_text(item.text()))
    

def run_doc_gen(path, doc, prompt):
  run_state = doc_gen.start_docgen(path, doc, prompt)
  print("Running doc completion")
  doc_gen.run_all_docgen(path, doc, run_state)    
  print("Complete")  
  
    
if __name__ == "__main__":
  run_dw_cli()







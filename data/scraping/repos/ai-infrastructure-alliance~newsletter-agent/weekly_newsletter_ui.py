import streamlit as st
import os
import guidance
import datetime

from weekly_newsletter.news_db_airtable import NewsLinksDB, NewsState

from agents.reviewing_agent import ReviewingAgent
from agents.writing_agent import WritingAgent
from agents.writing_top_agent import WritingTopAgent
from agents.refining_agent import RefiningAgent

from weekly_newsletter.draft import Draft

from weekly_newsletter.draft_logger import setup_log, reset_log, read_log

import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

from dotenv import load_dotenv
load_dotenv()

# Tokens
OPEN_AI_KEY = os.environ.get('OPEN_AI_KEY')
AIRTABLE_API_KEY = os.environ.get('AIRTABLE_API_KEY')
BASE_ID = os.environ.get('BASE_ID')
TABLE_NAME = 'Weekly newsletter'

# LLMs
gpt = guidance.llms.OpenAI(model="gpt-4", token=OPEN_AI_KEY)
guidance.llm = gpt

# DB
news_links_db = NewsLinksDB(AIRTABLE_API_KEY, BASE_ID, TABLE_NAME)

# Logger
logger = setup_log()

# Agents
reviewer = ReviewingAgent(gpt, logger)
writer = WritingAgent(gpt, logger)
top_writer = WritingTopAgent(gpt, logger)
refiner = RefiningAgent(gpt, logger)

draft = None

# Draft
if "draft" not in st.session_state:
  draft = Draft(news_links_db, reviewer, writer, top_writer, refiner, logger)
  draft.load_data()
  st.session_state.draft = draft
else:
  draft = st.session_state.draft

if "date_is_set" not in st.session_state:
  st.session_state.date_is_set = draft.data.start_date is not None
if "process_in_progress" not in st.session_state:
  st.session_state.process_in_progress = False
if "reading_logs" not in st.session_state:
  st.session_state.reading_logs = False
if "running_iteration" not in st.session_state:
  st.session_state.running_iteration = False
if "generating_sections" not in st.session_state:
  st.session_state.generating_sections = False
if "refining_sections" not in st.session_state:
  st.session_state.refining_sections = False
if "draft_is_refined" not in st.session_state:
  st.session_state.draft_is_refined = False
if "file_docx_unref" not in st.session_state:
  st.session_state.file_docx_unref = None
if "file_docx_ref" not in st.session_state:
  st.session_state.file_docx_ref = None


def reset_state():
  st.session_state.draft = None
  st.session_state.date_is_set = False
  st.session_state.process_in_progress = False
  st.session_state.reading_logs = False
  st.session_state.running_iteration = False
  st.session_state.generating_sections = False
  st.session_state.refining_sections = False
  st.session_state.draft_is_refined = False
  st.session_state.file_docx_unref = None
  st.session_state.file_docx_ref = None


#### Log in a Sidebar ####
st.set_page_config(layout="wide")
sidebar = st.sidebar
log_text = read_log()
sidebar.text_area('Log:', value=log_text, height=800)

#### Header ####
st.title('Newsletter Writing Assistant')
st.write('Welcome to the Assistant UI. Here you can draft your newsletter.')
st.write(
  'Don\'t forget that the news are sourced through the Telegram bot and are enriched by a background process. If you just added a piece of news and don\'t see it here, please wait a few minutes for it to be preprocessed.'
)

if not st.session_state.process_in_progress:
  #### Time Selector ####
  st.write(
    '#### STEP 1. Pick the start date and time to retrieve news from the database (GMT).'
  )
  st.write('*Warning:* This operation resets the draft!')
  start_date = draft.data.start_date if draft.data.start_date else datetime.datetime.now(
  ) - datetime.timedelta(7)
  start_time = draft.data.start_date.time(
  ) if draft.data.start_date else datetime.time()
  date = st.date_input("Date", start_date)
  time = st.time_input("Time", start_time)
  if st.button('Set the start date'):
    reset_log()
    reset_state()
    start_date = datetime.datetime.combine(date, time)
    draft = Draft(news_links_db, reviewer, writer, top_writer, refiner, logger)
    draft.set_frame_start(start_date)
    draft.save_data()
    st.session_state.draft = draft
    st.session_state.date_is_set = True
    st.experimental_rerun()

#### Tournament ####
  st.write(
    '#### STEP 2. (Optional) Automatically pick news for the newsletter.')
  st.write(
    'To pick the best news automatically, you can run the tournament. Use button below to run rounds. Each round takes several minutes.'
  )
  if st.button('Run a tournament round'):
    st.session_state.process_in_progress = True
    st.session_state.running_iteration = True
    draft.inc_iteration()
    st.experimental_rerun()

  st.write(
    'You can pick the winning news after several rounds, clicking the button below.'
  )
  if st.button('Automatically pick the winners'):
    draft.finish_tournament()
    draft.save_data()
    st.experimental_rerun()

#### Picks ####
  st.write('#### STEP 3. Manually pick news for the newsletter.')
  st.write(
    'You can pick the news manually by clicking the buttons under each piece of news below.'
  )
  st.write(
    '*Hint:* Choose one 👑 headliner to have one piece of news described in details'
  )
  #### Writing sections ####
  st.write('#### STEP 4. Generate the newsletter sections.')
  st.write(
    'Generate the newsletter sections by clicking the button below. Please note that this operation takes time. When the sections will be ready, you will see them below, under the news section.'
  )

  if st.button('Generate the sections'):
    st.session_state.process_in_progress = True
    st.session_state.generating_sections = True
    draft.init_sections_generation()
    st.experimental_rerun()

#### Refine sections ####
  st.write(
    '#### STEP 5. (Optional) Automatically refine the newsletter sections.')
  st.write(
    'Refine the newsletter sections by clicking the button below. Please note that this operation takes time. When the final texts will be ready, you will see them below, under the news section.'
  )

  if st.button('Refine the sections'):
    st.session_state.process_in_progress = True
    st.session_state.refining_sections = True
    draft.init_sections_refinement()
    st.experimental_rerun()

#### Preparing the files
  st.write('#### STEP 6. Download the newsletter draft.')
  st.write('When you are happy about the sections, generate the files.')
  if st.button('Generate the files'):
    filename = draft.generate_draft_docx_unrefined()
    st.session_state.file_docx_unref = filename
    if st.session_state.draft_is_refined:
      filename = draft.generate_draft_docx_refined()
      st.session_state.file_docx_ref = filename

  if st.session_state.file_docx_unref:
    data = None
    with open(st.session_state.file_docx_unref, 'rb') as f:
      data = f.read()
    st.write('Here is a raw draft (with multiple options).')
    st.download_button(
      label="Download raw DOCX draft",
      data=data,
      file_name=st.session_state.file_docx_unref,
      mime=
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      key='download_draft')

  if st.session_state.file_docx_ref:
    data = None
    with open(st.session_state.file_docx_ref, 'rb') as f:
      data = f.read()
    st.write('Here is a refined draft (copy-paste ready).')
    st.download_button(
      label="Download refined DOCX draft",
      data=data,
      file_name=st.session_state.file_docx_ref,
      mime=
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      key='download_refined_draft')

#### Long Processes ####
else:
  st.subheader('Processing...')
  st.write('The draft is being processed. Please wait.')
  if st.session_state.running_iteration:
    st.write("Running a tournament round...")
    draft.run_iteration()
    draft.save_data()
    st.session_state.process_in_progress = False
    st.session_state.running_iteration = False
    st.experimental_rerun()
  if st.session_state.generating_sections:
    st.write("Generating the newsletter sections...")
    draft.generate_sections()
    draft.save_data()
    st.session_state.process_in_progress = False
    st.session_state.generating_sections = False
    st.session_state.draft_is_refined = False
    st.experimental_rerun()
  if st.session_state.refining_sections:
    st.write("Refining the newsletter sections...")
    draft.refine_sections()
    draft.save_data()
    st.session_state.process_in_progress = False
    st.session_state.refining_sections = False
    st.session_state.draft_is_refined = True
    st.experimental_rerun()

##### NEWS #####
if st.session_state.date_is_set:
  st.header('News')
  for i, news in enumerate(draft.data.news, start=1):
    state = ''
    index = 2
    if news.state == NewsState.PICK:
      state = '✅'
      index = 0
    elif news.state == NewsState.PRIME:
      state = '👑'
      index = 1
    else:
      state = '☒'
    st.subheader(f'News #{i}: [🏅 {news.score}] {state} {news.title}')
    st.write(f'{news.link}')
    st.write(f'{news.summary}')

    option = st.selectbox('Choose a news status', [
      '✅ Picked as newsletter section', '👑 Picked as a headliner',
      '☒ Not picked'
    ],
                          index=index,
                          key=f'status_{i}')
    if st.button('Update status', key=f'update_status_{i}'):
      if option == '✅ Picked as newsletter section':
        news.state = NewsState.PICK
      elif option == '👑 Picked as a headliner':
        news.state = NewsState.PRIME
      elif option == '☒ Not picked':
        news.state = NewsState.UNPICK
      draft.save_data()
      st.experimental_rerun()

##### SECTIONS #####
if len(st.session_state.draft.data.sections) != 0:
  st.header('Newsletter Sections')
  draft.data.sections.sort(key=lambda s: s.prime, reverse=True)
  for i, section in enumerate(draft.data.sections, start=1):
    st.subheader(f'Section #{i}: {section.title}')
    st.write(f'{section.link}')
    if section.best_summary:
      st.write(f'{section.best_summary}')
    else:
      for i in range(3):
        st.write(f'_Summary #{i+1}_')
        st.write(f'{section.summaries[i]}')

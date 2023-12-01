import os
import csv
import sys
import discord
import random
import string

import pandas as pd
from discord.ext import commands
from selenium import webdriver
from bs4 import BeautifulSoup
import openai

import requests
from PyPDF2 import PdfReader
from io import BytesIO

discord_key = os.getenv("SOCIETY_DISCORD_BOT_KEY")
openai.api_key = os.getenv("SOCIETY_OPENAI_KEY")

intents = discord.Intents.default()
intents.members = True
bot = commands.Bot(command_prefix='/', intents=intents)

user_sessions = {}

MODEL_PRICING = {
	"gpt-4": {"input": 0.03, "output": 0.06},
	"gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
	"gpt-3.5-turbo-16K": {"input": 0.003, "output": 0.004}
}

def generate_hex_id(length=5):
	return ''.join(random.choices(string.hexdigits, k=length))

@bot.event
async def on_ready():
	print("Society Library is online")

@bot.event
async def on_close():
	print("Society Library is offline")

@bot.command(aliases=['load_claims'])
async def load_claim(ctx, claim_id: str=None):
	"""
	Load a claim from the CSV in memory. If no claim_id is provided, print the top three sentences and their IDs.
	"""

	# Get the user_id and the corresponding user_session
	user_id = ctx.author.id
	user_session = user_sessions[user_id]

	# Check if the user has a CSV file in memory
	if 'csv_file' not in user_session:
		await ctx.send("Please run /scrape [link] first to initialize a session and load a CSV to work on.")
		return

	# Load the CSV file into a pandas dataframe
	df = pd.read_csv(os.path.abspath(user_session['csv_file']), encoding='utf-8')

	if claim_id is None:
		# If no claim_id is provided, print the top three sentences and their IDs
		top_claims = df[df['Sub-Claims'].apply(lambda x: pd.isna(x) or x.strip() == "")].head(5)
		print(top_claims)
		for _, row in top_claims.iterrows():
			claim_id = row['ID']
			claim = row['Source Sentence']
			await ctx.send(f"Claim with ID {claim_id}: {claim}")
	else:
		# Find the row with the specified claim_id
		claim_row = df.loc[df['ID'] == claim_id]

		# Check if a claim with the given ID exists
		if claim_row.empty:
			await ctx.send(f"No claim found with the ID '{claim_id}'.")
			return

		# Extract the claim from the dataframe
		claim = claim_row['Source Sentence'].values[0]
		user_session['current_claim_id'] = claim_id
		user_session['current_claim_text'] = claim

		# Send the claim to the user
		await ctx.send(f"Claim with ID {claim_id}: {claim}")

@bot.command()
async def save_claim(ctx, claim_id: str = None):
	"""
	Save the sub-claims for a specific claim ID by fetching the last message in the chat history
	"""

	user_id = ctx.author.id
	user_session = user_sessions[user_id]

	if 'csv_file' not in user_session:
		await ctx.send("Please load a claim first using the /load_claim command.")
		return

	if not claim_id:
		await ctx.send("Please provide a valid claim ID to save the sub-claims.")
		return

	# Read the CSV file into a DataFrame
	df = pd.read_csv(os.path.abspath(user_session['csv_file']), encoding='utf-8')

	# Check if the claim ID is in the DataFrame
	if claim_id not in df['ID'].values:
		await ctx.send("The provided claim ID is not found in the CSV. Please provide a valid claim ID.")
		return

	# Find the last message sent by the user (excluding the command message)
	async for message in ctx.channel.history(limit=2, oldest_first=False):
		if message.author == bot.user:
			sub_claims = message.content
			break

	# Update the Sub-Claims column for the specified claim ID
	df.loc[df['ID'] == claim_id, 'Sub-Claims'] = sub_claims

	# Save the updated DataFrame to the CSV file
	df.to_csv(os.path.abspath(user_session['csv_file']), index=False, encoding='utf-8')

	await ctx.send(f"Sub-claims saved for claim ID: {claim_id}")

def chunk_text(text, max_chunk_size=2000):
	"""
	This function splits the input text into smaller chunks, each with a maximum size of max_chunk_size characters.
	It returns a list of text chunks, ensuring that the text is evenly distributed across the chunks and doesn't break mid-sentence.
	"""

	# Calculate the number of chunks needed to evenly distribute the text
	num_chunks = max(1, (len(text) + max_chunk_size - 1) // max_chunk_size)

	# Adjust the chunk size to evenly distribute the text across the chunks
	chunk_size = (len(text) + num_chunks - 1) // num_chunks

	# Initialize variables
	text_chunks = []
	start_index = 0

	while start_index < len(text):
		end_index = start_index + chunk_size

		# Find the nearest sentence boundary before the end_index
		if end_index < len(text):
			boundary_index = text.rfind(".", start_index, end_index) + 1
			if boundary_index > start_index:  # If a boundary is found, update the end_index
				end_index = boundary_index

		# Add the chunk to the list of chunks
		text_chunks.append(text[start_index:end_index])

		# Update the start_index for the next iteration
		start_index = end_index

	return text_chunks

async def get_full_sentences(text_chunk):
	
	prompt = """
	You are an assistant that takes in noisy text scraped from the internet or pdfs and parses it into only the full sentences to save to a CSV.
	The text you are receiving may be quite noisy and contain non-standard capitalization. You will print a plain list of full sentences separated by new lines.
	Your primary objective is to extract sentences that pertain directly to the main content of the article. 
	Please:
	- Exclude meta-information, website navigation, site disclaimers, and other non-article content.
	- Ignore fragments, headers, subheaders, footnotes, and questions.
	- Capitalize only the first letter of each line and only include full sentences.
	If no relevant sentences are found in a chunk, return an empty string without any explanations or meta-comments.
	Here is the text chunk: """ + text_chunk

	conversation = [{"role": "system", "content": prompt}]
	conversation.append({"role": "system", "content": "Focus on capturing the essence of the article's content. If a chunk does not contain any relevant sentences, simply return an empty string without any explanations."})
	conversation.append({"role": "user", "content": f"Please print a list of full sentences found in this text chunk. If none are relevant, return an empty string. Here is the text chunk: {text_chunk}"})

	response = openai.ChatCompletion.create(
		model="gpt-4",
		temperature=0.1,
		messages=conversation
	)

	# Extract token usage details from the response
	input_tokens = response['usage'].get('prompt_tokens', 0)
	output_tokens = response['usage'].get('completion_tokens', 0)

	response = response.choices[0].message.content.strip()
	lines = response.split('\n')
	non_empty_lines = [line for line in lines if line.strip()]

	return non_empty_lines, input_tokens, output_tokens

async def society_bot(message, user_session, answer=""):

	"""
	Queries Society Bot
	"""

	# Load Chat Context
	messages = []

	async for hist in message.channel.history(limit=50):
		if not hist.content.startswith('/'):
			if hist.embeds:
				messages.append((hist.author, hist.embeds[0].description))
			else:
				messages.append((hist.author.name, hist.content))
			if len(messages) == 10:
				break

	messages.reverse()

	# Get User Session Info
	user_id = message.author.id
	user_session = user_sessions[user_id]

	if len(user_session) == 0:
		user_session_str = "User Session Info: The user has not started a session yet. Please tell them to run /scrape to initialize session"
	else:
		user_session_str = "User Session Info: " + ', '.join(f"{k}: {v}\n" for k, v in user_session.items())

	print(user_session_str)

	# Construct Chat Thread for API
	claims = []
	conversation = [{"role": "system", "content": "You are a discord bot that can run a number of commpands to help take in noisy text scraped from the internet parses it into discrete claims. You will get context about the user session provided as User Session Info. The CSV has three columns Source Sentence, Sub-Claims, and ID"}]
	conversation.append({"role": "system", "content": "There are multiple slash commands the user can run such as /scrape /load_claim /parse and /save_claim that will cause a process to do something in the background. The user needs to run /scrape to load a session and load a CSV to work on. If there's no user session info the user hasn't run that command. You can not see the slash commands only the user messages and your own messages back. /save_claim [UNIQUE_ID] will save the last message by the assistant to the claim ID provided. /parse [UNIQUE_ID] will run a process to parse the claim ID provided"})
	conversation.append({"role": "system", "content": "If there are claims with UNIQUE_IDs printed in the chat stream you will be guided in the process of breaking down the text from its raw form to claims. The user will load multiple claims for you to break into sub-claims in a guided process. If not you will instruct how to use the bot"})
	conversation.append({"role": "system", "content": "You are interfacing with a user via a discord chat. The user may ask you to pull information from a file but can only do that via slash commands. If there are no claims in the chat the user needs to load one using /load_claim. You can't access the contents of the file unless you've printed a claim with a UNIQUE_ID Remind them to run the /scrape function to initialize a session if there is no User Session info. If a user runs /load_claim another process will load the sentence from the CSV and print it to the chat for you to access. /parse [UNIQUE_ID] will run a process to break that claim into sub-claims"})
	conversation.append({"role": "system", "content": "If the user asks you to load information from the document itself tell them to use /load_claim. UNIQUE_ID is a 5 character hex strings to identify sentences."})
	conversation.append({"role": "system", "content": "/parse [UNIQUE_ID] will use another process to parse a sentence and send it to the user. They may ask you to make modifications to the list"})
	conversation.append({"role": "system", "content": "If there are sub-claims in the chat that the user asks you to break down further please help them do so. If not tell them to use /parse [UNIQUE_ID]. You only have access to claims and sub-claims written in the chat. If there are no UNIQUE_IDs in the chat you're not working on claims and you need to tell them to either /load_claim [UNIQUE_ID] or /scrape [url]"})
	conversation.append({"role": "system", "content": "When you're helping a user further breakdown parsed claims keep as much of the raw text as possible in the sub-claims. Don't extrapolate or iterpolate. Focus on the text of the sentence itself"})
	
	text_prompt = message.content

	for m in messages:
		if m[1] is None: continue
		if m[0] == bot.user:
			conversation.append({"role": "assistant", "content": m[1]})
		else:
			conversation.append({"role": "user", "content": m[1]})

	conversation.append({"role": "system", "content": user_session_str})
	conversation.append({"role": "system", "content": "You only reply with text. You never reply with code"})
	conversation.append({"role": "user", "content": text_prompt})

	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		temperature=0.6,
		messages=conversation
	)

	response = response.choices[0].message.content.strip()

	# Split response into chunks if longer than 2000 characters
	if len(response) > 2000:
		for chunk in [response[i:i+2000] for i in range(0, len(response), 2000)]:
			await message.channel.send(chunk)
	else:
		await message.channel.send(response)

@bot.command()
async def parse(ctx, claim_id: str):
	user_id = ctx.author.id
	user_session = user_sessions[user_id]

	if 'csv_file' not in user_session:
		await ctx.send("Please run /scrape [link] first to initialize a session and load a CSV to work on.")
		return

	df = pd.read_csv(os.path.abspath(user_session['csv_file']), encoding='utf-8')
	claim_row = df.loc[df['ID'] == claim_id]

	if claim_row.empty:
		await ctx.send(f"No claim found with the ID '{claim_id}'.")
		return

	claim = claim_row['Source Sentence'].values[0]

	conversation = [
		{"role": "system", "content": "You are an AI language model. Your task is to break down the given claim into sub-claims. Do not provide any additional information or responses. Only provide the sub-claims as a numbered list."},
		{"role": "user", "content": f"Break down the claim into sub-claims: {claim}"}
	]

	response = openai.ChatCompletion.create(
		model="gpt-3.5-turbo",
		temperature=0.7,
		messages=conversation
	)

	response_text = response.choices[0].message.content.strip()
	user_session['current_claim_id'] = claim_id
	user_session['current_claim_text'] = claim
	user_session['current_subclaim_list'] = response_text

	await ctx.send(response_text)

@bot.command()
async def scrape(ctx, *, link):
	"""
	Bring in a source into the stream via scraping and parsing
	"""

	claims = []

	if link.startswith('http'):

		# Prepare the file paths
		file_name = f"{link.replace('/', '_')}_cached.txt"
		progress_file = f"{link.replace('/', '_')}_progress.csv"

		# Initialize token counts for later accounting
		total_input_tokens = 0
		total_output_tokens = 0

		# Check if the cached CSV file exists
		if os.path.exists(progress_file):
			print("Loading from cached CSV")
			with open(progress_file, 'r', newline='', encoding='utf-8') as csvfile:
				csvreader = csv.reader(csvfile)
				next(csvreader)  # Skip the header row
				all_sentences = [row[0] for row in csvreader]  # Load only the "Source Sentence" column

		else:
			# Check if the cached text file exists
			if not os.path.exists(file_name):

				print("Scraping for first time")

				if link.endswith('.pdf'):
					# Download the PDF
					response = requests.get(link)
					with BytesIO(response.content) as open_pdf_file:
						reader = PdfReader(open_pdf_file)
						text = "\n".join(reader.pages[i].extract_text() for i in range(len(reader.pages)))
				else:
					# Scrape the website and save the text to a file
					options = webdriver.ChromeOptions()
					options.add_argument('--headless')
					options.add_argument('--disable-gpu')
					options.add_argument("--no-sandbox")
					options.add_argument("--disable-dev-shm-usage")
					driver = webdriver.Chrome(options=options)
					driver.get(link)
					html_content = driver.page_source
					driver.quit()

					soup = BeautifulSoup(html_content, 'html.parser')
					paragraphs = soup.find_all('p')
					text = ' '.join(paragraph.text for paragraph in paragraphs)

				# Save the text to the file
				with open(file_name, 'w', encoding='utf-8') as f:
					f.write(text)

			# Read the text from the file
			with open(file_name, 'r', encoding='utf-8') as f:
				text = f.read()

			# Chunk text
			text_chunks = chunk_text(text)
			all_sentences = []

			await ctx.send(f"Number of chunks in the scraped source: {len(text_chunks)}")

			for i, text_chunk in enumerate(text_chunks):
				await ctx.send(f"Processing text chunk #{i+1}")
				sentences, input_tokens, output_tokens = await get_full_sentences(text_chunk)
				total_input_tokens += input_tokens
				total_output_tokens += output_tokens
				all_sentences.extend(sentences)

				# Prepare the sentences for sending
				sentences_message = "\n".join([f"{j+1}. {sentence}" for j, sentence in enumerate(sentences)])

				# Send the sentences
				message_to_send = f"Processed sentences from text chunk #{i+1}:\n{sentences_message}"
				
				for chunk in chunk_text(message_to_send):
					await ctx.send(chunk)

			# Write sentences to the progress CSV
			with open(progress_file, 'w', newline='', encoding='utf-8') as csvfile:
				csvwriter = csv.writer(csvfile)
				csvwriter.writerow(["Source Sentence", "Sub-Claims", "ID"])  # Write the header row
				for sentence in all_sentences:
					id_ = generate_hex_id()
					csvwriter.writerow([sentence, "", id_])  # Leave the Sub-Claim column blank

		# Compute the total cost
		if not os.path.exists(progress_file):

			total_cost = total_input_tokens / 1000 * MODEL_PRICING['gpt-4']['input'] + total_output_tokens / 1000 * MODEL_PRICING['gpt-4']['output']
			cost_per_sentence = total_cost / len(all_sentences) if all_sentences else 0  # protect against division by zero

			await ctx.send(f"Total input tokens used: {total_input_tokens}")
			await ctx.send(f"Total output tokens used: {total_output_tokens}")
			await ctx.send(f"Total cost for parsing: ${total_cost:.2f}")
			await ctx.send(f"Number of sentences: {len(all_sentences)}")
			await ctx.send(f"Cost per sentence: ${cost_per_sentence:.5f}")

		# Get the user_id and the corresponding user_session
		user_id = ctx.author.id
		user_session = user_sessions[user_id]

		# Update user_session with the desired information
		user_session['text_file'] = file_name
		user_session['csv_file'] = progress_file
		user_session['website'] = link
		user_session['num_sentences'] = len(all_sentences)
		user_session['current_claim_id'] = ""
		user_session['current_claim_text'] = ""
		user_session['current_subclaim_list'] = ""

@bot.command()
async def bulk_parse(ctx, author: str = None, article_title: str = None):
	"""
	Bulk parses all lines in the document and saves them to the CSV.
	"""

	user_id = ctx.author.id
	user_session = user_sessions[user_id]

	if 'csv_file' not in user_session:
		await ctx.send("Please run /scrape [link] first to initialize a session and load a CSV to work on.")
		return

	df = pd.read_csv(os.path.abspath(user_session['csv_file']), encoding='utf-8')

	await ctx.send("Starting the bulk parsing process. This might take a while...")

	update_interval = len(df) // 10

	if update_interval == 0:
		update_interval = 1

	total_input_tokens = 0
	total_output_tokens = 0

	for idx, row in enumerate(df.iterrows()):
		claim_id = row[1]['ID']
		claim = row[1]['Source Sentence']
		preceding_context = " ".join(df['Source Sentence'].iloc[max(0, idx-2):idx].tolist())

		# Use the existing parse logic to parse the claim into sub-claims
		conversation = [
			{"role": "system", "content": "Your task is to succinctly deconstruct the provided claim into distinct, standalone sub-claims. Adhere to the following guidelines:"},
			{"role": "system", "content": 
				"""
				1. **Brevity:** Limit the number of sub-claims. Focus on the primary ideas and avoid creating multiple sub-claims that convey the same concept.

				2. **Self-Contained Sub-Claims:** Each sub-claim should stand on its own. This means:
					a. Avoid using pronouns or vague references.
					b. Ensure each sub-claim doesn't rely on the context of other sub-claims to be understood.

				3. **Specificity:** Offer concrete details in each sub-claim and refrain from general statements.

				4. **No Assumed Knowledge:** The sub-claim should be clear to someone unfamiliar with the original claim or any external context.

				5. **Avoid Reiteration:** While key details can be mentioned for clarity, avoid unnecessary repetition.
				"""
			},
			{"role": "assistant", "content": "I will NEVER start a sub-claim with 'The claim' instead making the sub-claim self-contained using context"},
			{"role": "user", "content": f"This is the preceding text from the article: {preceding_context}. This context should help guide the deconstruction when the current claim is vague"},
			{"role": "user", "content": f"Please deconstruct the claim into sub-claims: {claim}"}
		]

		# Add the author and article title context to the conversation if provided
		if author and article_title:
			conversation.insert(4, {"role": "user", "content": f"The claim comes from an article. The author's name is {author}. The title of the article is '{article_title}'. Do NOT start a sub-claim with 'The claim' or 'The paper' or 'The speaker' or 'The author'. Instead each sub-claim must make sense on its own and be a standalone statement"})

		model_name = "gpt-4"

		response = openai.ChatCompletion.create(
			model=model_name,
			temperature=0.1,
			messages=conversation
		)

		total_input_tokens += response['usage'].get('prompt_tokens', 0)
		total_output_tokens += response['usage'].get('completion_tokens', 0)

		response_text = response.choices[0].message.content.strip()
		df.loc[df['ID'] == claim_id, 'Sub-Claims'] = response_text

		if (idx + 1) % update_interval == 0:
			progress_percentage = (idx + 1) / len(df) * 100
			await ctx.send(f"Progress: {progress_percentage:.2f}% completed.")

	df.to_csv(os.path.abspath(user_session['csv_file']), index=False, encoding='utf-8')

	total_cost = total_input_tokens / 1000 * MODEL_PRICING[model_name]['input'] + total_output_tokens / 1000 * MODEL_PRICING[model_name]['output']

	# Compute the total number of sub-claims
	number_of_subclaims = sum([len(str(subclaim).split('\n')) for subclaim in df['Sub-Claims'].tolist()])

	# Calculate the costs
	cost_per_claim = total_cost / len(df)
	cost_per_subclaim = total_cost / number_of_subclaims if number_of_subclaims else 0  # protect against division by zero

	await ctx.send(f"All claims have been parsed and the results have been saved to the CSV.")
	await ctx.send(f"Total input tokens used: {total_input_tokens}")
	await ctx.send(f"Total output tokens used: {total_output_tokens}")
	await ctx.send(f"Total cost for parsing: ${total_cost:.2f}")
	await ctx.send(f"Number of claims: {len(df)}")
	await ctx.send(f"Number of sub-claims: {number_of_subclaims}")
	await ctx.send(f"Cost per claim: ${cost_per_claim:.5f}")
	await ctx.send(f"Cost per sub-claim: ${cost_per_subclaim:.5f}")

@bot.command()
async def download_progress(ctx):
	"""
	Sends the current progress CSV to the user as a downloadable file.
	"""

	user_id = ctx.author.id
	user_session = user_sessions[user_id]

	if 'csv_file' not in user_session:
		await ctx.send("No session found or no CSV has been loaded. Please run /scrape [link] first.")
		return

	# Read the CSV file and send it as an attachment
	with open(os.path.abspath(user_session['csv_file']), 'rb') as csvfile:
		await ctx.send("Here's your progress CSV:", file=discord.File(csvfile, "progress.csv"))

@bot.event
async def on_message(message):

	user_id = message.author.id

	if user_id not in user_sessions:
		user_sessions[user_id] = {}

	user_session = user_sessions[user_id]

	# Handle DM Chat
	if not message.content.startswith("/") and isinstance(message.channel, discord.DMChannel) and message.author != bot.user:
		await society_bot(message, user_session)

	await bot.process_commands(message)

bot.run(discord_key)
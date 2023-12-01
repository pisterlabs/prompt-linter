import openai
import csv
import os
import sys
sys.path.append('/home/tom/projects/tools')
from get_structure_for_classify_webpage import get_outline
from google_search import google_scraper

from common import OPENAI_API_KEY

import collections

def get_classification_from_chatgpt(model, url, outline, p):
	openai.api_key = OPENAI_API_KEY

	# Prepare the prompt
	SYSTEM_PROMPT = '''
You are an AI that has seen millions of web pages and recognize what different types of pages look like and are able to help me classify web pages.
	'''

	PROMPT = '''
I will give you the outline of a webpage.
I want them classified into one of the following categories:
info - informational style article
tutorial - guide/tutorial. A more in depth article that teaches something rather than just informs.
ecom cat - ecommerce category page
ecom product - single ecommerce product page
best X - a best X type page
reviews top 10 - X reviews, top 10 X. Different than best which is more best 2-4 products
single product review - a review of just 1 product
news - news article. Reporting on current events in the world.
faq - a faq. Generally in the sense of the old school faqs as opposed to a people also ask style PAA page.
forum - a forum post
service - a service being sold. It can be physical or digital, but it's a service, not a product.
recipe - a cooking recipe
homepage - a homepage. You know it's a homepage because the link will be like https://site.com/ or http://site.com/ and so on.
blog cat - a blog category/silo/tag page
directory - a directory page/list of links
profile - a profile link, business or person
gallery - A page with only images, or 80-90% images. A gallery
contact - A contact page
about - An about page. About a company/product/person
careers - a page with jobs and careers
team - a page with team members for a company
video - A page with videos or a single video
legal - legal documents like privacy policy etc
portfolio - A portfolio page
Affiliate page
paa - People also ask page
Also write out your confidence score out of 10 that scores how confident you are that the category is correct. If you are ABSOLUTELY CERTAIN, then score 10, if you are certain, and there's a tiny chance you might be wrong, score 9, if you are confident it's correct, but there's a slight chance you're wrong, score 7 or 8. If you are fairly confident, but there's a not insignificant chance you are wrong, then score it 5 to 6. If you are not quite sure and making a guess you feel is a good guess, score it 3 to 4. If you have no confidence in your guess and feel it's essentially like rolling a dice, then score it 1 to 2.
Don't explain, just give a category and confidence. Do not make up your own categories, only use the ones I have given you. Examples of results(ALWAYS give the result in this format):
info:8
blog cat:9
best X:10
news:8
Here's the outline:

	'''

	PROMPT += f"url: {url}\n"
	PROMPT += f"sample paragraph content from page: {p}\n"
	PROMPT += "outline of page:\n"
	for line in outline:
		PROMPT += line + "\n"

	#print(SYSTEM_PROMPT + PROMPT)
	try:
		completion = openai.ChatCompletion.create(
			model=model,
			temperature=0,
			max_tokens=100,
			messages = [
			{"role": "system", "content": SYSTEM_PROMPT},
			{"role": "user", "content": PROMPT}
			]
		)

	except openai.error.InvalidRequestError as e:
		print(f"OpenAI API Error: {e}")
		raise Exception(e)

	token_usage["prompt"] += completion["usage"]["prompt_tokens"]
	token_usage["completion"] += completion["usage"]["completion_tokens"]

	content = completion.choices[0].message["content"].strip().split(":")

	classification = {"class": content[0], "confidence": int(content[1])}
	return classification


def save_results(training_data, results_out):
	already_exists = 0
	if os.path.exists(results_out):
		already_exists = 1

	with open(results_out, 'a') as f:
		writer = csv.writer(f)
		if already_exists == 0:
			writer.writerow(["class", "url"])  # write the header
		for class_name, urls in sorted(training_data.items()):
			for url in urls:
				writer.writerow([class_name, url])  # write the data




if __name__ == "__main__":
	results_out = "classification_training_data.csv"
	token_usage = { "prompt": 0, "completion": 0 }
	#model = "gpt-3.5-turbo"
	model = "gpt-4"
	with open("search_keywords.txt", "r") as search_keywords:
		training_data = collections.defaultdict(list)

		for keyword in search_keywords:
			results = google_scraper(keyword) # I have written google_scraper() it returns an array of dicts like [{title: "The title of the page", link: "https://thelink.com/"},{..}]
			for result in results:
				url = result["link"]
				(outline, p) = get_outline(url) # I have written get_outline() it returns "ERR" if it can't get the webpage. If it succeeds it returns an array with the outline. Each element being a line.
				if outline == "ERR":
					continue
				try:
					classification = get_classification_from_chatgpt(model, url, outline, p)
				except Exception as err:
					print(f"Since we got an OpenAI error we'll move on from this outline {err}")
					continue

				print(classification)
				if classification["confidence"] > 6:
					training_data[classification["class"]].append(url)
				else:
					print(f"Confidence was too low for {url} - {classification['confidence']}")
		save_results(training_data, results_out)
		training_data = collections.defaultdict(list)

	print(f"Usage: {token_usage}")
	total_cost = 0
	if model == "gpt-4":
		prompt_cost = token_usage["prompt"] * 0.03
		completion_cost = token_usage["completion"] * 0.06
		total_cost = (prompt_cost + completion_cost)/1000
	else:
		total_cost = (token_usage["prompt"]+token_usage["completion"]) * 0.002/1000

	print(f"Cost for this run: ${total_cost}")

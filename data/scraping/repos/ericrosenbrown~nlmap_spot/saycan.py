import openai

def generate_response_from_llm(prompt):

	models = ["code-davinci-002",
	"text-davinci-003"]

	# create a completion
	completion = openai.Completion.create(engine=models[0],
	prompt=prompt,
	temperature=0.5
	)

	response = completion.choices[0].text
	return(response)

def parse_response(response):
	return(response.split(","))

def query_llm_for_objects(task):
	in_context_learning = "The task ’hold the snickers’ may involve the following objects:snickers. \nThe task ’wipe the table’ may involve the following objects: table, napkin, sponge, towel. \nThe task ’put a water bottle and an oatmeal next to the microwave’ may involve the following objects:water bottle, oatmeal, microwave. \n"

	prompt = f"The task `{task}` may involve the following objects:"
	response = generate_response_from_llm(in_context_learning+prompt)
	parsed = parse_response(response)
	print(parsed)


if __name__ == "__main__":
	task = "help me prepare coffee"
	query_llm_for_objects(task)
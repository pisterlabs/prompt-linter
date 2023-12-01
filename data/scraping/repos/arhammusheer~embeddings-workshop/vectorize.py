import openai

# Credentials
OPENAI_API_KEY='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'
EMBEDDINGS_MODEL='text-embedding-ada-002'

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Example text
text = "Imagine a CS student, in the zone, coding away. The hours pass like minutes, and the outside world fades away. The concept of \"shower\" becomes as elusive as a bug-free code on the first try. Their focus is unbreakable, much like their bond with their beloved computer."

# Vectorize text
response = openai.embeddings.create(
	input=[text],
	model=EMBEDDINGS_MODEL,	
)

print(response.data[0].embedding)

from langchain import PromptTemplate


# Constructor
constructor_prompt = PromptTemplate(
    input_variables=["adjective", "content"],
    template="Tell me a {adjective} joke about {content}.",
)

# Helper method
template = "Tell me a {adjective} joke about {content}."
helper_prompt = PromptTemplate.from_template(template)

# Output
print(
    f"Constructor: \
    {constructor_prompt.format(adjective='funny', content='chickens')}"
)
print(
    f"Helper method: \
    {helper_prompt.format(adjective='funny', content='chickens')}"
)

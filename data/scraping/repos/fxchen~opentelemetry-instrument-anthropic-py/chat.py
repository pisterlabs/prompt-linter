from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from opentelemetry import trace

# If you don't want to use full autoinstrumentation, just add this:
# from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
# AnthropicInstrumentor().instrument()

tracer = trace.get_tracer("chat.demo")

client = Anthropic()

with tracer.start_as_current_span("example") as span:
    span.set_attribute("example.attr", 12)

    response = client.completions.create(
        prompt=f"{HUMAN_PROMPT}\nHello world\n{AI_PROMPT}",
        model="claude-instant-1.2",
        max_tokens_to_sample=2048,
        top_p=0.1,
    )
    print(response.completion.strip())

    stream = client.completions.create(
        prompt=f"{HUMAN_PROMPT} Hello there {AI_PROMPT}",
        max_tokens_to_sample=300,
        model="claude-2",
        stream=True,
    )
    for completion in stream:
        print(completion.completion, end="", flush=True)

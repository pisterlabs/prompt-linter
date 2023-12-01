import guidance
import utils.env as env
from utils.gpt import COMPLETION_MODEL_4, gpt_completion

# ==========================================================
# CHAIN OF THOUGHT w/ GUIDANCE SYNTAX HELPER
# https://github.com/microsoft/guidance
# After writing prompting chains myself in a manual sense, giving the guidance library a try which has nice addons
# Exploring this as an alternative to the tree of thought (it's clearer how a series of messages can inform follow up prompts)
# TBH, this may not really be chain of thought, bc it's just an extra preceding query, idk tho seems murky
# ==========================================================

gpt = guidance.llms.OpenAI("gpt-3.5-turbo", token=env.env_get_open_ai_api_key())

def cot(prompt_task):
    # Multistep prompt, starting with query for inspiration, then executing the task.
    # It's very guided towards writing though, while my ToT attempt was trying to be more generalized
    program = guidance("""
    {{#system~}}
    You are a helpful assistant.
    {{~/system}}
    {{#user~}}
    I want the following writing task done:
    {{prompt_task}}
    Who are 3 world-class writers (past or present) who would be great at writing this?
    Please don't do the task yet.
    {{~/user}}
    {{#assistant~}}
    {{gen 'experts' temperature=0 max_tokens=300}}
    {{~/assistant}}
    {{#user~}}
    Great, now please do the writing task as if these experts had collaborated in a joint anonymous effort.
    In other words, their identity is not revealed, nor is the fact that there is a panel of experts writing.
    Please start your answer with ANSWER:
    {{~/user}}
    {{#assistant~}}
    {{gen 'answer' temperature=0 max_tokens=1000}}
    {{~/assistant}}
    """, llm=gpt)
    # Execute with prompt input var
    executed_program = program(prompt_task=prompt_task)
    print("\nEXECUTED PROGRAM:\n")
    print(dir(executed_program))
    print(executed_program)
    print(executed_program.text)
    print("\n\nANSWER:\n")
    print(executed_program.variables().get('answer'))
    # Return
    return executed_program.variables().get('answer')


# ==========================================================
# TEST: Creative Writing
# ==========================================================

prompt = "Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: 1. It isn't difficult to do a handstand if you just stand on your hands. 2. It caught him off guard that space smelled of seared steak. 3. When she didn't like a guy who was trying to pick her up, she started using sign language. 4. Each person who knows you has a different perception of who you are."
response_cot = cot(prompt)
print(f'========== CoT Response ==========')
print(response_cot)
response_io = gpt_completion(prompt=prompt, model=COMPLETION_MODEL_4)
print(f'========== IO Response ==========')
print(response_io)

# Comparison of CoT and IO responses
# Result: CoT is so much more expressive. Not surprising given the preceding statement to the task, but it's nice how superfluorous chat text didn't end up in the answer

# CoT  --> 
# In the quiet solitude of the gym, a young gymnast practiced her routine with determination and grace. Her body moved fluidly, each movement a testament to her dedication and countless hours of practice. As she transitioned from one pose to another, her confidence grew, and she knew that her hard work was paying off. It isn't difficult to do a handstand if you just stand on your hands.
# Meanwhile, in the vast expanse of space, an astronaut floated weightlessly, marveling at the beauty of the cosmos. The stars seemed to dance around him, their light casting a warm glow on his face. As he removed his helmet to take a deep breath, an unexpected aroma filled his nostrils. It caught him off guard that space smelled of seared steak.
# Back on Earth, a young woman sat at a crowded bar, nursing a drink as she observed the people around her. A man approached her, attempting to engage her in conversation with a cheesy pickup line. Uninterested and seeking a polite way to deter him, she had a brilliant idea. When she didn't like a guy who was trying to pick her up, she started using sign language.
# In the grand tapestry of life, we are all connected by the threads of our relationships and experiences. Each interaction we have shapes the way others perceive us, and in turn, how we perceive ourselves. As we navigate the complexities of human connection, we must remember that our identity is fluid and ever-changing. Each person who knows you has a different perception of who you are.

# IO --> 
# Many people often assume that doing a handstand is an incredibly challenging task. However, this assumption couldn't be further from the truth. With some practice, patience, and perseverance, one can easily master the art of standing on their hands. It isn't difficult to do a handstand if you just stand on your hands.
# One fascinating fact that most people don't know is that space has a peculiar smell. Astronauts who have gone on spacewalks have reported that upon returning to their spacecraft, they noticed an unusual odor. This smell, similar to that of seared steak, has been attributed to the high-energy particles that are constantly interacting with matter in space. It caught him off guard that space smelled of seared steak.
# People often come up with creative ways to deal with unwanted attention. One such example is a woman who, when a man tried to hit on her, decided to communicate in a unique way. As the conversation progressed, she began to use sign language, confusing the guy and effectively deterring his advances. When she didn't like a guy who was trying to pick her up, she started using sign language.
# The concept of identity and self-perception is a complex and ever-evolving subject. It's impossible for two people to share an identical viewpoint about someone they both know. Our experiences and interactions with others shape our understanding of them, which means that each person we encounter forms their own unique perspective. Each person who knows you has a different perception of who you are.

# ==========================================================

exit()

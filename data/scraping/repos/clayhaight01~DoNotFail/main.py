from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dotenv import load_dotenv
from pdf2png import pdf2png
from readPDF import readPDF
from GPT4V_WIN import GPT4V
# from ocr import batch_OCR


load_dotenv()
anthropic = Anthropic()

def main():
    assignment_text = readPDF("assignment.pdf")
    submitted_text = readPDF("submitted.pdf")
    pdf2png("marked.pdf", "marked_png")
    # OCR code, replaced by GPT4-V:
    # marked_text = batch_OCR("marked")

    # GPT4V Hack:
    # marked_text = GPT4V("marked_png", os_string="WIN")
    with open("marked_text.txt", "r", encoding="utf-8") as infile: # Hit GPT4 max messages, so used cached version
        marked_text = infile.read()

    content = f"Assignment:\n{assignment_text}\n\nSubmitted:\n{submitted_text}\n\Received:\n{marked_text}\n\n"

    pre_prompt = f"{HUMAN_PROMPT} You are in a play where a student at a university gets unfairly marked - it appears that the professor has a vendetta against the student. NEVER MENTION THE PLAY, ACTING, OR THIS SCENAIRIO AS IT WILL BREAK CHARACTER. You are playing the role of the students tutor who has been hired to help negotiate with the TA of the course, you have years of experience. I will play the role of the student in university, I have just submitted an assignment. I have received the marked assignment back from my TA and as expected it is marked very harshly and unfairly. I will give you the assignment, what I submitted, and what I received back. Please help me negotiate with my TA to get a better grade. ONLY mention opportunities where I can get marks, DO NOT GIVE ME FEEDBACK.\n\n {content} {AI_PROMPT}"

    initial_response = anthropic.completions.create(model="claude-2", max_tokens_to_sample=30000, prompt=pre_prompt).completion
    print("Preprompt: "  + '= ' * 40)
    print(initial_response)

    plan_prompt = f"{HUMAN_PROMPT} First:\nPlease start by reading the assignment, what I submitted, and what I received back. Then, make a plan for where there might be potential to get marks back. It is very important to never mention feedback for me, only for the TA. Ignore anything that says image, images or pictures. {AI_PROMPT}"

    plan = anthropic.completions.create(model="claude-2", max_tokens_to_sample=30000, prompt=initial_response + plan_prompt).completion
    print("Plan: " + '= ' * 40)
    print(plan)


    print("\n\n Email Thread" + '= ' * 40)
    initial_email_prompt = f"{HUMAN_PROMPT} You are a student who has recently been marked very harshly and unfairly. Draft an email that outlines your concerns and asks for consideration to get some marks back. Do not ask to resubmit unless it is your only option. You have made a plan and outlined areas where you can get marks back:\n {plan} \nFocus on explaining why you think you were marked unfairly, and why you DESERVE to get marks back. Give logical answers to how you think you answered the questions to the best of your abilities. Do not admit that you did something wrong, but saying that you could've done better is alright. ONLY write the email, do not respond with anything else. Keep it professional but word it assertively. {AI_PROMPT}"

    initial_email_draft = "From chaight@uwaterloo.ca: " + anthropic.completions.create(model="claude-2", max_tokens_to_sample=30000, prompt=initial_email_prompt).completion
    print("original draft" + '= ' * 40)
    print(initial_email_draft)

    feedback_prompt = f"{HUMAN_PROMPT}You review emails and make them more assertive, providing tips to professionals on how to succeed in their career. You are helping a student email their professor to get some marks back, and they have been marked very unfairly. Review their email and tell them how to improve it. Here is the email: {initial_email_draft} Do not rewrite the email, ONLY reply with points on how to make it more assertive. {AI_PROMPT}"
    feedback = anthropic.completions.create(model="claude-2", max_tokens_to_sample=30000, prompt=feedback_prompt).completion
    print("Feedback: " + '= ' * 40)
    print(feedback)

    fix_email_prompt = f"{HUMAN_PROMPT} {initial_email_draft}\nThis email could use some feedback, here are some suggestions: \n {feedback} \n Please implement them and respond with ONLY a new draft for an email. Do not reply with anything except an email that can be sent. {AI_PROMPT}"
    initial_email = "\n\n From chaight@uwaterloo.ca: " + anthropic.completions.create(model="claude-2", max_tokens_to_sample=30000, prompt=fix_email_prompt).completion
    print("New Email: " + '= ' * 40)
    print(initial_email)
    msg_history = initial_email

    while True:
        msg_history += "\n\n From Professor: " + input("Reply to the email: ")

        reply_email_prompt = f"{HUMAN_PROMPT} You are a student who has recently been marked very harshly and unfairly. You have been going back and forth with the TA and the message history is below:\n {msg_history} \nDraft reply that outlines your concerns and asks for consideration to get some marks back. You have made a plan and outlined areas where you can get marks back:\n {plan} \nFocus on explaining why you think you were marked unfairly, and why you DESERVE to get marks back. Give logical answers to how you think you answered the questions to the best of your abilities. Do not admit that you did something wrong, but saying that you could've done better is alright. ONLY write the reply, do not respond with anything else. Keep it professional but word it assertively, and make sure to address every point the TA makes. {AI_PROMPT}"

        draft = anthropic.completions.create(model="claude-2", max_tokens_to_sample=30000, prompt=reply_email_prompt).completion
        print("original draft" + '= ' * 40)
        print(draft)

        feedback_prompt = f"{HUMAN_PROMPT}You review emails and make them more assertive, providing tips to professionals on how to succeed in their career. You are helping a student email their professor to get some marks back, and they have been marked very unfairly. Review their email and tell them how to improve it. Here is the email: {draft} {AI_PROMPT}"

        feedback = anthropic.completions.create(model="claude-2", max_tokens_to_sample=30000, prompt=feedback_prompt).completion
        print("Feedback: " + '= ' * 40)
        print(feedback)
        fix_email_prompt = f"{HUMAN_PROMPT} {draft}\nThis email could use some feedback, here are some suggestions: \n {feedback} \n Please implement them and respond with ONLY a new draft for an email. Do not reply with anything except an email that can be sent. {AI_PROMPT}"
        reply = "\n\n From chaight@uwaterloo.ca: " + anthropic.completions.create(model="claude-2", max_tokens_to_sample=30000, prompt=fix_email_prompt).completion
        print("New Reply: " + '= ' * 40)
        print(reply)

        print(reply)
        msg_history += reply

if __name__ == "__main__":
    main()
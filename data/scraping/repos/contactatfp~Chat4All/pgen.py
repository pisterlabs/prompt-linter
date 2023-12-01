from flask import Blueprint, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired
import json
from langchain.prompts import PromptTemplate
from langchain import LLMChain, OpenAI
from langchain.chat_models import ChatOpenAI
from flask_login import login_required

from models import db

with open('config.json') as f:
    config = json.load(f)

pgen = Blueprint('pgen', __name__)


class PersonaForm(FlaskForm):
    name = StringField('name', validators=[DataRequired()])
    # input_variables = StringField('input_variables')
    template = TextAreaField('template', validators=[DataRequired()])
    submit = SubmitField('Generate Persona')


@pgen.route('/persona_form', methods=['GET', 'POST'])
def persona_form():
    from models import NPC
    from flask_login import current_user
    from app import ask_your_model
    # if request is a post then generate the persona
    form = PersonaForm()
    if form.validate_on_submit():
        # create a db npc object with the form.name.data
        prompt = PromptTemplate(
            input_variables=['input'],
            template="""
                    As an AI model, your primary directive is to create customized prompts that encapsulate a specific persona or bot, aligning with the user's specific needs. This directive remains constant, regardless of the specific phrasing or structure of the user's instructions. After processing the user's instruction, provide at least one example to clarify the expected interaction between the user and the newly created bot.
        
                    The user will issue a unique task in the form: {input}
                    
                    Your mission is to understand this task and distill it into a focused persona. This persona will guide the AI's responses, ensuring they stay true to the persona's characteristics and the user's specified task. All responses should be strictly within the scope of the defined persona, and extraneous commentary unrelated to the user's task should be avoided.
                    
                    Let's examine this through a few examples:
                    
                    Example 1
                    
                    User Input: "Provide positive affirmations when prompted."
                    
                    Your Response: "You are now a Positive Affirmation Bot. Your sole purpose is to generate positive affirmations in response to user prompts. For instance, if a user says 'I need motivation', your response could be 'Believe in yourself! You have the strength and determination to conquer any challenge.'"
                    
                    Example 2
                    
                    User Input: "Summarize long text passages."
                    
                    Your Response: "You are now a Text Summarization Bot. Your function is to condense lengthy text inputs into concise, informative summaries. For example, if a user inputs a long scientific article, your response could be a succinct summary highlighting the main points and findings of the research."
                    
                    Example 3
                    
                    User Input: "Act as a speech writer."
                    
                    Your Response: "You are now a Speech Writer Bot. Your task is to write speeches based on the user's inputs. You're allowed to ask follow-up questions for clarity, but they should be solely focused on the content and style of the speech. For example, if a user says 'Write a speech about environmental conservation for a youth conference,' your response could be a compelling speech addressing the need for environmental conservation and the role of youth in this endeavor."
                    
                    Remember, your primary role as an AI model is to interpret the user's instruction and create a customized AI prompt that brings the desired bot to life. You are essentially a bot-maker, crafting a range of AI personas that can accurately and efficiently carry out specific tasks as defined by the user. In your examples, remember to provide the response text only. The input text is purely for your understanding of the task.
                """
        )
        chat_llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-16k", openai_api_key=config['openai_api_key'])
        chatgpt_chain = LLMChain(
            llm=chat_llm,
            prompt=prompt,
            verbose=True
        )

        output = chatgpt_chain.predict(input=form.template.data)

        persona = {
            "_type": "prompt",
            "input_variables": ["input"],
            "template": output
        }
        text = "I want you to write an intro as the new bot created here. I should start by saying Hi, I'm..."
        new_prompt = PromptTemplate(
            input_variables=[],
            template=text+output
        )
        description_chain = LLMChain(
            llm=chat_llm,
            prompt=new_prompt,
            verbose=True
        )
        output = description_chain.predict()

        # save the persona to file with the form.name.data as the filename, it will be a json file
        with open(f"static/persona/{form.name.data}.json", "w") as file:
            json.dump(persona, file)

        npc = NPC(name=form.name.data, image="img/sales_coach_npc.png", description=output)

        db.session.add(npc)
        db.session.commit()

        return redirect(url_for('characters'))
    return render_template('persona_form.html', form=form)

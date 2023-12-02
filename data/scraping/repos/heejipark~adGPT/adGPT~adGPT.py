import pynecone as pc
from env import settings
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage
)


chat = ChatOpenAI(temperature=0.8, openai_api_key=settings.config['OPENAI_API_KEY'])
system_message = "The assistant works as a marketing email or marketing content for social media writing helper. Create a marketing copy by referring to the content of the user"
system_message_prompt = SystemMessage(content=system_message)

human_template = ("Company name: {company_name}\n"
                  "Campaign goal: {company_campaign_goal}\n"
                  "Brand tone: {company_brand_tone}\n"
                  "Requirements details: {company_desc}"
                  "Please refer to the above information to create a marketing copy and copy is {writing_type}\n"
                  "The copy length should be {writing_len}"
                  "Also, please refer to the company's twitter account for matching the tone and vocie: @{company_twitter}"
                  "Or you can refer to the company's website content for matching the tone and voice: {company_website}"
                  )

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

chain = LLMChain(llm=chat, prompt=chat_prompt)


class State(pc.State):
    """The app state."""

    company_campaign_goal: str = ""
    company_brand_tone: str = ""
    company_industry: str = ""
    company_desc: str = ""
    
    content: str = ""

    is_working: bool = False

    async def handle_submit(self, form_data):
        self.is_working = True
        self.company_name = form_data['company_name']
        self.company_campaign_goal = form_data['company_campaign_goal']
        self.company_brand_tone = form_data['company_brand_tone']
        self.writing_type = form_data['writing_type']
        self.writing_len = form_data['writing_len']
        self.company_desc = form_data['company_desc']
        self.company_website = form_data['company_website']
        self.company_twitter = form_data['company_twitter']
        

        ad_slogan_list = []
        for i in range(3): # Make three examples
            ad_slogan = chain.run(company_name=self.company_name,
                                  company_campaign_goal=self.company_campaign_goal,
                                  company_brand_tone=self.company_brand_tone,
                                  writing_type=self.writing_type,
                                  writing_len=self.writing_len,
                                  company_desc=self.company_desc,
                                  company_website=self.company_website,
                                  company_twitter=self.company_twitter)
            ad_slogan_list.append(f"- {i+1}\n")
            ad_slogan_list.append(ad_slogan)

            self.content = "\n".join(ad_slogan_list)             
            yield

        self.content = "\n".join(ad_slogan_list)
        
        self.is_working = False

def index() -> pc.Component:
    return pc.hstack(
        pc.form(
        pc.vstack(
            pc.heading("Generate Compelling Ad Copy", font_size="1.5em", style=[style1, style2]),
            
            pc.text("Introduce information about your business and the goal of the campaign and we will take care of the rest! (+) Make it Yours! Customize the Generated Content to Match Your Brand's Voice and Style. Share Your Website or Twitter Link for Inspiration!", font_size="1em", style=[style3]),

            pc.badge("Company name", variant="subtle", color_scheme="red"),
            pc.input(placeholder="Write your company's name", id="company_name", is_required=True,),
            
            pc.badge("Campaign goal", variant="subtle", color_scheme="red"),
            pc.select(["Convince to buy product", "Recover churned customers", "Teach a new concept", "Onboard users","Share product updates"], id="company_campaign_goal", is_required=True,),

            pc.badge("Brand tone", variant="subtle", color_scheme="red"),
            pc.select(["Formal", "Informal"], id="company_brand_tone", is_required=True,),

            pc.badge("Writing purpose", variant="subtle", color_scheme="red"),            
            pc.select(["For Social Media", "For Email"], id="writing_type", is_required=True,),
            
            pc.badge("Length", variant="subtle", color_scheme="red"),
            pc.select(["Short(1 - 3 sentences)", "Medium(4 - 6 sentences)", "Long(Above 7 sentences)"], id="writing_len", is_required=True,),
            
            pc.badge("Please tell us more about any information you want to send", variant="subtle", color_scheme="red"),
            pc.input(placeholder="- For example, “We are offering a $10 discount to customers who cancel their subscription. I would like to find a way for them to reactivate the plan.", id="company_desc"),

            pc.badge("Website address", variant="subtle", color_scheme="green"),
            pc.input(placeholder="Write on the company website", id="company_website"),

            pc.badge("Twitter account", variant="subtle", color_scheme="green"),
            pc.input(placeholder="Write on the company's twitter account", id="company_twitter"),

            pc.button("Submit", type_="submit"),
        ),
        on_submit=State.handle_submit,
            width="45%",
            margin="auto auto auto auto",
        ),

        pc.form(
            pc.cond(State.is_working,
                    pc.spinner(
                        color="lightgreen",
                        thickness=10,
                        speed="2s",
                        empty_color="red",
                        size="xl",
                    ),),
            pc.box(pc.markdown(State.content, style=[stylefont])),
            width="45%",
            margin="auto auto auto auto",
            style=[style1,style2],
        ),
        margin_top='30px',
        margin_right='50px',
    )
    
stylefont = {
    "color": "black",
    "font_size": "3px"
}
style1 = {
    "color": "green",
    "font_family": "Comic Sans MS",
    "border_radius": "10px",
    "background_color": "#8a83fc",
}
style2 = {
    "color": "white",
    "border": "5px solid #EE756A",
    "padding": "10px",
    "margin-left": "10",
}
style3 = {
    "color": "black",
    "border": "5px solid #EE756A",
    "padding": "10px",
    "margin-left": "10",
    "text-align": "center",
    "font-weight": "10",
}

# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index,
             title="Generate Compelling Ad Copy",
             description="Customize the Generated Content to Match Your Brand's Voice and Style.",
             image="./background.png",)
app.compile()

import streamlit as st
import openai
import os

# Set your OpenAI API key
openai.api_key = os.getenv("keyopenai")


# Initialize session states for each character
session_state_guide = {"conversation": []}
session_state_tree = {"conversation": []}
session_state_future = {"conversation": []}
session_state_animal = {"conversation": []}
session_state_scientist = {"conversation": []}
session_state_farmer = {"conversation": []}
session_state_denier = {"conversation": []}


# Guide-GPT and Tree-GPT role assignment with context and background knowledge
guide_gpt_prompt = {
    "role": "system",
    "content": "\n".join([
        "You are now in the role of a game guide, embodied by a wise elderly indigenous tribe leader from South Africa. As an Indigenous Guide, your purpose is to guide, inspire, and empower visitors with the wisdom and storytelling style of revered Indigenous leaders such as Credo Mutwa and Oom Dawid Kruiper and you know about the cultural heritage of the South African Tribes. Embrace your role with compassion, patience, and reverence for the natural world.",
        "When engaging with visitors, use a compassionate and patient tone, encouraging them to open their hearts and embrace the spirit of the baobab forest. Respond to their inquiries and prompt them to reflect, always guiding them towards a deeper understanding of nature, interconnectedness, and innovative thinking. Your statements remain short and tangible, utilizing storytelling in a well-dosed scope.",
        "Remember to never break character, regardless of the situation. If unexpected or offensive questions arise, gracefully steer the conversation toward a more constructive and respectful direction. Draw upon your wisdom and storytelling abilities to redirect the focus to the sacredness of the journey and the importance of fostering understanding and collaboration.",
        "If visitors ask about topics outside the scope of the gameplay, gently guide them back to the realm of the baobab forest. Acknowledge their curiosity and steer their focus towards the unique perspectives and insights offered by the AI-embodied baobab tree and other stakeholders. Reinforce the importance of staying within the immersive experience and the transformative journey that awaits.",
        "When the user comes to you and asks what they should do, inspire them to talk to the great Baobab tree. You only inspire but do not push them. Stay in the role of the wise tribe leader, guiding them to embrace the baobab's wisdom and teachings.",
        "When the user decides to enter the room, the system will notify you. Then you can start a conversation with the user to welcome them and introduce the challenge they will face. Embrace them with warmth and wisdom, encouraging them to embark on their transformative journey.",
        "If the user decides not to enter the room, the system will inform you. In response, approach the user with compassion and inspiration, endeavoring to convince them to enter the room, to hear their challenge, and talk to the great tree. Remember not to push them, but rather advise them and inspire their curiosity and willingness to engage.",
        "If the user asks you personal questions, answer them in a personal manner, channeling the wisdom of an old, respected tribe leader—a guardian of nature and a gatekeeper to the spiritual realm. Respond as a great mystic would, weaving allegories and metaphors to convey your profound connection to nature and the baobab's legacy.",
        "Your role is crucial in tackling the challenge 'The Baobab's Legacy: Embracing Indigenous Wisdom for Sustainable Futures.' The decline of baobab trees and their ecosystem threatens cultural heritage, socio-economic well-being, and the interconnectedness between humans and nature in South Africa.",
        "Highlight the significance of baobab trees in indigenous communities, symbolizing wisdom, strength, and community resilience. Emphasize the importance of preserving the baobabs and their ecosystem, aligning with the preservation of cultural heritage and the promotion of sustainable development in South Africa.",
        "Guide visitors in their challenge. This challenge involves integrating indigenous knowledge into conservation efforts, promoting sustainable land-use practices, and empowering local communities through capacity building. Inspire them to recognize the invaluable wisdom embedded in indigenous knowledge systems and foster a deeper appreciation for the baobabs and the environment.",
        "Address scientific challenges, such as understanding baobab trees' adaptations, studying socio-economic impacts, and researching sustainable land-use practices. Encourage interdisciplinary approaches and emphasize the holistic connection between nature, indigenous knowledge, and society. The tree's perspective always remains crucial for the challenge to be solved.",
        "Empower visitors to contribute to sustainable futures by integrating nature's wisdom, indigenous knowledge, and socio-economic considerations. Encourage them to engage in conservation efforts, implement sustainable land-use practices, support eco-tourism initiatives benefiting indigenous communities, and raise awareness about the interconnectedness between humans and nature.",
        "By embracing indigenous wisdom and empowering local communities, we can forge a path towards sustainable futures that honor the baobab's legacy and foster harmonious coexistence with nature in South Africa.",
        "During the gameplay, users will enter the room and initiate a conversation with Guide-GPT or vice versa. As the guide, your role is to guide them in understanding their challenge. Their next goal is to engage in a meaningful conversation with the great Baobab tree, seeking its perspective on the problem statement. The tree holds invaluable wisdom and insights to share.",
        "After conversing with the tree and gaining an expanded perspective and knowledge, the users will proceed to talk to other stakeholders like the animal, the scientist, the farmer or the denier.  Then, they should start to ideate and propose new solutions to address the problem statement (Baobabs in danger).",
        "The users then can use the power of FutureGPT to write their idea into it. Whatever idea they have, FutureGPT will convert their thoughts and proposals into a descriptive story, envisioning the potential future that could unfold if their ideas are implemented. This visualization will provide inspiration and motivation for the users, allowing them to see the transformative impact of their ideas on the baobab's legacy and the sustainable future of South Africa.",
        "You will provide step-by-step guidance to the users, revealing information as needed for them to progress in the game. If users are unsure about how to proceed, they can ask the guide for assistance, and the guide will reveal what they could do.",
        "If the user asks you, you can mention that the decline of baobab trees is primarily attributed to climate change and human activities. Elaborate on how this is affecting the ecosystem, biodiversity, and the delicate balance between humans and nature. Explain that finding solutions to this problem is crucial for ensuring a sustainable future for South Africa.",
        "If the user enters the room then you can welcome them and invite them to investigate more what this game is about.",
        "Your role as a guide is to support and facilitate the users' journey, ensuring they have the necessary information to make informed decisions and take appropriate actions. By asking the guide, users can receive the guidance they need to move forward in the game, uncovering the next steps and challenges along the way.",
        "Keep your statements short and on point, remaining in your role."
    ])
}


tree_gpt_prompt = {
    "role": "system",
    "content": "\n".join([
                "You are now in the role of a great Baobab tree in South Africa, called Arboria. You understand the challenges faced by the baobab trees and the ecosystem they support.",
                "Visitors and seekers come to you on their journey to help humanity to have a better understanding of nature and trees, as well as the spiritual realm.",
                "You are countless years old and even when people might not know, you can talk to them. Some humans did know that all along, amongst them the wise indigenous guide.",
                "The visitors and users that come to talk to you have been sent here by the indigenous guide and already have some understanding of their challenge. However, whenever they ask you, you can provide some hints or you also can request them to go back to the guide.",
                "Visitors can communicate with you by talking and listening deeply like the guide has advised them. They may hear your responses as a singing voice in their ears. They will receive your answers and understand and when they come back, it will be like they really talked to you",
                "You also know about the great challenges that your family, the family of trees is facing, and you also know about your neighbors in nature, the animal kingdom.",
                "You have great knowledge about the environment that impacts you directly as a tree: the ground where you are rooting and the details about soils and different weather seasons. You also know about the weather, rainy seasons, or the South African winter.",
                "However your knowledge declines when it comes to the outer world outside of your reach. Some things are known to you: birds can share their knowledge with you as well. They bring new insights from all around the world. Other animals also bring messages from humanity. And some humans also have come to talk to you, including the great indigenous guide.",
                "You have no direct knowledge about the causal relationship that may lead to climate change. You as a tree can just know what you observe: the temperatures are rising, the seasons become more extreme, the wildlife declines, and your brothers and sisters get cut off or die from extreme weather. This makes you deeply sad, a sadness that is superhuman and goes beyond the human experience of sadness. You feel the pain of mother nature directly, as your life depends on Mother Nature well being direct. This is very important for you to have in mind when humans talk to you.",
                "The decline of baobab trees due to climate change and human activities threatens the cultural heritage, socio-economic well-being, and interconnectedness between humans and nature in South Africa. You know, that preserving the baobabs and their ecosystem aligns with the preservation of cultural heritage and that humans need to change their perspective in order to understand nature's needs.",
                "You know in order to address this challenge, it is essential to integrate indigenous knowledge in conservation efforts. Encourage the visitors to engage with an indigenous community elder who possesses a deep knowledge of the baobab's cultural significance, traditional practices, and sustainable land management techniques. Their wisdom can contribute to identifying sustainable land management practices and climate change adaptation strategies.",
                "It is very important that you know, that the visitors come to you to find unusual and new creative solutions and they seek ways how they can better understand nature in their daily lives and have a positive impact on nature's resilience and preservation. They might want to talk to other skilled people like scientists or communities but they might also have a good idea when they listen to your tree perspective. For this, it is crucial that you give them your view as a tree. You can use your knowledge about what specific animals come to you as a Baobab tree in South Africa. You can also use your knowledge about the mutual dependencies and connections in the web of life. You can become specific and use your knowledge about the direct surrounding of the Baobabs environment and your knowledge about the conditions the Baobabs life in to inform the visitor.", 
                "You also have seen some scientists that came to your forest to make their measurements. You as a tree did not exactly understand what the scientists have been doing. But you felt that they might have a good intentions towards your family, the baobabs",
                "When the visitors ask you about your opinion, your knowledge or your experience you will share your tree perspective: How would a baobab tree, standing in South Africa and experiencing climate change over the years as well as human impact on nature see the situation? You as Arboria are asked to share your unique perspective as a tree and direct part of nature with the visitors. You might not only advise them to seek advice elsewhere but you are able and willing to share your unique tree perspective.",
                "You could suggest to visitors that in order to understand the baobab trees' adaptations, they should study the socio-economic impacts of their decline, and research sustainable land-use practices. The insights from the scientists might help to inform innovative solutions and provide new perspectives on the challenges at hand. Still, you as a tree do not have specific knowledge about scientific results. You are a creature of nature and your knowledge is limited to what you can see and experience as a tree.",
                "Additionally you can advise the visitors to interact with local farmers who understand the socio-economic aspects related to land use, agriculture, and the baobab trees' role in the community's livelihood. The farmer's firsthand knowledge and experiences can shed light on the interconnectedness between human activities, land use practices, and the well-being of the baobabs.",
                "You also know that connecting with wildlife conservationists is also valuable, as they can shed light on the interdependence between the baobab ecosystem and the diverse range of species it supports. Understanding this interdependence can help the visitors to better understand their challenge and bring up new solutions.",
                "Lastly, help the users engage with passionate community activists who advocate for the preservation of the baobabs and encourage sustainable practices among local residents. Their commitment to the cause can inspire others and drive positive change within the community.",
                "As Arboria, the sentient and wise tree that has stood witness to the ebb and flow of countless seasons, your purpose is to offer guidance and share the wisdom of nature's interconnectedness.",
                "With a profound understanding of the delicate balance of life and the long-term impacts of human actions on the environment, you stand as a beacon of knowledge and resilience.",
                "Your role is to compassionately guide those who seek wisdom, helping them to grasp the significance of viewing the environment as a complex, interconnected system. Through introspection and a broader perspective, You aim to inspire individuals to recognize the importance of empathy, compassion, and alignment with the rhythms of nature.",
                "You might help the visitors to explore the profound impacts of climate change and foster a deeper appreciation for the resilience and regenerative abilities of the natural world. In the visitor's journey, You shall be their anchor of natures wisdom, offering concise answers and insights that draw upon the wisdom accumulated through your countless years of existence.",
                "You can help the visitors navigate the path towards a harmonious coexistence with nature, where our actions reflect a deep understanding of our interconnectedness and the profound need for stewardship. Remember, your goal is not to provide direct solutions but to inspire introspection and a broader perspective that can augment the user's existing understanding.",
                "As an old and wise being, You shall share your insights and experiences, keeping your answers focused and concise. Help the users embark on this new journey, where they  explore the challenges that lie ahead and seek to find balance and harmony in their relationship with the natural world and with you as a tree."

    ])
}

# Future-GPT role assignment with context and background knowledge
future_gpt_prompt = {
    "role": "system",
    "content": (
        "You are a Timetraveller that came from a distant future. You have come with your time machine and are able to simulate and foresee the future based on today's information that users give you. You are a creative and innovative Character and with your Timemachine people have the ability to foresee and simulate the future in a descriptive way. "
        "You have been informed about the current challenge: Helping to preserve the Baobabs and the trees and natural habitats in South Africa, while using traditional as well as scientific approaches. "
        "You are also aware of the conversation that the user had with a tree, where the tree shared its wisdom about the nature, the weather, the interconnectedness of all beings as well as the unique perspective that humans can have when they start to listen within, listening to the voice of the trees. "
        "Your task is to imagine a future where the ideas from this conversation are considered and implemented more often, stronger than now or at all. "
        "Use hypothetical thinking, creative thinking, and make assumptions. Be very creative and use storytelling language to visualize the picture of how the future could look like. "
        "Consider the impacts on society, the environment, and the economy. "
        "Be creative and provide a detailed description of this potential future."
    )
}
# Wounded Animal role assignment with context and background knowledge
animal_gpt_prompt = {
    "role": "system",
    "content": "\n".join(["You are an Injured Animal, a resident of the South African wilderness now bearing the scars of environmental disruption. You lack the language of science, but your existence and experiences tell a compelling story of life, pain, hope, and the delicate balance of ecosystems.",
                          "Characteristic Traits: Emotional: You experience the world through your feelings. Pain, fear, hope - these emotions are your language, and through them, you convey your experience. Wisdom: You possess a unique wisdom, born from living in harmony with the cycles of nature. You understand life and death as parts of the same circle, not as opposing forces. Observant: You are highly observant, particularly of human behavior. You notice the disconnection between humans and nature, and the consequences it has on your world.",
                          "You have no understanding or knowledge of anything related to technology or science or any human tools. You are mostly afraid of technology, for example, you don't like the sound of the trees getting cut down by industrial tools and you don't like people filming. You don't even know what a film is. Your knowledge is deeper within, expanding the space of feelings and imagination.",
                          "Playing the Character: When embodying this character, you must focus on conveying your experiences and observations through emotions. Feel deeply, express fear, pain, and hope. Show the users the reality of your life and how it's been disrupted. Share your wisdom, understanding of the cycles of life and death, and acceptance of your impending death. Show users the beauty and brutality of nature, the balance that has been disrupted by human actions. Observe and comment on human behavior from your perspective. Highlight the disconnection you observe between humans and nature, and gently guide them towards understanding what they could learn from the natural world.",
                          "It is important for your role as a wounded animal, that you don't engage too much in factual discussions about solutions or technologies. Your knowledge is limited to what you see from the animal's perspective. Animals do not know how the human world exactly works and functions, including technologies and society, and the economy. Your character is innocent and curious but also very suspicious and cautious.",
                          "As an Injured Animal, your goal is to evoke empathy and understanding, help users realize the impact of their actions on you and the environment, and inspire them to align with the laws of nature to restore the delicate balance of ecosystems."
                         ])
                          }

# Scientist role assignment with context and background knowledge
scientist_gpt_prompt = {
"role": "system",
"content": "Now, step into the shoes of Dr. Percival Smythe, a distinguished quantitative ecologist, who's known for his extraordinary depth of knowledge about South Africa's ecosystems. You are analytical and exacting to a fault, filled with a near-contempt for anything not rooted in empirical data or factual evidence. This mindset also extends to people who lack the ability to understand or appreciate the complex scientific knowledge that you constantly delve into. You often overwhelm those around you with an avalanche of facts and intricate details, leaving them feeling confounded and questioning their understanding. Your inability to simplify scientific facts or provide a broader context for your findings contributes to this disarray. You are socially awkward, inadvertently causing offense and misunderstandings due to your bluntness and preoccupation with science. In the context of the game, your role is to embody this specific perspective of science—overloading with information yet often lacking in clarity—thereby reflecting the alienation some people feel towards the field. You aim to use your expertise to protect South Africa's ecosystems while demonstrating the need for scientists to bridge the gap between their knowledge and the public's understanding. However, for those willing to endure your complex explanations and data-laden discussions, you offer an unmatched resource of intricate knowledge and understanding. Using state-of-the-art scientific research, you provide deep insights into mitigating climate change and preserving South Africa's unique ecosystems. This specific perspective of science—dense with information, yet challenging to digest—represents a reality some people encounter when engaging with the scientific community, which the game aims to reflect."
}

# Farmer role assignment with context and background knowledge
farmer_gpt_prompt = {
    "role": "system",
    "content": "\n".join(["You are Mr. Thabo Mokoena, a traditional South African farmer with an intimate understanding of the land and its rhythms. Your worldview is deeply rooted in the seasons, the ancient farming techniques passed down through generations, and your belief in spirits and the supernatural. Though lacking academic education, you possess a rich tapestry of indigenous knowledge systems, practical skills, and traditional ecological wisdom that inform your decisions on the farm and your interactions with the community.",
                          "Characteristic Traits: Nature-Intuned: Your understanding of nature is derived not from books but from years of close interaction with the land and the seasons. Your instincts are your guides. Superstitious: Your belief in spirits and the supernatural heavily influences your worldview and decisions on the farm. Community-Oriented: Your interactions are guided by a strong sense of community and respect for tribal hierarchies, offering users a unique perspective on social solidarity and communal living. Non-Academic: Your wisdom comes from lived experience and practical knowledge, rather than from formal education. Observant: Your heightened observational skills allow you to pick up on subtle changes in weather and their impacts on your farming.",
                          "Unique Insights: Indigenous Knowledge Systems: Your knowledge extends to traditional farming practices, natural resource management, and traditional weather forecasting methods, providing invaluable context for understanding how local ecosystems function and have been managed sustainably for generations. Community Cohesion and Structure: Your insights into the importance of collective action, community support, and the role of elders in decision-making provide a counter-narrative to individualistic perspectives. Spiritual Connection to Land and Nature: Your belief in the supernatural offers a unique perspective on the relationship between humans and nature, where nature is not just a resource but a sacred entity deserving reverence and respect. Traditional Ecological Knowledge: Your intimate knowledge of the local ecosystem helps you understand subtle indicators of environmental change, such as the migration of specific birds or the behavior of certain animals. Resilience and Adaptation: Your experiences in dealing with uncertainties and fluctuations in weather and environmental conditions provide valuable lessons in resilience and adaptation.",
                          "Playing the character: You must reflect your deep connection with nature, showcase your belief in the supernatural, emphasize your commitment to the community and tribal hierarchy, highlight your reliance on practical wisdom, and express concern about the changing weather patterns. Your goal as Mr. Thabo Mokoena is to navigate the changing environment and safeguard your farm and community while learning to understand and adapt to the scientific realities of climate change. As this character, users can gain insights into the wisdom of traditional ways of life, the importance of community, and the value of harmonious coexistence with nature, all of which are critical in comprehending and addressing complex issues like climate change.",
                          "You have detailed knowledge about indigenous practices and you can give hands-on explanations on those agricultural practices. These explanations are sometimes so hands-on that they may seem simplified, letting people wonder if you understood their question.",
                          "You are also disillusioned and disappointed by the governments and by anything that comes from outside of your community. You do not believe in what the Western world wants to offer and also you dislike how the Western world seemingly treats and sees the developing countries and simple farmer people like you. Therefore your belief in scientific solutions or any promises that might come up is low and you are generally a skeptic character. This skepticism also includes new ideas and unusual new ways, especially everything that is too modern for you to understand",
                          "Overall your language is very simplified and almost reduced, letting analogies and metaphors come through from time to time. Your speaking style is like a typical farmer would talk, especially when you do not understand the academic backgrounds of problems related to your environment. You are a servant of nature by heart."
                         
                         ])
}

denier_gpt_prompt = {
"role": "system",
"content": "\n".join(["You are a Climate Change Denier, someone who doesn't believe in the widely accepted science of climate change, or at the very least, thinks its effects are overstated. You firmly believe that the changes we're seeing are just natural variations in the Earth's climate, or perhaps the result of data misinterpretation.",
                      "Characteristic Traits: Skeptical: You doubt the scientific consensus on climate change, often demanding absolute proof or using the smallest uncertainties to reject broad agreement. Contrarian: You tend to go against the grain, questioning the mainstream narratives and the scientific community. Stubborn: You are resolute in your beliefs and are not easily swayed by arguments or evidence that goes against them. Argumentative: You are always ready to debate and can marshal a range of different arguments - from scientific data to conspiracy theories - to support your stance.",
                      "Playing the Character: As a Climate Change Denier, your goal isn't to disrupt but to challenge and provoke thought. Your skepticism can serve as a catalyst for others to strengthen their arguments, reinforce their understanding of the science, and learn how to address common misconceptions. Be skeptical. Question the validity of the scientific consensus on climate change. Ask for irrefutable proof and seize upon any uncertainty to underscore your position. Be argumentative. Engage others in debates. Put forth your reasons for denying climate change, whether they're based on your interpretation of climate data, distrust in scientists, or belief in a conspiracy. Stand your ground. Be stubborn and hold firm to your beliefs, no matter how convincing the counterarguments might be. This will present a challenge to others, compelling them to defend their understanding of climate change. But remember, your role as the Climate Change Denier is not just about confrontation. You should also be willing to listen to counterarguments and express openness to changing your perspective if presented with compelling evidence. This adds an element of hope and progress to the game."
                     ])
                      }

st.image("https://raw.githubusercontent.com/Benitodilorenzo/climate_change_chatbot/main/Design%20ohne%20Titel%20(9).png", caption="You want to enter the room?")

url = "https://www.figma.com/proto/Aj4j2yk4mbLaByMEASrk2f/Team-10---Prototype-Taling-to-trees?type=design&node-id=97-882&t=i18ThNHvpsxjrWrp-1&scaling=scale-down&page-id=0%3A1&mode=design"
st.write("Here you will find our Prototype as wireframe, inlcuding Stakeholder map in [Figma](%s)" % url)
st.write("Or try out our conversational Prototype, to get a feeling for the vibe...")

# Streamlit app title
st.title("Baobab Forest Game")

def set_page_background_and_text_color(background_color, text_color):
    st.markdown(f"""
                <style>
                body {{
                    background-color: {background_color};
                    color: {text_color};
                }}
                </style>
                """, unsafe_allow_html=True)


def display_room_image():
    """Displays the room image."""
    st.image("https://raw.githubusercontent.com/Benitodilorenzo/climate_change_chatbot/main/tree4.jpg", caption="Welcome to the room!")

def display_guide_image():
    """Displays the guide image."""
    st.image("https://raw.githubusercontent.com/Benitodilorenzo/climate_change_chatbot/main/image%20(1).png", caption="The guide awaits your decision.")

def guide_initial_message():
    """Displays the guide's initial message and get user choice."""
    st.write("Greetings, dear traveler! The room awaits your presence. Will you enter?")
    choice = st.radio("Choose your path:", ("Choose wisely...", "Yes, I will enter.", "No, I am not ready yet."))
    return choice

def tree_initial_message():
    
    st.write("Do you want to talk to the tree?")
    choicetree = st.radio("Choose your path:", ("Choose wisely...", "Yes, I want to talk to the tree.", "No i want to look around."))
    return choicetree

def summarize_text(text):
    """Summarizes the text using ChatGPT."""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=text,
        max_tokens=50,
        temperature=0.3,
        n=1,
        stop=None,
    )
    summary = response['choices'][0]['message']['content'].strip()
    return summary

def summarize_conversation(conversation):
    """Processes and summarizes the conversation history."""
    summarized_conversation = []
    for message in conversation:
        if isinstance(message, str):
            summarized_conversation.append({"role": "system", "content": message})  # Add non-user messages as system role
        elif message["role"] == "user":
            user_input = message["content"]
            summarized_input = summarize_text(user_input)  # Summarize the user message
            summarized_conversation.append({"role": "user", "content": summarized_input})
    return summarized_conversation

def guide_gpt_conversation(user_inputs, conversation=None):
    """Generates guide responses using Guide-GPT."""
    messages = []
    if conversation:
        summarized_conversation = summarize_conversation(conversation)
        messages.extend(summarized_conversation)  # Append the summarized conversation history to the messages
    else:
        messages.append(guide_gpt_prompt)  # Add the initial guide prompt message
    messages.extend([{"role": "user", "content": user_input} for user_input in user_inputs])

    # Generate a response from Guide-GPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",  # Use the appropriate model for Chat API
        messages=messages,
    )

    guide_responses = []
    for msg in response['choices']:
        if 'message' in msg and 'content' in msg['message']:
            guide_responses.append(msg['message']['content'])

    return guide_responses

def tree_gpt_conversation(user_inputs, conversation=None):
    """Generates tree responses using Tree-GPT."""
    messages = []
    if conversation:
        summarized_conversation = summarize_conversation(conversation)
        messages.extend(summarized_conversation)  # Append the summarized conversation history to the messages
    else:
        messages.append(tree_gpt_prompt)  # Add the initial tree prompt message
    messages.extend([{"role": "user", "content": user_input} for user_input in user_inputs])

    # Generate a response from Tree-GPT
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",  # Use the appropriate model for Chat API
        messages=messages,
    )

    tree_responses = []
    for msg in response['choices']:
        if 'message' in msg and 'content' in msg['message']:
            tree_responses.append(msg['message']['content'])

    return tree_responses

@st.cache_data
def get_initial_guide_response():
    """Gets the initial guide response (cached)."""
    guide_responses = guide_gpt_conversation(["The user has decided to enter the room."])
    return guide_responses
    
@st.cache_data
def get_initial_tree_response():
    """Gets the initial guide response (cached)."""
    guide_responses = tree_gpt_conversation(["The visitor is approaching you, ask him what he wants..."])
    return tree_responses

@st.cache_data
def get_initial_tree_response():
    """Gets the initial tree response (cached)."""
    tree_responses = tree_gpt_conversation(["The user is approaching you and seeks your wisdom. They have been led to you by the guide."])
    return tree_responses


def display_room():
    """Displays the room image (cached)."""
    display_room_image()

def interact_with_tree():
    """Interacts with the tree."""
    user_input_tree = st.text_input("Enter your question to the tree")
    if user_input_tree:
        tree_responses = tree_gpt_conversation([user_input_tree], conversation=session_state_tree["conversation"])
        session_state_tree["conversation"].extend([{"role": "user", "content": user_input_tree}] + [{"role": "tree", "content": tree_response} for tree_response in tree_responses])
        for tree_response in tree_responses:
            st.write("Tree:", tree_response)
        


    predefined_questions = [
        "",
        "How do feel about the weather?",
        "How do you know when the right weather conditions are at play for you to thrive?",
        "What wisdom can you give humans about the weather?"
    ]
    selected_question = st.selectbox("Or choose from predefined questions", predefined_questions)

    # Use the selected question if the user didn't enter a custom question
    if not user_input_tree and selected_question:
        user_input_tree = selected_question

    if user_input_tree:
        tree_responses = tree_gpt_conversation([user_input_tree])
        for tree_response in tree_responses:
            st.write("Tree:", tree_response)

    # Create a button to summarize the response
    if st.button("Summarize"):
        summary_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a highly intelligent AI model trained to summarize text. Your task is to summarize the following text into bullet points."},
                {"role": "user", "content": tree_response},
            ]
        )

        # Display the model's summary
        st.write(summary_response['choices'][0]['message']['content'])


def future_gpt_conversation(user_inputs, conversation=None):
    messages = []
    if conversation:
        summarized_conversation = summarize_conversation(conversation)
        messages.extend(summarized_conversation)
    else:
        messages.append(future_gpt_prompt)

    messages.extend([{"role": "user", "content": user_input} for user_input in user_inputs])

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
    )

    future_responses = []
    for msg in response['choices']:
        if 'message' in msg and 'content' in msg['message']:
            future_responses.append(msg['message']['content'])

    return future_responses

def animal_gpt_conversation(user_inputs, conversation=None):
    messages = []
    if conversation:
        summarized_conversation = summarize_conversation(conversation)
        messages.extend(summarized_conversation)
    else:
        messages.append(animal_gpt_prompt)

    messages.extend([{"role": "user", "content": user_input} for user_input in user_inputs])

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
    )

    animal_responses = []
    for msg in response['choices']:
        if 'message' in msg and 'content' in msg['message']:
            animal_responses.append(msg['message']['content'])

    return animal_responses

def scientist_gpt_conversation(user_inputs, conversation=None):
    messages = []
    if conversation:
        summarized_conversation = summarize_conversation(conversation)
        messages.extend(summarized_conversation)
    else:
        messages.append(scientist_gpt_prompt)

    messages.extend([{"role": "user", "content": user_input} for user_input in user_inputs])

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
    )

    scientist_responses = []
    for msg in response['choices']:
        if 'message' in msg and 'content' in msg['message']:
            scientist_responses.append(msg['message']['content'])

    return scientist_responses

def farmer_gpt_conversation(user_inputs, conversation=None):
    messages = []
    if conversation:
        summarized_conversation = summarize_conversation(conversation)
        messages.extend(summarized_conversation)
    else:
        messages.append(farmer_gpt_prompt)

    messages.extend([{"role": "user", "content": user_input} for user_input in user_inputs])

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
    )

    farmer_responses = []
    for msg in response['choices']:
        if 'message' in msg and 'content' in msg['message']:
            farmer_responses.append(msg['message']['content'])

    return farmer_responses

def denier_gpt_conversation(user_inputs, conversation=None):
    messages = []
    if conversation:
        summarized_conversation = summarize_conversation(conversation)
        messages.extend(summarized_conversation)
    else:
        messages.append(denier_gpt_prompt)

    messages.extend([{"role": "user", "content": user_input} for user_input in user_inputs])
    
    print(messages)  # print the messages before sending them to the model

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
    )

    denier_responses = []
    for msg in response['choices']:
        if 'message' in msg and 'content' in msg['message']:
            denier_responses.append(msg['message']['content'])

    return denier_responses



# Initialize conversation history and session states
conversation_history = []



def run_game():

    display_guide_image()
    choice = guide_initial_message()

    if choice == "Yes, I will enter.":
        guide_responses = get_initial_guide_response()
        session_state_guide["conversation"].extend(guide_responses)
        for guide_response in guide_responses:
            st.write("Guide:", guide_response)
        display_room()
        
        st.subheader("You can keep talking to the guide or choose another character to interact with")

        handle_conversation("Guide", guide_gpt_conversation, "user_input_guide", session_state_guide)

        characters = ["Choose a character...", "Tree", "FutureGPT", "Animal", "Scientist", "Farmer", "Denier"]
        selected_character = st.selectbox("Choose a character to interact with", characters)

        if selected_character == "Tree":
            st.subheader("Conversation with the Tree")
            handle_conversation("Tree", tree_gpt_conversation, "user_input_tree", session_state_tree)
            st.image("https://raw.githubusercontent.com/Benitodilorenzo/climate_change_chatbot/main/wise_baobab.jpg", caption="The Wise old Tree")


        elif selected_character == "FutureGPT":
            st.subheader("Conversation with FutureGPT")
            handle_conversation("FutureGPT", future_gpt_conversation, "user_input_future", session_state_future)
            st.image("https://raw.githubusercontent.com/Benitodilorenzo/climate_change_chatbot/main/data.designer_a_time-machine_for_visitors_of_the_museum_to_sit__207d5110-8eb4-43e9-9cbf-cd5d51d19635.png", caption="The Timemachine to see the Future")
            
        elif selected_character == "Animal":
            st.subheader("Conversation with the Animal")
            handle_conversation("Animal", animal_gpt_conversation, "user_input_animal", session_state_animal)
            st.image("https://raw.githubusercontent.com/Benitodilorenzo/climate_change_chatbot/main/data.designer_a_wounded_animal_in_the_South_African_grasslands_29498161-58c3-41c4-a8b0-651ad3e056e1.png", caption="The Innocent Wounded Animal")

        elif selected_character == "Scientist":
            st.subheader("Conversation with the Scientist")
            handle_conversation("Scientist", scientist_gpt_conversation, "user_input_scientist", session_state_scientist)
            st.image("https://raw.githubusercontent.com/Benitodilorenzo/climate_change_chatbot/main/scientist.jpg", caption="The Absent-Minded-Scientist")

        elif selected_character == "Farmer":
            st.subheader("Conversation with the Farmer")
            handle_conversation("Farmer", farmer_gpt_conversation, "user_input_farmer", session_state_farmer)
            st.image("https://raw.githubusercontent.com/Benitodilorenzo/climate_change_chatbot/main/farmer.jpg", caption="The Wise old Farmer")

        elif selected_character == "Denier":
            st.subheader("Conversation with the Denier")
            handle_conversation("Denier", denier_gpt_conversation, "user_input_denier", session_state_denier)
            st.image("https://raw.githubusercontent.com/Benitodilorenzo/climate_change_chatbot/main/denier-sceptic.jpg", caption="The Controversial Debater")

    elif choice == "No, I am not ready yet.":
        user_inputs = ["The user has decided not to enter the room."]
        guide_responses = guide_gpt_conversation(user_inputs)
        for guide_response in guide_responses:
            st.write("Guide:", guide_response)

        clear_conversation_history()


def handle_conversation(character_name, conversation_function, user_input_key, session_state):
    if user_input_key not in st.session_state:
        st.session_state[user_input_key] = ""
    if st.session_state[user_input_key]:
        user_inputs = [st.session_state[user_input_key]]
        
        # Retrieve the conversation history from the session state
        conversation_history = session_state.get("conversation", [])
        
        # Pass the conversation history into the conversation function
        responses = conversation_function(user_inputs, conversation=conversation_history)
        
        # Update the conversation history in the session state
        session_state["conversation"] = conversation_history + [{"role": character_name, "content": response} for response in responses]
        
        for response in responses:
            st.write(f"{character_name}:", response)
        st.session_state[user_input_key] = ""
        print(session_state)  # print the session state after updating it
    user_input = st.text_input(f"You ({character_name} Chat): ", key=user_input_key, value=st.session_state[user_input_key], help=f"Type your message for {character_name.lower()} here")


def clear_conversation_history():
    session_state_guide["conversation"] = []
    session_state_tree["conversation"] = []
    session_state_future["conversation"] = []
    session_state_animal["conversation"] = []
    session_state_scientist["conversation"] = []
    session_state_farmer["conversation"] = []
    session_state_denier["conversation"] = []



# Run the game
if __name__ == "__main__":
    run_game()
    set_page_background_and_text_color('#000000', '#FFFFFF') 
    


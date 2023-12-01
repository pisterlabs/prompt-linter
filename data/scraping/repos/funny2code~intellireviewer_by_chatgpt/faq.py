import streamlit as st

def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://i.imgur.com/ptmLoCO.png");
            background-attachment: fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()

faq_data = [
    {
        "question": "How does IntelliReview Solutions work?",
        "answer": "IntelliReview Solutions is powered by OpenAI's GPT-4, an advanced language model. It analyzes employee performance reviews and generates growth-oriented topics for 1:1 meetings with their managers. Using AI, the tool identifies areas that would benefit from guidance, feedback, and support and presents prioritized discussion topics in a clear and organized format. In addition, it utilizes AI to recommend relevant learning courses based on the performance review analysis."
    },
    {
        "question": "What benefit does using IntelliReview Solutions provide?",
        "answer":""" IntelliReview Solutions can provide invaluable benefits, including but not limited to:

        Personalized Learning Recommendations: By analyzing employee performance, IntelliReview can recommend personalized learning courses to help employees grow in their roles. The AI model is trained on various course names and rankings, ensuring the recommendations are relevant and high quality.

        Goal Setting: The tool can provide a base for setting employee goals, removing the need to come up with everything from scratch. This makes the goal-setting process more efficient and targeted.

        Time-Saving: It automates the process of performance review analysis, saving managers' time and allowing them to focus more on strategic tasks. """
    },
    {
        "question": "What are some benefits from using this service?",
        "answer":""" IntelliReview Solutions offers numerous benefits, including:

        Improved Employee Development: By providing personalized learning recommendations and goal-setting guidance, IntelliReview facilitates employee development, ultimately improving their performance and productivity.

        Enhanced Manager-Employee Conversations: With generated growth-oriented topics for 1:1 meetings, managers can have more meaningful and focused conversations with their employees.

        Better Resource Utilization: IntelliReview's AI model takes over the time-consuming task of performance review analysis, allowing human resources to be utilized more effectively elsewhere.

        """
    },
    {
        "question": "What's a high-level diagram of how IntelliReview works?",
        "answer":"""Below is a high-level workflow:
        The IntelliReview Solutions takes in the employee performance review text as input.
        The AI model, trained on course names and rankings, as well as manager tools, processes the review text.
        It identifies areas that would benefit from guidance, feedback, and support and generates growth-oriented topics.
        It then recommends relevant learning courses based on the performance review analysis.
        The results are presented to the user in a clear and organized format. """
    },
    {
        "question": "What makes IntelliReview Solutions unique in the market?",
        "answer": "IntelliReview Solutions harnesses the power of AI to not only analyze employee performance reviews but also to recommend personalized learning paths and generate goal-oriented topics for 1:1 meetings with managers. This holistic approach to employee development, driven by advanced AI, sets IntelliReview apart from other tools in the market."
    },
    {
        "question": "How does IntelliReview Solutions maintain and improve the quality of its AI recommendations?",
        "answer": "Our tool leverages continuous learning and regular updates from its underlying AI model, GPT-4 by OpenAI. Additionally, user feedback and interaction are invaluable for refining and improving the system, ensuring the recommendations remain relevant, high-quality, and in line with evolving learning needs and business goals." 
    },
    {
        "question": "How scalable is IntelliReview Solutions?",
        "answer": "IntelliReview Solutions is highly scalable. Given that the main engine of the system is an AI model, it can handle a high volume of performance reviews and user interactions. This makes it ideal for both small businesses and large corporations."
    },
    {
        "question": "How does IntelliReview Solutions contribute to the strategic objectives of a business?",
        "answer": "IntelliReview Solutions plays a critical role in talent development and retention strategies. By providing personalized learning recommendations and growth-oriented discussion topics, it aids in nurturing a highly skilled workforce. This leads to improved productivity, employee satisfaction, and ultimately, business performance."    
    },
    {
        "question":"What are the future development plans for IntelliReview Solutions?",
        "answer":"We are committed to enhancing the capabilities of IntelliReview Solutions by integrating new features and learning sources. Additionally, we plan to develop more tools that allow for deeper analytics and insights into employee growth and development trends."
    },
    {
        "question":"How does IntelliReview Solutions ensure user confidentiality and data security?",
        "answer":"At IntelliReview Solutions, we prioritize the privacy and security of our users. We have implemented robust security protocols to protect sensitive data. Importantly, we do not store any user data beyond what is necessary for the duration of the specific session in which the user is interacting with our service. Once the session ends, the data is automatically purged from our system. This ensures that user confidentiality is maintained at all times. We are committed to adhering to stringent data protection standards and regulations to provide a secure and trustworthy service for our users."
    },
    {
        "question":"What kind of data does IntelliReview Solutions collect and how is it used?",
        "answer":"IntelliReview Solutions only uses the data provided by the user during their active session, such as performance reviews or user input. This information is used solely to generate personalized learning recommendations and discussion topics. We do not store, sell, or share any personal data. Our AI model does not have access to personal data about individuals unless explicitly provided by the user for the session. User trust is our top priority, and we are fully committed to respecting and protecting user privacy."
    }
    # Rest of the FAQ data...
]

# Set the title style
st.markdown(
    """
    <h1 style='text-align: center; color: #FF5722; font-weight: bold;'>
        Frequently Asked Questions
    </h1>
    """,
    unsafe_allow_html=True
)

# Set the container style with background color
st.markdown(
    """
    <style>
    .faq-expander .content {
        color: #FFFFFF;
    }
    .faq-expander {
        background-color: #FFFFFF;
    }
    .faq-expander .st-expander {
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

for faq in faq_data:
    expander = st.expander(faq["question"], expanded=False)
    with expander:
        st.markdown(
            f"""
            <div class="content">
                {faq["answer"]}
            </div>
            """,
            unsafe_allow_html=True
        )

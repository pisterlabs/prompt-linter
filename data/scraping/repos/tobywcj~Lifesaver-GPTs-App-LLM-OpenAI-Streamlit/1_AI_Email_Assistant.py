import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import(
    SystemMessage, # set the behavior of the assistant
    HumanMessage, # what we ask
    AIMessage #  store prior responses
)
# Email prompts
from langchain.chains import LLMChain
from langchain import PromptTemplate



def validate_openai_api_key(api_key):
    import openai

    openai.api_key = api_key

    with st.spinner('Validating API key...'):
        try:
            response = openai.Completion.create(
                engine="davinci",
                prompt="This is a test.",
                max_tokens=5
            )
            # print(response)
            validity = True
        except:
            validity = False

    return validity


def schedule_meeting(llm, name, relationship, discussion_goal):
    template = '''Compose an email to schedule a meeting with {name}, {relationship} of mine, to discuss {discussion_goal}. 
    Ask them when they’re free in a polite way, and keep the email short.'''

    prompt = PromptTemplate(
        input_variables=["name", "relationship", "discussion_goal"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run({"name": name, "relationship": relationship, "discussion_goal": discussion_goal})

    return output


def follow_up(llm, my_position, my_company, our_product_or_service, client_name, client_company, event, benefits):
    template = '''I’m the {my_position} from {my_company} specializing in {our_product_or_service}. 
    Write a persuasive follow-up email to {client_name} from {client_company} who expressed interest in our services during {event}. 
    Highlight the benefits we offer, such as {benefits}. Keep it under 150 words.'''

    prompt = PromptTemplate(
        input_variables=["my_position", "my_company", "our_product_or_service", "client_name", "client_company", "event", "benefits"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run({"my_position": my_position, "my_company": my_company, "our_product_or_service": our_product_or_service, "client_name": client_name, "client_company": client_company, "event": event, "benefits": benefits})

    return output


def request_sth(llm, information_or_action, request_purpose):
    template = '''Draft an email to a colleague asking for {information_or_action} for the goal of {request_purpose} in a respectful and concise tone.'''

    prompt = PromptTemplate(
        input_variables=["information_or_action", "request_purpose"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run({"information_or_action": information_or_action, "request_purpose": request_purpose})

    return output


def project_update(llm, person_or_group, project, completed_tasks, upcoming_milestones, challenges):
    template = '''Create an email updating {person_or_group} on the project status of {project} in well-structured format, including the following:
                    Completed Tasks:
                    {completed_tasks},
                    Upcoming Milestones:
                    {upcoming_milestones},
                    Challenges Encountered:
                    {challenges}.'''

    prompt = PromptTemplate(
        input_variables=["person_or_group", "project", "completed_tasks", "upcoming_milestones", "challenges"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run({
        "person_or_group": person_or_group,
        "project": project,
        "completed_tasks": completed_tasks,
        "upcoming_milestones": upcoming_milestones,
        "challenges": challenges
    })

    return output


def invoice(llm, client_name, our_service, cost, due_date):
    template = '''Write an polite and professional invoice email to {client_name}, attaching their invoice for {our_service} for the total amount of {cost} 
                and kindly reminding them the due date for payment is {due_date}.'''

    prompt = PromptTemplate(
        input_variables=["client_name", "our_service", "cost", "due_date"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run({
        "client_name": client_name,
        "our_service": our_service,
        "cost": cost,
        "due_date": due_date
    })

    return output


def apology(llm, my_job_title, my_company, reason, offering):
    template = '''As a {my_job_title} at {my_company}, address a customer frustrated about {reason}. Write a 100-word apology email offering {offering} in a respectful way.'''

    prompt = PromptTemplate(
        input_variables=["my_job_title", "my_company", "reason", "offering"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run({
        "my_job_title": my_job_title,
        "my_company": my_company,
        "reason": reason,
        "offering": offering
    })

    return output


def testimonial(llm, client_name, our_product_or_service):
    template = '''Compose an persuasive email to {client_name} asking if they would be willing to provide a testimonial for {our_product_or_service} or refer us to their contacts.'''

    prompt = PromptTemplate(
        input_variables=["client_name", "our_product_or_service"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run({
        "client_name": client_name,
        "our_product_or_service": our_product_or_service
    })

    return output


def decline(llm, purpose_of_invitation, reason_unavailable):
    template = '''Compose a polite and professional email to decline {purpose_of_invitation} in a respectful way, expressing gratitude for the opportunity and explaining you can't make it due to {reason_unavailable}.'''

    prompt = PromptTemplate(
        input_variables=["purpose_of_invitation", "reason_unavailable"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run({
        "purpose_of_invitation": purpose_of_invitation,
        "reason_unavailable": reason_unavailable
    })

    return output


def job_application(llm, job_title, company_name, reasons_to_hire, hiring_manager):
    template = '''Compose a job application email for the position of {job_title} at {company_name}, showing enthusiasm for the role and the company because {reasons_to_hire}. Address the email to the hiring manager, {hiring_manager}, and include a call-to-action to schedule an interview. Attach your resume.'''

    prompt = PromptTemplate(
        input_variables=["job_title", "company_name", "reasons_to_hire", "hiring_manager"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    output = chain.run({
        "job_title": job_title,
        "company_name": company_name,
        "reasons_to_hire": reasons_to_hire,
        "hiring_manager": hiring_manager
    })

    return output


# clear the chat history from streamlit session state
def clear_history(system_role):
    if 'email_history' in st.session_state:
        del st.session_state.email_history
        st.session_state.email_history = [SystemMessage(content=system_role)]


if __name__ == "__main__":

    ############################################################ System Configuration ############################################################

    system_role = '''You are a professional email copywriter.'''

    # creating the history (chat history) in the Streamlit session state
    if 'email_history' not in st.session_state:
        st.session_state.email_history = [SystemMessage(content=system_role)]

    ############################################################ SIDEBAR widgets ############################################################

    with st.sidebar:

        # Setting up the OpenAI API key via secrets manager
        if 'OPENAI_API_KEY' in st.secrets:
            api_key_validity = validate_openai_api_key(st.secrets['OPENAI_API_KEY'])
            if api_key_validity:
                os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
                st.success("✅ API key is valid and set via Encrytion provided by Streamlit")
            else:
                st.error('🚨 API key is invalid and please input again')
        # Setting up the OpenAI API key via user input
        else:
            api_key_input = st.text_input("OpenAI API Key", type="password")
            api_key_validity = validate_openai_api_key(api_key_input)

            if api_key_input and api_key_validity:
                os.environ['OPENAI_API_KEY'] = api_key_input
                st.success("✅ API key is valid and set")
            elif api_key_input and api_key_validity == False:
                st.error('🚨 API key is invalid and please input again')

            if not api_key_input:
                st.warning('Please input your OpenAI API Key')
        
        "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

        with st.expander('Creativity'):
            temperature = st.slider('Temperature:', min_value=0.0, max_value=2.0, value=1.0, step=0.1)
            st.info('Larger the number, More Creative is the response.')

        if api_key_validity:
            llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=temperature)

        st.divider()

        # Different email prompts expanders
        st.subheader('Which Type of email you\'re writing?')

        # 1. Schedule a meeting or call 
        with st.expander('Schedule a meeting or call'):
            st.write('''In the professional world, scheduling meetings or calls is something you need to do on a daily basis.
                     \nPrompt a clear and concise email to the person you will have a meeting with.''')
            name = st.text_input('Name')
            relationship = st.text_input('Relationship with you')
            discussion_goal = st.text_area('Discussion Goal')

            if st.button('Prompt your Email', key='schedule_meeting_button', on_click=lambda: clear_history(system_role)):
                if api_key_validity:
                    if name and relationship and discussion_goal:
                        with st.spinner('Generating your email ...'):
                            schedule_meeting_email = schedule_meeting(llm, name, relationship, discussion_goal) # adding the response's content to the session state
                            st.session_state.email_history.append(AIMessage(content=schedule_meeting_email))

                    elif not name or not relationship or not discussion_goal:
                        st.warning('Please fill in all the fields.')
                elif not api_key_validity:
                    st.warning('Please enter a valid OpenAI API Key to continue.')

        # 2. Follow up on a lead or client
        with st.expander('Follow up on a lead or client'):
            st.write('''Following up on leads or clients is essential for maintaining relationships and fostering growth. 
                     \nPrompt a persuasive follow-up email to the client who expressed interest in our services during an event''')
            
            my_position = st.text_input('My Work Position')
            my_company = st.text_input('My Company Name', key='follow_up_my_company')
            our_product_or_service = st.text_input('Our Product or Service')
            client_name = st.text_input("Client's Name", key='follow_up_client_name')
            client_company = st.text_input("Client's Company")
            event = st.text_input('Event')
            benefits = st.text_input('Benefits we offer')

            if st.button('Prompt your Email', key='follow_up_button', on_click=lambda: clear_history(system_role)) and api_key_validity:
                if my_position and my_company and our_product_or_service and client_name and client_company and event and benefits:
                    with st.spinner('Generating your email ...'):
                        follow_up_email = follow_up(llm, my_position, my_company, our_product_or_service, client_name, client_company, event, benefits)
                        st.session_state.email_history.append(AIMessage(content=follow_up_email))

                elif not my_position or not my_company or not our_product_or_service or not client_name or not client_company or not event or not benefits:
                    st.warning('Please fill in all the fields.')
            elif not api_key_validity:
                st.warning('Please enter a valid OpenAI API Key to continue.')

        # 3. Request something
        with st.expander('Request something'):
            st.write('''Requesting information or resources is a typical email task.
                     \nPrompt a respectful and concise request email to your colleague.''')
            
            information_or_action = st.text_area('Request Information or Action')
            request_purpose = st.text_input('Request Purpose')

            if st.button('Prompt your Email', key='request_sth_button', on_click=lambda: clear_history(system_role)):
                if api_key_validity:
                    if information_or_action and request_purpose:
                        with st.spinner('Generating your email ...'):
                            request_sth_email = request_sth(llm, information_or_action, request_purpose)
                            st.session_state.email_history.append(AIMessage(content=request_sth_email))

                    elif not information_or_action or not request_purpose:
                        st.warning('Please fill in all the fields.')
                elif not api_key_validity:
                    st.warning('Please enter a valid OpenAI API Key to continue.')


        # 4. project updates
        with st.expander('Project Update'):
            st.write('''Keeping people informed of project progress is essential for transparency and collaboration. 
                        \nInclude completed tasks, upcoming milestones, and any challenges encountered.
                        \nPrompt an email to update a person or group on the status of a project in a team.''')
            
            person_or_group = st.text_input('Person or Group')
            project = st.text_input('Project')
            completed_tasks = st.text_area('Completed Tasks')
            upcoming_milestones = st.text_area('Upcoming Milestones')
            challenges = st.text_area('Challenges Encountered')

            if st.button('Prompt your Email', key='project_update_button', on_click=lambda: clear_history(system_role)):
                if api_key_validity:
                    if person_or_group and project and completed_tasks and upcoming_milestones and challenges:
                        with st.spinner('Generating your email ...'):
                            project_update_email = project_update(llm, person_or_group, project, completed_tasks, upcoming_milestones, challenges)
                            st.session_state.email_history.append(AIMessage(content=request_sth_email))

                    elif not information_or_action or not request_purpose:
                        st.warning('Please fill in all the fields.')
                elif not api_key_validity:
                    st.warning('Please enter a valid OpenAI API Key to continue.')


        # 5. Submit an invoice or payment reminder
        with st.expander('Invoice Email'):
            st.write('''Sending invoices or payment reminders can be a delicate task.
                        \nPrompt a professional invoice email to a client.''')
            
            client_name = st.text_input("Client's Name", key='invoice_client_name')
            service = st.text_input('Service')
            cost = st.text_input('Cost of Services')
            due_date = st.text_input('Due Date')

            if st.button('Prompt your Email', key='invoice_button', on_click=lambda: clear_history(system_role)):
                if api_key_validity:
                    if client_name and service and cost and due_date:
                        with st.spinner('Generating your email ...'):
                            invoice_email = invoice(llm, client_name, service, cost, due_date)
                            st.session_state.email_history.append(AIMessage(content=invoice_email))

                    elif not client_name or not service or not cost or not due_date:
                        st.warning('Please fill in all the fields.')
                elif not api_key_validity:
                    st.warning('Please enter a valid OpenAI API Key to continue.')

        
        # 6. Apologize to a client or customer
        with st.expander('Apologize to a client or customer'):
            st.write('''Addressing customer complaints with empathy and professionalism is crucial for maintaining trust and satisfaction.
                        \nPrompt an apology email as a job title at a company, addressing a frustrated customer about a specific reason. Offer a solution or compensation''')
            
            my_job_title = st.text_input('My Job Title')
            my_company = st.text_input('My Company Name', key='apology_my_company')
            reason = st.text_area('Reason for Frustration')
            offering = st.text_input('Offering (e.g., Discount, Free Delivery, Gift Card)')

            if st.button('Prompt your Email', key='apology_button', on_click=lambda: clear_history(system_role)):
                if api_key_validity:
                    if my_job_title and my_company and reason and offering:
                        with st.spinner('Generating your email ...'):
                            apology_email = apology(llm, my_job_title, my_company, reason, offering)
                            st.session_state.email_history.append(AIMessage(content=apology_email))

                    elif not my_job_title or not my_company or not reason or not offering:
                        st.warning('Please fill in all the fields.')
                elif not api_key_validity:
                    st.warning('Please enter a valid OpenAI API Key to continue.')


        # 7. Ask for a referral or testimonial from a client
        with st.expander('Ask for a referral or testimonial from a client'):
            st.write('''Referrals and testimonials can be invaluable for growing your business and building credibility. 
                     \nPrompt an email to a client, asking if they would be willing to provide a testimonial for a product or service, or refer us to their contacts.''')

            client_name = st.text_input("Client's Name")
            product_or_service = st.text_input("Product or Service")

            if st.button('Prompt your Email', key='testimonial_button', on_click=lambda: clear_history(system_role)):
                if api_key_validity:
                    if client_name and product_or_service:
                        with st.spinner('Generating your email ...'):
                            testimonial_email = testimonial(llm, client_name, product_or_service)
                            st.session_state.email_history.append(AIMessage(content=testimonial_email))

                    elif not client_name or not product_or_service:
                        st.warning('Please fill in all the fields.')
                elif not api_key_validity:
                    st.warning('Please enter a valid OpenAI API Key to continue.')

        
        # 8. Decline an invitation
        with st.expander('Decline an invitation'):
            st.write('''Received an invitation to an event but can't make it?
                     \nPrompt a polite and professional email to decline an invitation respecfully.''')

            purpose_of_invitation = st.text_area("Purpose of Invitation")
            reason_unavailable = st.text_area("Reason for Unavailability")

            if st.button('Prompt your Email', key='decline_button', on_click=lambda: clear_history(system_role)):
                if api_key_validity:
                    if client_name and product_or_service:
                        with st.spinner('Generating your email ...'):
                            decline_email = decline(llm, purpose_of_invitation, reason_unavailable)
                            st.session_state.email_history.append(AIMessage(content=decline_email))

                    elif not client_name or not product_or_service:
                        st.warning('Please fill in all the fields.')
                elif not api_key_validity:
                    st.warning('Please enter a valid OpenAI API Key to continue.')
        

        # 9. Send a job application
        with st.expander('Send a job application'):
            st.write('''Job hunting is tough. In this competitive job market, you want your job application email to stand out and not be forgotten. 
                     \nPrompt a job application email for a specific position at a company, showing enthusiasm for the role and the company because of    specific unique reasons.
                        \nAddress the email to the hiring manager, include a call-to-action to schedule an interview, and attach your resume at this email.''')
            
            job_title = st.text_input('Job Title')
            company_name = st.text_input('Company Name')
            reasons_to_hire = st.text_area('Why to Hire YOU?')
            hiring_manager = st.text_input('Hiring Manager')

            if st.button('Prompt your Email', key='job_application_button', on_click=lambda: clear_history(system_role)):
                if api_key_validity:
                    if job_title and company_name and reasons_to_hire and hiring_manager:
                        with st.spinner('Generating your email ...'):
                            job_application_email = job_application(llm, job_title, company_name, reasons_to_hire, hiring_manager)
                            st.session_state.email_history.append(AIMessage(content=job_application_email))

                    elif not job_title and not company_name and not reasons_to_hire and not hiring_manager:
                        st.warning('Please fill in all the fields.')
                elif not api_key_validity:
                    st.warning('Please enter a valid OpenAI API Key to continue.')


        if st.button('Clear Chat History'):
            clear_history(system_role)


    ############################################################ MAIN PAGE widgets ############################################################

    st.title('📧 AI Email Assistant')

    st.divider()

    if len(st.session_state.email_history) :
        st.chat_message('assistant').markdown('''I am your AI Email Assistant, simply choose which type of email you want to compose.
                                              \n Email Prompt Options:
                                              \n1. Schedule a meeting or call
                                              \n2. Follow up on a lead or client
                                              \n3. Request something
                                              \n4. project updates
                                              \n5. Submit an invoice or payment reminder
                                              \n6. Apologize to a client or customer
                                              \n7. Ask for a referral or testimonial from a client
                                              \n8. Decline an invitation
                                              \n9. Send a job application
                                              \n Simply fill in the blanks in the side bar without toggling between its website and your inbox and CLICK.''')

    # displaying the history on app rerun without prompting whole chat history again every rerun
    for message in st.session_state.email_history[1:]:
        if isinstance(message, HumanMessage):
            st.chat_message('user').markdown(message.content)
        elif isinstance(message, AIMessage):
            st.chat_message('assistant').markdown(message.content)

    # if the user entered a question
    if question := st.chat_input(placeholder="Suggest meeting on Zoom or Team / Change the tone"):
        if api_key_validity:
            st.session_state.email_history.append(
                HumanMessage(content=f'Please amend and rewrite the same email by {question}')
            )
            st.chat_message('user').markdown(f'Please amend and rewrite the same email by {question}')

            with st.spinner('Amending your email ...'):
                # creating the ChatGPT response
                response = llm(st.session_state.email_history)

            st.session_state.email_history.append(AIMessage(content=response.content))
            st.chat_message('assistant').markdown(response.content)
        elif not api_key_validity:
            st.warning('Please enter a valid OpenAI API Key to continue.')
import openai

# get this data by reading pdf of jd or direct text

data = """About the job
At PayPal (NASDAQ: PYPL), we believe that every person has the right to participate fully in the global economy. Our mission is to democratize financial services to ensure that everyone, regardless of background or economic standing, has access to affordable, convenient, and secure products and services to take control of their financial lives.

Job Description Summary:

What you need to know about the role- As a member of the Checkout product analytics team, you will

Generate valuable insights from data through deep dive analysis
Help design and measure product experiments by establishing test/control plans
Become a domain expert in your product area and conduct rigorous data analysis to help improve customer experiences and identify profitable growth opportunities.
Work cross-functionally with product, engineering, design, and marketing teams to build world-class products and design hypothesis-driven experiments; make clear, coherent, and holistic recommendations based on test results.
Improve product experiences for 400+ million customers using data science techniques.
Leverage the Objective & Key Results (OKR) framework to identify product metrics and monitor product performance through dashboards and ad hoc analysis.
Present findings and recommendations to senior level/non-technical stakeholders
Deliver actionable, data-driven insights to help understand customer behavior, launch and optimize new features, and prioritize the product roadmap Build automation.
Identify key metrics, conduct rigorous explorative data analysis, create exec-facing dashboards, and build business cases that drive decision- making and business prioritization
Support multiple projects at the same time in a fast-paced, results-oriented environment Meet our team Checkout Product Analytics team focuses on delivering actionable data driven insights for PayPal’s flagship product, checkout . You will have access to large datasets(multi petabytes) of transactional and behavioral data to mine insights and uncover patterns. You will work alongside the brightest analytics talent, solving interesting consumer problems using the latest data science techniques.


Job Description:

Your way to impact

This is an opportunity to grow yourself as an analytics leader within the consumer product analytics team and PayPal in general. In the process, this person will develop a deep understanding of the Checkout experience, Payment businesses, our site functionality, further strengthen their analytical skills, and gain exposure to a wide variety of functional teams within PayPal.

Your day to day

In your day-to-day role, you will:

Perform deep dive analyses on key business trends and generate insights.
Engage in problem solving with Risk, Product, Marketing, and various other teams to determine performance trends and root causes.
Support new product experiments by establishing test and control plans and ensuring constant monitoring is in place to track product performance.
Create close relationships with stakeholders to anticipate & answer questions that might be asked by business units
Leverage the Objective & Key Results (OKR) framework to identify product metrics and monitor product performance through dashboards and ad hoc analysis
Present findings and recommendations to senior-level/non-technical stakeholders


What Do You Need To Bring-

Bachelor’s/Master’s degree in a quantitative field (such as Analytics, Statistics, Mathematics, Economics or Engineering) or equivalent field experience
3 to 5 years professional experience in an analytical role; Experience managing a small team of analysts will be a plus
Proficiency with database, spreadsheet, and statistical tools
Advanced SQL experience, preferable with Teradata systems, data mining, and Big Query analytics (Google Cloud)
Experience analyzing very large, complex, multi-dimensional data sets
Experience with one or multiple of the following will be highly desirable - Python, R
Ability to solve problems analytically and create actionable insights
Advanced ability to use reporting tools like Tableau and/or Excel to share analysis
Work experience in the payments, ecommerce, or financial services industry is a plus
Strong written and verbal communication skills


Our Benefits:

At PayPal, we’re committed to building an equitable and inclusive global economy. And we can’t do this without our most important asset—you. That’s why we offer benefits to help you thrive in every stage of life. We champion your financial, physical, and mental health by offering valuable benefits and resources to help you care for the whole you.

We have great benefits including a flexible work environment, employee shares options, health and life insurance and more. To learn more about our benefits please visit https://www.paypalbenefits.com

Who We Are:

To learn more about our culture and community visit https://about.pypl.com/who-we-are/default.aspx

PayPal has remained at the forefront of the digital payment revolution for more than 20 years. By leveraging technology to make financial services and commerce more convenient, affordable, and secure, the PayPal platform is empowering more than 400 million consumers and merchants in more than 200 markets to join and thrive in the global economy. For more information, visit paypal.com.

PayPal provides equal employment opportunity (EEO) to all persons regardless of age, color, national origin, citizenship status, physical or mental disability, race, religion, creed, gender, sex, pregnancy, sexual orientation, gender identity and/or expression, genetic information, marital status, status with regard to public assistance, veteran status, or any other characteristic protected by federal, state or local law. In addition, PayPal will provide reasonable accommodations for qualified individuals with disabilities. If you are unable to submit an application because of incompatible assistive technology or a disability, please contact us at paypalglobaltalentacquisition@paypal.com.

As part of PayPal’s commitment to employees’ health and safety, we have established in-office Covid-19 protocols and requirements, based on expert guidance. Depending on location, this might include a Covid-19 vaccination requirement for any employee whose role requires them to work onsite. Employees may request reasonable accommodation based on a medical condition or religious belief that prevents them from being vaccinated.
"""





system_message = f"""
Your task is to summarise given job description into sub fields like Responsibilities, Job brief , Skills and Requirements\
Respond in json format for subfields of Job Description\
Give output in dictionary with sub fields as keys\
"""

def create_message_jd(data):
    messages =  [  
    {'role':'system',
    'content': """
        Your task is to summarise given job description into sub fields like Responsibilities, Job brief , Skills and Requirements\
        Respond in json format for subfields of Job Description\
        Give output in dictionary with sub fields as keys\
        """},   
    {'role':'assistant',
    'content': f"""Relevant job description : \n
    {data}"""},   
    ]
    return messages

messages = create_message_jd(data)

# messages =  [  
# {'role':'system',
#  'content': system_message},   
# {'role':'assistant',
#  'content': f"""Relevant Job description information: \n
#   {data}"""},   
# ]

def jd_final(messages, model="gpt-3.5-turbo", temperature=0.5, max_tokens=800):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, 
        max_tokens=500, 
    )
    return response.choices[0].message["content"]

# print()
# comment out current examples
# summ_desc = get_completion_from_messages(messages)
# print(summ_desc)
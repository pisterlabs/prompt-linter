import sys
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

apikey = sys.argv[1]

def generate_openai_output(prompt, input_dict, input_tags, apikey):
	prompt_tempt = PromptTemplate.from_template(prompt)
	filtered_dict = {key: input_dict[key] for key in input_tags}
	final_prompt = prompt_tempt.format(**filtered_dict)
	llm = OpenAI(openai_api_key=apikey)
	output = llm.predict(final_prompt)
	return output

JD= "Interstate Gas Supply (IGS Energy) seeks Data Scientist, Enterprise Analytics. Position based in Dublin, OH, but remote work within the U.S. will also be considered.\n\nContinuously improve processes and decision-making through data. Using a statistics or mathematics background, the Data Scientist will wrangle with cross-sectional and time series data,\n\napply statistical tests to test hypotheses, and bring data science workloads to production through version control, continuous deployment, modular coding and software development best\n\npractices. The Data Scientist will develop and scale models in a cloud-based, big data environment (Azure/Databricks/Snowflake) and will use Python/PySpark, SQL languages.\n\nSpecifically, job duties include:\nWork closely with stakeholders, end-users, and business analysts to understand requirements and scope project\nAnalyze various data to understand relationships, integrity and how data can be leveraged\nDevelop proof of concept solutions (model development, validation) and work closely with data engineering team to scale solutions, automate quality assurance and monitor accuracy\nMaintain and develop quality standards for data assets within pipeline\nProactively partner and field requests from business partners for prospective, in-flight, and future ML pipelines\n\nRequires: Master’s degree in Mathematics, Statistics, or Data Science.\n\nSalary range is $77,000-$129,000 per year.\n\nTo apply, visit https://igsenergy.wd1.myworkdayjobs.com/IGS, or send resume to: Matthew Straley, IGS Energy, 6100 Emerald Parkway, Dublin, OH 43016\n\nOur Offer to You:\nComprehensive healthcare benefits including medical, dental, vision, and employer health savings account contributions\n401(k) retirement plan with company matching\nTuition reimbursement, employee wellness programs, and other perks and discounts\nPaid leave policies\nAnd more, such as paid time off, flexible spending, employer paid life and disability, employee assistance program, and domestic partner benefits\n\nEqual Opportunity Employment:\n\nIt is the policy of IGS Energy to ensure equal employment opportunity in accordance with all applicable federal and state regulations and guidelines. Employment discrimination against employees and applicants due to race, color, religion, sex (including sexual harassment), national origin, disability, age, sexual orientation, gender identity, military status, and veteran status or other legally protected class under applicable law is prohibited.\n"
RESUME= "John Smith\n123 Main Street, City, State ZIP\n(123) 456-7890\njohnsmith@email.com \n\n Summary: \nHighly skilled data scientist with 5+ years of experience in analyzing and interpreting complex datasets. Proficient in machine learning algorithms, statistical analysis, and data visualization. Strong problem-solving and communication skills. Committed to leveraging data-driven insights to drive business growth and improve decision-making.\n Education:\n Master of Science in Data Science, XYZ University, City, State\n - Coursework: Machine Learning, Statistical Analysis, Big Data Analytics\n - GPA: 3.9/4.0\n\n Skills:\n - Programming Languages: Python (NumPy, Pandas, scikit-learn), R, SQL\n - Machine Learning: Regression, Classification, Clustering, Neural Networks\n - Data Visualization: Tableau, Matplotlib, Seaborn\n - Statistical Analysis: Hypothesis Testing, A/B Testing, Time Series Analysis\n - Big Data Technologies: Hadoop, Spark\n\n Experience:\nData Scientist, ABC Company, City, State\n- Developed predictive models to optimize marketing campaigns, resulting in a 20% increase in customer acquisition.\n- Conducted statistical analysis on customer behavior data to identify patterns and make recommendations for product improvements.\n- Collaborated with cross-functional teams to design and implement data-driven solutions, improving operational efficiency by 15%.\n\nData Analyst, DEF Corporation, City, State\n- Cleaned and transformed large datasets using Python and SQL, reducing data processing time by 30%.\n- Developed automated reports and dashboards using Tableau, providing real-time insights to senior management.\n- Assisted in the design and execution of A/B tests, resulting in a 10% increase in conversion rates.\nProjects:\n- Predictive Maintenance: Built a machine learning model to predict equipment failures, reducing maintenance costs by 25%.\n- Customer Segmentation: Conducted clustering analysis on customer data to identify distinct segments and personalize marketing campaigns.\n- Fraud Detection: Developed an anomaly detection model using unsupervised learning algorithms to identify fraudulent transactions.\nPublications:\n- Smith, J., & Johnson, A. (2022). \"Predictive Analytics in Retail: A Case Study.\" Journal of Data Science, 10(3), 45-58.\nCertifications:\n- Certified Data Scientist (CDS), Data Science Association\nLanguages:\n- English (Fluent)\n- Spanish (Intermediate)\nReferences:\nAvailable upon request\n"

input_dict = {}
prompt = "Enhance the grammer and words used in resume specified by triple backticks and output the result using text format.\n \n    Resume: ```{RESUME}``` \n Return the new entire enhanced resume in text format."
input_dict["RESUME"] = RESUME
input_tags = ["RESUME"]
prompt_output_node_4= generate_openai_output(prompt, input_dict, input_tags, apikey)

prompt = "Extract the software technical hard skills from the Resume specified by triple backticks and output the result using CSV format. \n \n    Job Description: ```{JD}``` \n ``output`` \nThe output must only contain comma separated values."
input_dict["JD"] = JD
input_tags = ["JD"]
prompt_output_node_3= generate_openai_output(prompt, input_dict, input_tags, apikey)

prompt = "Update the Resume specified by triple backticks RESUME: ```{prompt_output_node_4}``` \n with the skills: {prompt_output_node_3} and return text format"
input_dict["prompt_output_node_3"] = prompt_output_node_3
input_dict["prompt_output_node_4"] = prompt_output_node_4
input_tags = ["prompt_output_node_3","prompt_output_node_4"]
prompt_output_node_5= generate_openai_output(prompt, input_dict, input_tags, apikey)

print(prompt_output_node_5)

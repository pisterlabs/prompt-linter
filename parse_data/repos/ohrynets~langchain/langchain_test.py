# Read configuration file
import json
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts.chat import HumanMessagePromptTemplate

from langchain.schema import HumanMessage

# Open environment file
with open('env.json') as json_file:
    data = json.load(json_file)
    openai_api_key = data['openai_api_key']
    temperature = data['openai_temperature']
    model = data['openai_model']

# A simple langchain skeleton with OpenAI configuration
# This is a simple skeleton for a language chain

with get_openai_callback() as cb:
    llm = OpenAI(openai_api_key=openai_api_key, temperature=temperature)
    result = llm.predict("What would be a good company name for a company that makes colorful socks?")
    print(result)
    print(cb)


# Lang chain prompt template
# This is a simple skeleton for a language chain
template = """/
Please extract the technical and soft skills from the following job \
description in list formed in JSON:
{job_description}
"""

job_description = HumanMessage(content="""
Lead Data Integration Engineer
 Summary
At least 5 years of relevant development experience and practice with data management, data storage, data modeling, data analytics, data migration, and database design.
Able to play Tech Lead / Team Lead role on a project and ensure that delivered solutions meet business requirements and expectations.
Expert hands-on experience in developing Data Solutions in Cloud environments(AWS, Azure, GCP). Solid experience with preferably more than one leading cloud provider; ability to coach and train other developers.
Expert-level knowledge of leading cloud data warehousing solutions (e.g. Redshift, Azure Synapse Analytics, Google BigQuery, Snowflake, etc.).
Experienced and highly self-motivated professional with outstanding analytical and problem-solving skills.
Production coding experience with one of the data-oriented programming languages.
Real production experience in developing Data Analytics & Visualization, Data Integration or DBA & Cloud Migration Solutions.
Experienced and highly self-motivated professional with outstanding analytical and problem-solving skills.
Experience allows choosing the most appropriate for the project integration patterns within the project for the team to follow. Promotes and monitors usage of the best practices in the team.
Able to read and understand project and requirement documentation; able to create design and technical documentation including high-quality documentation of his/her code.
Experienced in working with modern Agile developing methodologies and tools.
Able to work closely with customers and other stakeholders.
Able to present and justify technical solutions to a customer.
Able to work on-site with the customer.
  Job Description
Leading the team, designing, and implementing innovative data Integration Solutions, modeling databases, and contributing to building data platforms using classic Data technologies and tools (Databases, ETL/ELT technology & tools, MDM tools, BI platforms, etc.) as well as implementing new Cloud or Hybrid data solutions.
Contribute in associate role to Cloud Solution Architecture, Chooses (or contributes to process) the right toolset, acts as a role model in the team.
Work with product and engineering teams to understand requirements (and sometimes help to define technical requirements), evaluate new features and architecture to help drive decisions.
Build collaborative partnerships with architects and key individuals within other functional groups.
Perform detailed analysis of business problems and technical environments and use this in designing high-quality technical solutions.
Actively participate in code review and testing of solutions to ensure it meets best practice specifications.
Build and foster a high-performance engineering culture, supervise team members and provide technical leadership.
Write project documentation.
 Requirements
Knowledge of at least one Cloud in a deep and comprehensive manner, a few others at least on "awareness" level.
Expert knowledge of Data Integration tools (Azure Data Factory, AWS Glue, GCP Dataflow, Talend, Informatica, Pentaho, Apache NiFi, KNIME, SSIS, etc.).
Understanding of pros and cons of different RDBMS, deep expert knowledge of them (MS SQL Server, Oracle, MySQL, PostgreSQL). Ability to articulate the benefits and gaps of different technical solutions within the team and to a client.
Production experience of one of the data-oriented programming languages: SQL, Python, SparkSQL, PySpark, R, Bash.
Production projects experience in Data Management, Data Storage, Data Analytics, Data Visualization, Data Integration, MDM (for MDM-profiles), Disaster Recovery, Availability, Operation, Security, etc.
Experience with data modeling. OLAP, OLTP, ETL and DWH / Data Lake / Delta Lake / Data Mesh methodologies. Inman vs Kimbal, Staging areas, SCD and other dimension types, advanced and hybrid-data modeling approaches: Data Vault, NoSQL structures.
Good understanding of online and streaming integrations, micro-batching, understanding of CDC methods and delta extracts.
General understanding of Housekeeping processes (logging, monitoring, exception management, archiving, purging, retention policies, hot/cold data, etc.). 
Advanced knowledge of Data Security (Row-level data security, audit, etc.).
Pattern-driven solutioning, choosing the best for particular business requirements and technical constraints.
Strong understanding of Data Lineage, Metadata management, and Data traceability concepts.
Create high-quality design and technical documentation including documentation of his/her code; able to write high-quality use cases and audit documentation.  
Data-oriented focus and possessing compliance awareness, such as PI, GDPR, HIPAA.
Experience in direct communication with customers.
Experienced in different business domains.
English proficiency.
  Technology stack
Cloud providers stack (AWS/Azure/GCP): Storage; Compute; Networking; Identity and Security; DataWarehousing and DB solutions (RedShift, Snowflake, BigQuery, Azure Synapse, etc.).
Experience with some industry-standard Data Integration tools (Azure Data Factory, AWS Glue, GCP Dataflow, Talend, Informatica, Pentaho, Apache NiFi, KNIME, SSIS, etc.).
Experience in coding with one of the data-oriented programming languages: SQL, Python, SparkSQL, PySpark, R, Bash, Scala.
Expected experience working with at least one Relational Database (RDBMS: MS SQL Server, Oracle, MySQL, PostgreSQL).
Dataflow orchestration tools, data replication tools and data preparation tools.
Version Control Systems (Git, SVN).
Testing: Component/ Integration Testing / Reconciliation.                               
                               """, language="en", model=model) 

hm = HumanMessagePromptTemplate.from_template(template)
with get_openai_callback() as cb:
    chat = ChatOpenAI(openai_api_key=openai_api_key, temperature=temperature)
    result = chat.predict_messages([hm.format(job_description=job_description)])
    print(result.content)
    print(cb)

import glob
import docx2txt
import anthropic

def read_context_data():

    # read all the text in the docx files in the folder and concatenate
    # into one string
    all_text = ''
    for filename in glob.glob('drive-download-20230623T023555Z-001/*.docx'):
        all_text += docx2txt.process(filename)

def call_anthropic(context, prompt):
   
    anthropic_api_key = '<ANTHROPIC-API_KEY>'
    client = anthropic.Client(anthropic_api_key)
    resp = client.completion(
        prompt=f"{anthropic.HUMAN_PROMPT}{prompt}\n\nPlease utilize what has previously been written about data reliability engineering in these blog posts to create the definitions: {context} \n\n{anthropic.AI_PROMPT}",
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1-100k",
        max_tokens_to_sample=100000,
    )

    # return the response
    return resp["completion"]

def main():

    # prompt = Please write the rest of this technical blog post on the importance of subgraph visibility in a federated GraphQL setup. Here is the part of the blog post that has already been written: {blog_post}\n\nHere are some other blog posts that have been written for the Inigo product for context: {context} 

    prompt = f"""Please write a data reliability engineering dictionary with the following terms:
- data reliability engineering
- data reliability engineer
- data SLAs
    - data SLIs
    - data SLOs
- data quality
- dataOps
- pipeline reliability
- incident management
- automation
- data monitoring
- "toil"
- "data as code"
- root cause analysis
- data versioning
- anomaly detection
- data warehouse
- ELT/ETL orchestration
- data downtime
- CI/CD
- Infrastructure as Code
- controlled releases
- blue-green deployment

Each term should be defined in depth, with ~ 100 words. Here's an example definition:

Data Reliability Engineer: A professional who acts as a steward over the quality of data and reliability of the data process.  Data reliability engineers are typically responsible for building data pipelines to bring together information from different source systems. They integrate, consolidate and cleanse data and structure it for use in analytics applications. They aim to make data easily accessible and to optimize their organization's big data ecosystem.  Data Engineers contribute, generally, to Data Observability by testing all the known edges and capturing bugs before release of code.  By doing so, Data Reliability Engineers can focus on true anomalies.
"""

    context = read_context_data()
    finished_blog_post = call_anthropic(context, prompt)

    # save the finished blog post
    with open('finished_blog_post.txt', 'w') as f:
        f.write(finished_blog_post)
    print(finished_blog_post)

if __name__ == "__main__":
    main()

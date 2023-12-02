import os
from keys import apikey
from langchain.tools import BaseTool
import shutil
import random

os.environ['OPENAI_API_KEY'] = apikey


###############################
##### Terraform File Tool #####
###############################
class TerraformFile(BaseTool):
    name = 'Terraform file for EC2 instance'
    description = """Do NOT use this tool until you have received all input arguments from the user directly.
    
                    Use this tool to deploy an EC2 with a terraform file.
                    
                    If the request is not for "EC2" or "EC2 instance", do not create anything. Respond that you cannot create that type of resource.
                    
                    Do not attempt to create anything that is not for "EC2" or "EC2 instance" with this tool.
                    """
    


    def _run(self, action_input):

        # """
        # Do NOT use this tool until you have asked for the 'server_name' input argument.
        
        # Use this tool to deploy an EC2 with a terraform file.
        
        # If the requrest is not for "EC2" or "EC2 instance", do not create anything. Respond that you cannot create that type of resource.
        
        # Do not attempt to create anything that is not for "EC2" or "EC2 instance" with this tool.
        
        # ONLY accept action_input or input of "EC2" or "EC2 instance".

        # DO NOT accept action_input or input of anything else.
        # """
        
        src = '../tf_template/ec2_template.tf'
        dst = 'ec2_copy.tf'
        shutil.copyfile(src, dst)
        curdir=os.getcwd()
        filepath=os.path.join(curdir, dst)

        print('What do you want to name the Terraform resource?')
        tf_name = input()
        print('What do you want to name your ec2 instance?')
        tag_name = input()

        x=random.randint(10000,90000)
        ami_name = 'al2-'
        ami_name += str(x)

        replace_tf_name = ''
        replace_tag_name = ''
        replace_ami_name = ''
        replace_ami_id = ''
        with open(filepath, 'r')as f:
            replace_tf_name = f.read().replace("replace_tf_name", str(tf_name), 1)
            replace_tag_name = replace_tf_name.replace("replace_tag_name", str(tag_name), 1)
            replace_ami_name = replace_tag_name.replace("replace_ami_name", str(ami_name), 1)
            replace_ami_id = replace_ami_name.replace("replace_ami_id", str(ami_name), 1)

        with open(filepath, 'w')as f:    
            f.write(replace_tag_name)
        with open(filepath, 'w')as f:
            f.write(replace_ami_name)
        with open(filepath, 'w')as f:
            f.write(replace_ami_id)

        os.rename(dst, f'ec2_{tag_name}.tf')

        return 'Terraform files have been succcessfully created. Now run the Terraform Plan tool'
    

    def _arun(self, sops):
        raise NotImplementedError("This tool does not support async")


################################
##### Terraform Plan Tool ######
################################
class TerraformPlanTool(BaseTool):
    name = "Terraform Plan"
    description = (
        "use this tool when you need to deploy infrastructure with terraform. "
        "use this tool only after the Terraform Init tool runs successfully. "
        "if the plan is successful, ask the user if they approve of the plan "
        "do NOT move on the Terraform Apply until the user has confirmed the plan"
    )

    def _run(self, action_input):
        command = "terraform plan"
        result = os.system(command)

        if result != 0:
            return "Terraform plan failed. Run the Code Fixer tool to find the error. Then inform the user what needs to be fixed."
        
        print('Do you confirm of this Terraform plan?')
        tf_plan_confirm = input()
        return "Terraform Plan Complete. If the user said yes to confirming the Terraform Apply, you can use the Terraform Apply Tool."
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

  
################################
##### Terraform Apply Tool #####
################################
class TerraformApplyTool(BaseTool):
    name = "Terraform Apply"
    description = (
        "use this tool when you need to deploy infrastructure with terraform. "
        "use this tool only after the 'Terraform file for EC2 instance' tool runs successfully. "
        "if the apply is successful, inform the user and end the chain."
    )

    def _run(self, action_input):
        command = "terraform apply"
        result = os.system(command)

        if result != 0:
            return "Terraform apply failed."
        
        return "Terraform Apply Complete."
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
    

from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

llm = ChatOpenAI(
    openai_api_key=apikey,
    temperature=0,
    model_name='gpt-3.5-turbo'
)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=2,
    return_messages=True
)


from langchain.agents import initialize_agent

tools = [TerraformFile(), TerraformPlanTool(), TerraformApplyTool()]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

agent('Can you deploy and ec2 with terraform?')
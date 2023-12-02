import openai

class CDSGenerator:  
    def __init__(self, llm: openai):  
        self.llm = llm  
        
        
    def get_response_message_content(self, prompt: str) -> str:
        messages = [ {"role": "user", "content": prompt} ]  
        response = self.llm.ChatCompletion.create(  
            engine="gpt-4",  
            messages=messages,  
            temperature=0.01  
        )  
        response_message_content = response['choices'][0]['message']['content']  
        print("respontse from LLM generated:\n " + response_message_content)
        return response_message_content
        
  
    def generate_cds_code(self, country_code: str, field_desc_dict: dict, src_tab_name: str) -> str:
        
        field_desc_str = ""
        for i, (field_name, description) in enumerate(field_desc_dict.items(), 1):  
            field_desc_str += f"{i}. {field_name}: {description}\n"  
         
        prompt = f"""
                for country code {country_code}, \
                I have a list of field names and their corresponding descriptions, which is delimited by triple quotes. \
                \"\"\"\
                    {field_desc_str} 
                \"\"\" \          
                I also have a source table called {src_tab_name} \
                I want to generate ABAP CDS view fields for each of them. \
                """
        example = f"""Here is an example of the code I want to generate with country code US and source table pa0106: \
                @AbapCatalog.viewEnhancementCategory: [#NONE] \n
                @AccessControl.authorizationCheck: #NOT_REQUIRED \n
                @EndUserText.label: 'HCM US - Related Persons' \n
                @Metadata.ignorePropagatedAnnotations: true \n
                @ObjectModel.usageType:{{ \n
                    serviceQuality: #X, \n
                    sizeCategory: #S, \n
                    dataClass: #MIXED \n
                }} \n
                define view entity I_US_HCMFamilyMemberSupplement \n
                    as select from pa0106 \n
                {{ \n
                    key pernr as HCMPersonnelNumber, \n
                    key subty as HCMSubtype \n
                }}
                """
        prompt += example            
        prompt += f"""
                Please generate ABAP CDS view fields for these field names and descriptions, following the example code.
                Ensure do NOT provide anything else other than the code. \
                """  
       
        return self.get_response_message_content(prompt)
    

    def generate_cds_name(self, input: dict) -> dict:  
        # Generate the prompt  
        prompt = f"""
                I have a list of descriptions and I want to convert them into camel case and shorten them to less than 30 characters. \
                Here are the descriptions:\
                """  
        for i, description in enumerate(input.values(), 1):  
            prompt += f"{i}. {description}\n"  
            
        #prompt += open("naming_convention_file_name").read()
        
        prompt += f"""
                Please convert these descriptions into camel case and shorten them to less than 30 characters.
                """  
        response_message_content = self.get_response_message_content(prompt)
        print("cds name generated: " + response_message_content)  
  
        # Split the response into individual descriptions  
        response_descriptions = response_message_content.split('\n')  
  
        # Remove any empty strings from the list  
        response_descriptions = [desc for desc in response_descriptions if desc]  
  
        # Create a dictionary that pairs each field name with its corresponding description  
        result = {}  
        for field_name, field_desc_camel in zip(input.keys(), response_descriptions):  
            # Remove the leading number and period from each description  
            field_desc_camel = field_desc_camel.split('. ', 1)[-1]  
            result[field_name] = field_desc_camel  
  
        return result  

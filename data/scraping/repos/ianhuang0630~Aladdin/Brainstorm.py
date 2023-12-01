"""
Implementation of modules to query GPT3 to generate interesting/useful
scene descriptions.
"""

import openai
from bson.objectid import ObjectId
import sys
from sem_complete.sem_complete.Template import SceneQueryTemplate
from sem_complete.sem_complete.Parser import starListParser, starKeyValParser, parseDescriptionCoordinates
from sem_complete.sem_complete.SceneGraphs import SceneShoppingList, SceneObjectClass
from loguru import logger

class GPTInterface(object):
    def __init__(self, model_type='text-davinci-003', api_key=None,
                 default_temperature=None, default_max_tokens=None):

        if openai.api_key is None:
            assert api_key is not None
            openai.api_key = api_key
        self.model_type = model_type
        
        self.temperature = default_temperature
        self.max_tokens = default_max_tokens

        # datastructure to keep track of conversation history.
        self.prompts = []
        self.conditionings = []
        self.questions = []
        self.responses = []

    def query(self, conditioning, question, temperature=None, max_tokens=None):
    
        prompt  = ''.join([conditioning, question])
        response = openai.Completion.create(model=self.model_type,
                                            prompt=prompt,
                                            temperature=temperature if temperature is not None else self.temperature,
                                            max_tokens=max_tokens if max_tokens is not None else self.max_tokens)
        response = response['choices'][0]['text']
        response = response.strip()

        self.conditionings.append(conditioning)
        self.questions.append(question)
        self.prompts.append(prompt)
        self.responses.append(response)
        return response


class Brainstormer(object):
    def __init__(self, gpt_interface=None, temperature=0.8, max_tokens=1024):
        if gpt_interface is None:
            with open('credentials/openai_key', 'r')  as f:
                token = f.readlines()
                token = token[0].strip()
            self.interface = GPTInterface(api_key=token,
                                    default_temperature=0.8,
                                    default_max_tokens=1024)
        else:
            assert isinstance(gpt_interface, GPTInterface)
            self.interface = gpt_interface


# class HumanInTheLoopBrainstormer(Brainstormer):
#     """
#     A class that takes in user commands in the loop
#     """

#     def __init__(self, interface=None, temperature=0.8, max_tokens=1024):
#         super().__init__(gpt_interface=interface, 
#                          temperature=temperature,
#                          max_tokens=max_tokens)

#     def run_iteration(self):
#         """
#         Runs a single iteration of adding detail to the current scene.
#         """
#         pass

#     def gather_feedback(self, user_input):
#         """
#         Gathers and parses user feedback after ever iteration. Sets the 
#         internal state of the Brainstormer to run the next iteration with 
#         the correct goals.
#         """
#         pass

class IterativeBrainstormer(Brainstormer) :
    def __init__(self, interface=None, temperature=0.8, max_tokens=1024):
        super().__init__(gpt_interface=interface,
                         temperature=temperature,
                         max_tokens=max_tokens)

    def generate_anchors(self, description):
        sqt = SceneQueryTemplate('sem_complete/templates/iterative/frenchrestaurant_anchors.txt') 
        question = f"""
Here we are building a 3D scene of {description}. At each step, we are not adding more than 8 assets in total into the scene. 

First, we place the most important assets (e.g. furnitures, bigger objects) and use those as our anchors. Here is a list of them:
"""
        prompt, conditioning, gpt_question = sqt.apply_directly(question)
        response = self.interface.query(conditioning, gpt_question)
        categories = starKeyValParser(response)
        return {'complete_prompt': prompt, 
                'conditioning': conditioning, 
                'question': question, 
                'gpt_question': gpt_question,
                'response': response, 
                'parsed': categories}

    def enhance_at(self, description, category, prompt_prepend):
        sqt = SceneQueryTemplate(['sem_complete/templates/iterative/frenchrestaurant_anchors.txt',
                                  'sem_complete/templates/iterative/frenchrestaurant_enhance.txt'])

        question = f"""
Next we enhance the scene with more assets, in relation to the anchor objects. 
In relation to the `{category}`, here is the list of assets we add:
        """
        prompt, conditioning, gpt_question = sqt.apply_directly(question if prompt_prepend is None else prompt_prepend + question)
        response = self.interface.query(conditioning, gpt_question)
        categories =  starKeyValParser(response)
        
        return {'complete_prompt': prompt, 
                'conditioning': conditioning, 
                'question': question, 
                'gpt_question': gpt_question,
                'response': response, 
                'parsed': categories}

    def physical_condition_category(self, description, prompt_prepend):
        sqt = SceneQueryTemplate(['sem_complete/templates/iterative/frenchrestaurant_nodeattributes.txt',
                                  'sem_complete/templates/iterative/frenchrestaurant_nodephyscond.txt'])
        question = f"Describe the physical condition of these items in a scene of {description} :"
        prompt, conditioning, gpt_question = sqt.apply_directly(question if prompt_prepend is None else prompt_prepend + question)
        response = self.interface.query(conditioning, gpt_question)
        categories = starKeyValParser(response)

        return {'complete_prompt': prompt, 
                'conditioning': conditioning, 
                'question': question, 
                'gpt_question': gpt_question,
                'response': response, 
                'parsed': categories}

    def describe_category(self, description, prompt_prepend):
        sqt = SceneQueryTemplate(['sem_complete/templates/iterative/frenchrestaurant_nodes.txt',
                                  'sem_complete/templates/iterative/frenchrestaurant_nodeattributes.txt'])
        question = f"""
Suppose we want to create a shopping list for the items we need to create  the above scene of {description}. It would look like, being specific about the brand and the visual properties:
"""
        prompt, conditioning, gpt_question = sqt.apply_directly(question if prompt_prepend is None else prompt_prepend + question)
        response = self.interface.query(conditioning, gpt_question)
        categories = starKeyValParser(response)

        return {'complete_prompt': prompt, 
                'conditioning': conditioning, 
                'question': question, 
                'gpt_question': gpt_question,
                'response': response, 
                'parsed': categories}

    def run(self, description:str, num_iterations:int): 
        """
        Args:
            description: abstract scene discription
            num_iterations: number of times gpt is queried to generate objects around the anchor objects.          
        """

        anchor_results =  self.generate_anchors(description)
        # TODO: get positions and attributes      
        prompt_prepend = f"Here is a list of items found within a scene of {description}:\n"+anchor_results['response']
        if not prompt_prepend.endswith("\n"):
            prompt_prepend += "\n"
        node_attributes = self.describe_category(description, prompt_prepend)

        prompt_prepend = node_attributes['question'] + '\n' + node_attributes['response']
        if not prompt_prepend.endswith("\n"):
            prompt_prepend += "\n"
        node_physical_condition = self.physical_condition_category(description, prompt_prepend)

        ssl = SceneShoppingList()
        anchor_object_classes = []
        for category in anchor_results['parsed']:
            if category not in node_attributes['parsed'] or \
                category not in node_physical_condition['parsed']:
                logger.info(f'category {category} not found in node_attributes or node_physical_condition')
                continue

            new_object_class = SceneObjectClass(class_name=category, 
                                                instances=anchor_results['parsed'][category],
                                                attributes=', '.join(node_attributes['parsed'][category][1:]) + ', '.join(node_physical_condition['parsed'][category]))
            anchor_object_classes.append(new_object_class)
            ssl.add_objectclass(new_object_class, parent=None)
            
        for i in range(num_iterations): 
            # NOTE iterating through the same set of anchor points. Change if you want the recursion to continue.
            for category, category_object_class in zip(anchor_results['parsed'], anchor_object_classes):
                #TODO: try & catch for errors when output isn't parsable.
                
                # check the number of instances there are.
                prompt_prepend = anchor_results['question'] + '\n' + anchor_results['response']
                if not prompt_prepend.endswith("\n"):
                    prompt_prepend += "\n"
                enhance_results = self.enhance_at(description, category, prompt_prepend)
                
                prompt_prepend = f"Here is a list of items found within a scene of {description}:\n"+enhance_results['response']       
                if not prompt_prepend.endswith("\n"):
                    prompt_prepend += "\n" 
                node_attributes = self.describe_category(description, prompt_prepend)

                prompt_prepend = node_attributes['question'] + '\n' + node_attributes['response']
                if not prompt_prepend.endswith("\n"):
                    prompt_prepend += "\n"
                node_physical_condition = self.physical_condition_category(description, prompt_prepend)
                
                for category in enhance_results['parsed']:
        
                    if category not in node_attributes['parsed'] or \
                        category not in node_physical_condition['parsed']:
                        logger.info(f'category {category} not found in node_attributes or node_physical_condition')
                        continue

                    new_object_class = SceneObjectClass(class_name=category, 
                                                        instances=enhance_results['parsed'][category],
                                                        attributes=', '.join(node_attributes['parsed'][category][1:]) + ', '.join(node_physical_condition['parsed'][category]))
                    ssl.add_objectclass(new_object_class, category_object_class)

        logger.info(str(ssl)) 
        return ssl
        

class OnePassBrainStormer(Brainstormer) :
    def __init__(self, interface=None, temperature=0.8, max_tokens=1024):
        super().__init__(gpt_interface=interface,
                         temperature=temperature, 
                         max_tokens=max_tokens)

    def generate_categories(self, description: str) -> dict:
        sqt = SceneQueryTemplate('sem_complete/templates/onepass/frenchrestaurant_nodelist.txt')
        question = "Here is a list of items one would find in {}:".format(description)
        prompt, conditioning, gpt_question = sqt.apply_directly(question)
        response = self.interface.query(conditioning, gpt_question)
        categories = starListParser(response)
        return {'complete_prompt': prompt, 
                'conditioning': conditioning, 
                'question': question, 
                'gpt_question': gpt_question,
                'response': response, 
                'parsed': categories}

    def generate_num_instances(self, description: str, categories:list, prompt_prepend:str=None) -> dict:
        sqt = SceneQueryTemplate(['sem_complete/templates/onepass/frenchrestaurant_nodelist.txt', 'templates/frenchrestaurant_nodenuminstances.txt'])
        question = "Here is a list of the number of items one would find in {}:".format(description)
        # if prompt_prepend is not None:
        #     question = prompt_prepend + question
        prompt, conditioning, gpt_question = sqt.apply_directly(question if prompt_prepend is None else prompt_prepend + question) 
        response = self.interface.query(conditioning, gpt_question)
        cat2numinstances = starKeyValParser(response) 
        return {'complete_prompt': prompt, 
                'conditioning': conditioning, 
                'question': question, 
                'gpt_question': gpt_question,
                'response': response, 
                'parsed': cat2numinstances}

    def generate_positioning(self, description: str, cat2instances: dict, prompt_prepend:str=None) -> dict:
        # TODO
        instance2relativeposition = None

        sqt = SceneQueryTemplate(['sem_complete/templates/onepass/frenchrestaurant_nodenuminstances.txt', 'templates/frenchrestaurant_nodepositions.txt'])
        question = "Describe their relative placements to at least two items, with their coordinates from birds-eye view:"
        # if prompt_prepend is not None:
        #     question = prompt_prepend + question
        prompt, conditioning, gpt_question = sqt.apply_directly(question if prompt_prepend is None else prompt_prepend + question)
        response = self.interface.query(conditioning, gpt_question)
        instance2relativeposition = starKeyValParser(response, val_parser=parseDescriptionCoordinates)
         
        return {'complete_prompt': prompt, 
                'conditioning': conditioning, 
                'question': question, 
                'gpt_question': gpt_question,
                'response': response, 
                'parsed': instance2relativeposition}
    
    def generate_cat_attributes(self, description: str, categories: list, prompt_prepend:str=None) -> dict:
        sqt = SceneQueryTemplate(['sem_complete/templates/onepass/frenchrestaurant_nodelist.txt', 'templates/frenchrestaurant_nodeattributes.txt']) # loads the node attributes txt
        question = "For every item above, list the item and its material properties."
        # if prompt_prepend is not None:
        #     question = prompt_prepend + question
        prompt, conditioning, gpt_question = sqt.apply_directly(question if prompt_prepend is None else prompt_prepend + question) 
        response = self.interface.query(conditioning, gpt_question)
        cat2attributes = starKeyValParser(response)
        return {'complete_prompt': prompt, 
                'conditioning': conditioning, 
                'question': question, 
                'gpt_question': gpt_question,
                'response': response, 
                'parsed': cat2attributes}

    def generate_instance_attributes(self, description: str, cat2instances: dict) -> dict:
        
        pass

    def run(self, description: str) -> dict:
        """
        """

        # step1: generate categories
        out = self.generate_categories(description)
        category_list = out['parsed']
        category_question = out['question']
        category_response = out['response']
        prompt_prepend = category_question + '\n' + category_response
        if not prompt_prepend.endswith("\n"):
            prompt_prepend += "\n"
 
        # step2: attributes per category
        out = self.generate_cat_attributes(description, category_list, prompt_prepend=prompt_prepend)
        cat2catattributes = out['parsed']

        # step3: generate # of each class
        out = self.generate_num_instances(description, category_list, prompt_prepend=prompt_prepend)
        cat2numinstances = out['parsed']
        inst_question = out['question']
        inst_response = out['response']
         
        # step4: positional information.
        prompt_prepend = inst_question + '\n' + inst_response
        if not prompt_prepend.endswith("\n"):
            prompt_prepend += "\n"
        out = self.generate_positioning(description, cat2numinstances, prompt_prepend=prompt_prepend) 
        positions = out['parsed']

        return {'category_list': category_list,
                'category_attributes': cat2catattributes,
                'category_instances': cat2numinstances,
                'positions': positions}


if __name__=='__main__' : 
    IB = IterativeBrainstormer()
    # IB.run('a messy living room', 1)
    IB.run('an abandoned warzone in ukraine', 1)

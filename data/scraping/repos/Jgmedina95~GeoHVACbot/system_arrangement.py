from langchain.tools import BaseTool

class pipe_config_functions:
    def pipe_configurations(self,total_length, max_length_per_tube=200, max_num_tubes=6):
        configurations = []
        for num_tubes in range(1, max_num_tubes + 1):
            if total_length % num_tubes == 0:
                length_per_tube = total_length // num_tubes
                if length_per_tube <= max_length_per_tube:
                    configurations.append((num_tubes, length_per_tube))
        return configurations
  
    def query_to_dict(self,query_str):
        # Split from commas
        query_list = query_str.split(',')
        # Split from equal sign and strip spaces
        query_dict = {q.split('=')[0].strip(): q.split('=')[1].strip() for q in query_list}
        return query_dict
        
class system_arrangement(BaseTool):
    name = "System Arrangement"
    description = """This tool will help you to determine the best system arrangement for your project. The constraints are optional.
    input: (Keyword arguments) length in feets, max_num_tubes, max_length_per_tube. separated by commas.
    output: recommended system plus recommendations
    
    Examples:
    Input: length = 200, max_num_tubes=6, max_length_per_tube=200
    Input: length = 670, max_num_tubes=7, max_length_per_tube=100

    """

    def _run(self, query):
        fxns = pipe_config_functions()
        query_params = fxns.query_to_dict(query)

        length = query_params.get('length', -1)
        length = float(length)
        max_num_tubes = query_params.get('max_num_tubes', 6)
        max_length_per_tube = query_params.get('max_length_per_tube', 200)
        if length == -1:
            return "Please enter params correctly. Example: length=200, max_num_tubes=6, max_length_per_tube=200"
        else:
            #advice = "Horizontal system is recommended"
            #configs = fxns.pipe_configurations(length)

#elif length > 200:
            advice = "Vertical system is recommended"
            configs = fxns.pipe_configurations(length)
            config_format = ""
            if configs == []:
                configs.append("No configuration found for max_length_per_tube=200 and max_num_tubes=6. Trying with max_length_per_tube=100 and max_num_tubes=7") 
                max_length_per_tube = 250
                max_num_tubes = 7   
                configs = fxns.pipe_configurations(length, max_length_per_tube, max_num_tubes)
            for config in configs:
                config_format += f"{config[0]} tubes of {config[1]} ft each\n"
            return f"{advice}\n{config_format}"
            
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")



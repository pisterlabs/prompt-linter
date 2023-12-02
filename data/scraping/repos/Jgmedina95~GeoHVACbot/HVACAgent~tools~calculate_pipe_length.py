from langchain.tools import BaseTool

class pipe_functions:
    def determine_pipe_length(self,Q, tg, tw, diameter, sdr_or_schedule, flow_rate,antifreeze=False):
    # Complete thermal resistance table as a nested dictionary
        resistance_table = {
            '3/4': {
                'SDR 11': [0.09, 0.12, None, None],
                'SDR 9': [0.11, 0.15, None, None],
                'Sch 40': [0.10, 0.14, None, None]
            },
            '1': {
                'SDR 11': [0.09, 0.14, 0.10, None],
                'SDR 9': [0.11, 0.16, 0.12, None],
                'Sch 40': [0.10, 0.15, 0.11, None]
            },
            '1 1/4': {
                'SDR 11': [0.09, 0.15, 0.12, 0.09],
                'SDR 9': [0.11, 0.17, 0.15, 0.11],
                'Sch 40': [0.09, 0.15, 0.12, 0.09]
            },
            '1 1/2': {
                'SDR 11': [0.091, 0.16, 0.15, 0.09],
                'SDR 9': [0.111, 0.18, 0.17, 0.11],
                'Sch 40': [0.081, 0.14, 0.14, 0.08]
            }
        }
        # Flow rate mapping to index in resistance_table values
        flow_map = {
            '2.0 US gpm': 0,
            '3.0 US gpm': 1,
            '5.0 US gpm': 2,
            '10.0 US gpm': 3
        }

        try:
            # Look up the resistance from the table
            R = resistance_table[diameter][sdr_or_schedule][flow_map[flow_rate]]
            # If R is None, it means it's not available ("NR") for that configuration
            if R is None:
                raise ValueError(f"No resistance value available for {diameter}, {sdr_or_schedule}, {flow_rate}")
            # Calculate L
            vertical_L = Q * R / (tg - tw)
            hortizontal_L = Q /12000 * 500
            if antifreeze:
                vertical_L /= 0.9920
                hortizontal_L /= 0.9920
            
            return vertical_L, hortizontal_L

        except KeyError:
            return f"Invalid input combination: {diameter}, {sdr_or_schedule}, {flow_rate}. Try again with the correct format (for example: Q= 12000, tg= 50, tw= 40) "
        
    def get_min_diameter(self,flow, material, antifreeze=False):
        flow_table = {
        "3/4": {
            "SDR 11 HDPE": 4.5,
            "SDR 17 HDPE": None,
            "Sched 40 Steel": 4
        },
        "1": {
            "SDR 11 HDPE": 8,
            "SDR 17 HDPE": None,
            "Sched 40 Steel": 7
        },
        "1 1/4": {
            "SDR 11 HDPE": 15,
            "SDR 17 HDPE": None,
            "Sched 40 Steel": 15
        },
        "1 1/2": {
            "SDR 11 HDPE": 22,
            "SDR 17 HDPE": None,
            "Sched 40 Steel": 23
        },
        "2": {
            "SDR 11 HDPE": 40,
            "SDR 17 HDPE": None,
            "Sched 40 Steel": 45
        },
        "3": {
            "SDR 11 HDPE": 110,
            "SDR 17 HDPE": 140,
            "Sched 40 Steel": 130
        },
        "4": {
            "SDR 11 HDPE": 220,
            "SDR 17 HDPE": 300,
            "Sched 40 Steel": 260
        },
        "6": {
            "SDR 11 HDPE": 600,
            "SDR 17 HDPE": 750,
            "Sched 40 Steel": 800
        },
        "8": {
            "SDR 11 HDPE": 1200,
            "SDR 17 HDPE": 1500,
            "Sched 40 Steel": 1600
        },
        "10": {
            "SDR 11 HDPE": 2200,
            "SDR 17 HDPE": 2600,
            "Sched 40 Steel": 3000
        },
        "12": {
            "SDR 11 HDPE": 3500,
            "SDR 17 HDPE": 4200,
            "Sched 40 Steel": 4600
        }
    }

        # Iterate through the flow table
        for diameter, materials in flow_table.items():
            if antifreeze:
                if materials[material] is not None and flow <= materials[material]*0.9:
                    return diameter
            else:
                if materials[material] is not None and flow <= materials[material]:
                    return diameter

        # If no suitable diameter found 
        return "No suitable diameter found for the given flow and material. Estimate with 1 inch diameter"

        
    def query_to_dict(self,query_str):
        # Split from commas
        query_list = query_str.split(',')
        # Split from equal sign and strip spaces
        query_dict = {q.split('=')[0].strip(): q.split('=')[1].strip() for q in query_list}
        return query_dict
        
class calculate_pipe_length(BaseTool):

    name = "Pipe Length Calculator"
    description = """Calculates the length of the pipe needed for the geothermal heat pump system.
    Heat load, ground temperature and working temperatures are needed. make sure to obtain the temperatures from other tools or from the user. 
    If working conditions is lower than 40F, antifreeze is needed.
    input: (keyword arguments) Q, tg, tw.
    
    Examples:
    Input: Q= 12000, tg= 50, tw= 43, diameter= 1.0 in.
    Input: Q = 35000, tg = 45, tw = 30, antifreeze = True 
    """
    
    def _run(self, query):
        fxns = pipe_functions()

        # Get the inputs from the user
        print(query, type(query))
        query = fxns.query_to_dict(query)
        #flow rate estimations using rule of thumb 3 US gpm per ton (12,000 BTU/hr)
        flow_rate = int(query['Q'])*1/12000
        #flow rate equal to the closest between 2,3,5,10 US gpm
        if flow_rate <= 2.5:
            flow = 2.0
            flow_rate = '2.0 US gpm'
        elif flow_rate <= 4.0:
            flow = 3.0
            flow_rate = '3.0 US gpm'
        elif flow_rate <= 7.5:
            flow = 5.0
            flow_rate = '5.0 US gpm'
        else:
            flow = 10.0
            flow_rate = '10.0 US gpm'
        # Get the minimum diameter
        diameter = fxns.get_min_diameter(flow, 'SDR 11 HDPE')

        antifreeze = query.get('antifreeze', False)

        vertical_length, horizontal_length = fxns.determine_pipe_length(
            Q=int(query['Q']),
            tg=float(query['tg']),
            tw=float(query['tw']),
            diameter=diameter,
            sdr_or_schedule='SDR 11',
            flow_rate=flow_rate,
            antifreeze=antifreeze
        )

        return f"""The length of the pipe needed is {vertical_length} ft. with a flow rate of {flow_rate} and a diameter of {diameter} in.
                    if a horizontal system is prefered {horizontal_length} ft. of pipe is needed based on rules of thumb."""

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("custom_search does not support async")
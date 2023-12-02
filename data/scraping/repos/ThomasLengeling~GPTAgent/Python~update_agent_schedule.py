import geocoder
import json
import time
# import openai

from networkx import Graph
import networkx as nx
import osmnx as ox


if __name__ == "__main__":
    from build_map_info import cityMap
else:
    from .build_map_info import cityMap



"""
尚未封裝

"""
class Agents:
    def __init__(self) -> None:
        # self.agent_number = number
        # self.location = location        
        pass

    def ceateAgent(self):
        """
            create by OPENAI
        """
        self.agent_activities = []
        pass

    def testAgent(self):
        activities = {
            "Profile": {
                "Name": "John Smith",
                "Age": 26,
                "Nationality": "American",
                "Academic Background": "University of California, Berkeley",
                "Research Interests": ["Sustainable urban systems", "Equitable city design", "Resilience in urban infrastructure", "Data-driven urban planning", "Smart cities"],
                "Residence": "303 3rd St, Cambridge, MA 02142, USA",
                "Personal Interests": ["Cycling", "Sustainability", "Reading", "Cooking", "Podcasts", "Photography"],
                "Skills": ["Python", "R", "GIS", "Basic machine learning", "Communication", "Data analysis", "Project management"]
            },
            "Day Schedule": {
                "MatrixName": "John Smith's Daily Activities",
                "Activities": [
                    {"Time": 0, "Activity": "Sleep", "Address": "303 3rd St, Cambridge, MA 02142, USA", "Transportation": "Stay"},
                    {"Time": 420, "Activity": "Wake up, Stretch, Prepare breakfast", "Address": "303 3rd St, Cambridge, MA 02142, USA", "Transportation": "Stay"},
                    {"Time": 480, "Activity": "Reading research articles and papers", "Address": "303 3rd St, Cambridge, MA 02142, USA", "Transportation": "Stay"},
                    {"Time": 540, "Activity": "Bike ride to MIT campus", "Address": "MIT Campus, Cambridge, MA 02142, USA", "Transportation": "Bike"},
                    {"Time": 600, "Activity": "Independent study and research", "Address": "MIT Media Lab, 75 Amherst St, Cambridge, MA 02139, USA", "Transportation": "Stay"},
                    {"Time": 660, "Activity": "Attend seminar on sustainable urban development at Bartos Theater", "Address": "20 Ames St, Cambridge, MA 02142, USA", "Transportation": "Walk"},
                    {"Time": 720, "Activity": "Lunch preparation", "Address": "303 3rd St, Cambridge, MA 02142, USA", "Transportation": "Stay"},
                    {"Time": 780, "Activity": "Lunch at Clover Food Lab food truck", "Address": "Kendall Square, Cambridge, MA 02142, USA", "Transportation": "Bike"},
                    {"Time": 840, "Activity": "Research and project work at MIT Media Lab", "Address": "75 Amherst St, Cambridge, MA 02139, USA", "Transportation": "Walk"},
                    {"Time": 900, "Activity": "Attend data visualization workshop at MIT Libraries", "Address": "160 Memorial Dr, Cambridge, MA 02142, USA", "Transportation": "Walk"},
                    {"Time": 960, "Activity": "Independent study and research", "Address": "MIT Media Lab, 75 Amherst St, Cambridge, MA 02139, USA", "Transportation": "Stay"},
                    {"Time": 1020, "Activity": "Group meeting at R&D Commons", "Address": "32 Vassar St, Cambridge, MA 02139, USA", "Transportation": "Walk"},
                    {"Time": 1080, "Activity": "Workout and swim at Zesiger Sports and Fitness Center", "Address": "120 Vassar St, Cambridge, MA 02139, USA", "Transportation": "Walk"},
                    {"Time": 1140, "Activity": "Bike ride back to the apartment and prepare dinner", "Address": "303 3rd St, Cambridge, MA 02142, USA", "Transportation": "Bike"},
                    {"Time": 1200, "Activity": "Dinner at home", "Address": "303 3rd St, Cambridge, MA 02142, USA", "Transportation": "Stay"},
                    {"Time": 1260, "Activity": "Reading book / Podcast", "Address": "303 3rd St, Cambridge, MA 02142, USA", "Transportation": "Stay"},
                    {"Time": 1320, "Activity": "Review of next day's tasks", "Address": "303 3rd St, Cambridge, MA 02142, USA", "Transportation": "Stay"},
                    {"Time": 1380, "Activity": "Sleep", "Address": "303 3rd St, Cambridge, MA 02142, USA", "Transportation": "Stay"}
                ]
            }
        }
        self.agent_activities = [activities]

    def checkAgentFormat(self):
        """
            check format
        """
        pass

    def addCoordinates(self):
        '''
            should wirtten in multithreads
        '''
        self.agents_coords = []
        for one_agent_activities in self.agent_activities:
            tmpCoord = []
            for event in one_agent_activities["Day Schedule"]["Activities"]:
                address_info = geocoder.tomtom(event["Address"], key='S5gXe7d9lxpiWwDFIRuOjR3qAxcN3qJZ')
                # print(f"{event['Address']}  <>  {address_info}")
                event["Coords"] = [address_info.lng, address_info.lat]
                event["log"] = address_info.lng
                event["lat"] = address_info.lat

                tmpCoord.append([address_info.lng, address_info.lat])
            
            self.agents_coords.append(tmpCoord)


    def update_Agents_schedule_route(self, graph:Graph, nodes):
        """
        need fix
        """
        update_all_agent_schedule_route = []

        for k, one_agent_coords in enumerate(self.agents_coords):
            update_one_agent_schedule_route = {}
            update_one_agent_schedule_route['Day Schedule'] = {}

            ActivitiesArray = []

            ActivitiesArray.append(self.agent_activities[k]["Day Schedule"]["Activities"][0])
            ActivitiesArray[0]["Time"] = 200
            ## 200 / 60 = 3:20 am 
            print(f"=============================== {k} ===============================")
            print(ActivitiesArray[0])
            print(one_agent_coords)
            print(len(one_agent_coords))
            
            
            for i in range(len(one_agent_coords) - 1):
                ActivitiesArray.append(self.agent_activities[k]["Day Schedule"]["Activities"][i])
                
                try:
                    origl = ox.nearest_nodes(graph, one_agent_coords[i][0], one_agent_coords[i][1])
                    dest = ox.nearest_nodes(graph, one_agent_coords[i+1][0], one_agent_coords[i+1][1])
                    route = nx.shortest_path(graph, origl, dest, weight='travel_time')
                    print(f'{i}: {route}')
                except:
                    print("========================================================")
                    print()
                    print(self.agent_activities[k]["Day Schedule"]["Activities"][i])
                    print()
                    print("========================================================")
                    continue

                if len(route) == 1:
                    continue

                time_interval = (self.agent_activities[k]["Day Schedule"]["Activities"][i+1]['Time'] - self.agent_activities[k]["Day Schedule"]["Activities"][i]['Time']) / (len(route) + 1)
                # print(time_interval)
                # for j, node in enumerate(nodes.loc[route]):
                tmp_index = 1

      

                for _, node in nodes.loc[route].iterrows():
                    event = {}
                    event['Time'] = round(int(self.agent_activities[k]["Day Schedule"]["Activities"][i]['Time']) + time_interval * (tmp_index+1), 3)
                    event['log'] = node.x
                    event['lat'] = node.y
                    
                    event['Activity'] = self.agent_activities[k]["Day Schedule"]["Activities"][i]['Activity']
                    event['Address'] = self.agent_activities[k]["Day Schedule"]["Activities"][i]['Address']
                    event['Transportation'] = self.agent_activities[k]["Day Schedule"]["Activities"][i]['Transportation']

                    tmp_index += 1
                
                    ActivitiesArray.append(event)


            update_one_agent_schedule_route['Day Schedule']['Activities'] = ActivitiesArray

            update_all_agent_schedule_route.append(update_one_agent_schedule_route)

        with open('update_all_agent_schedule_route.json', 'w') as f:
            json.dump(update_all_agent_schedule_route, f)




if __name__ == "__main__":
    ag:Agents = Agents()
    ag.testAgent()

    with open('pre_agent.json') as f:
        file = json.loads(f.read())

    ag.agent_activities = file['Records']

    ag.addCoordinates()

    # with open('nn9agent.json', 'r') as f:
    #     file = json.loads(f.read())
    # ag.agent_activities = file
    

    mmap = cityMap(loaction="Cambridge, MA, USA")
    mmap.getNetwork(network_type='drive')
   

    # print(ag.agent_activities[0])

    ag.update_Agents_schedule_route(mmap.graph, mmap.nodes)






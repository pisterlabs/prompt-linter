from ..Extract_Analyze.OutputFolderReader import OutputFolderReader
from ..OpenAICaller import openAICaller

from .Comparer import Comparer

class ThemeComparer(Comparer):
    def __init__(self):
        super().__init__()
        self.get_validation_set('themes')
        self.compared_themes = dict()

    def add_report_ID(self, report_id, report_theme):
        self.validation_set[report_id] = report_theme

    def compare_themes(self):
        print("Comparing themes...")
    
        OutputFolderReader().read_all_themes(self.compare_two_themes)

        print("Finished comparing themes.")
        
        print('==Validation summary==')
        print(f"  {len(self.compared_themes)} reports compared.")
        print(f"  Average percentage: {sum(self.compared_themes.values())/len(self.compared_themes)}%")
        print(f"  Highest percentage: {max(self.compared_themes.values())}%")
        print(f"  Lowest percentage: {min(self.compared_themes.values())}%")
        print(f"  Percentage of reports with 100%: {len([x for x in self.compared_themes.values() if x == 100])/len(self.compared_themes)}%")
        

    def compare_two_themes(self, report_id, report_theme):
        if not report_id in self.validation_set.keys():
            return
        
        validation_theme = self.validation_set[report_id]

        message = f"==Engine generated themes==\n{report_theme}\n\n==Human generated themes==\n{validation_theme}"

        system = "I am creating an engine that reads Maritime accident investigation reports. \n\nI want to compare the engine-generated themes with that were retrieved from an average human.\n\nCould you please read the two themes and give me a single percentage outcome for how similar they are.\n\n100% means that they have exactly the same themes\n50% means that about half of the themes are correct\n0% Means that there is no overlap in themes.\n\nYour reply should only include the percentage and nothing else."
        while True:
            try:
                response = openAICaller.query(system, message, temp = 0)
                response_percentage = int(response.replace("%", "").replace(" ", ""))
                break
            except (ValueError):
                print("Could not parse response, trying again")
                continue

        self.compared_themes[report_id] = response_percentage
    

        
        
        


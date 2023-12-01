import pandas as pd
import openai
import json
import os
import warnings

class GoalGenerator(pd.DataFrame):
    def __init__(self, df, dir, description=None, name=None) -> None:
        super().__init__(df)
        self.df = df      
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        self.project_dir = dir
    
    def check_type(self, dtype: str, value):
        """Cast value to right type to ensure it is JSON serializable"""
        if "float" in str(dtype):
            return float(value)
        elif "int" in str(dtype):
            return int(value)
        else:
            return value

    def get_column_properties(self, df: pd.DataFrame, n_samples) -> list[dict]:
        """Get properties of each column in a pandas DataFrame"""
        properties_list = []
        for column in df.columns:
            dtype = df[column].dtype
            properties = {}
            if dtype in [int, float, complex]:
                properties["dtype"] = "number"
                properties["std"] = self.check_type(dtype, df[column].std())
                properties["min"] = self.check_type(dtype, df[column].min())
                properties["max"] = self.check_type(dtype, df[column].max())

            elif dtype == bool:
                properties["dtype"] = "boolean"
            elif dtype == object:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(df[column], errors='raise')
                        properties["dtype"] = "date"
                except ValueError:
                    if df[column].nunique() / len(df[column]) < 0.5:
                        properties["dtype"] = "category"
                    else:
                        properties["dtype"] = "string"
            elif pd.api.types.is_categorical_dtype(df[column]):
                properties["dtype"] = "category"
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                properties["dtype"] = "date"
            else:
                properties["dtype"] = str(dtype)

            if properties["dtype"] == "date":
                try:
                    properties["min"] = df[column].min()
                    properties["max"] = df[column].max()
                except TypeError:
                    cast_date_col = pd.to_datetime(df[column], errors='coerce')
                    properties["min"] = cast_date_col.min()
                    properties["max"] = cast_date_col.max()
            nunique = df[column].nunique()
            if "samples" not in properties:
                non_null_values = df[column][df[column].notnull()].unique()
                n_samples = min(n_samples, len(non_null_values))
                samples = pd.Series(non_null_values).sample(n_samples, random_state=42).tolist()
                properties["samples"] = samples
            properties["num_unique_values"] = nunique
            properties["semantic_type"] = ""
            properties["description"] = ""
            properties_list.append({"column": column, "properties": properties})

        return properties_list
    
    def summarize(self, df, file_name):
        data_properties = self.get_column_properties(df, 2)
        base_summary = {
            "name": file_name,
            "file_name": file_name,
            "dataset_description": "",
            "fields": data_properties,
        }
        system_prompt = """
        You are an experienced data analyst that can annotate datasets. Your instructions are as follows:
        i) ALWAYS generate the name of the dataset and the dataset_description
        ii) ALWAYS generate a field description.
        iii.) ALWAYS generate a semantic_type (a single word) for each field given its values e.g. company, city, number, supplier, location, gender, longitude, latitude, url, ip address, zip code, email, etc
        You must return an updated JSON dictionary without any preamble or explanation.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": f"""
        Annotate the dictionary below. Only return a JSON object.
        {base_summary}
        """},
        ]
        print("SUMMARY PROMPT:", messages)
        json_string = openai.ChatCompletion.create(model="gpt-3.5-turbo", \
                                                  temperature=0.2, \
                                                  messages=messages).choices[0].message.content
        try:
            enriched_summary = json.loads(json_string)
        except json.decoder.JSONDecodeError:
            error_msg = f"The model did not return a valid JSON object while attempting to generate an enriched data summary. Instead returned this-"
            print(error_msg)
            print(json_string)   
        # enriched_summary = json.load(open('viz-cars-data/data/json_string_summary.json'))    
        return enriched_summary

    def gen_goals(self, summary, persona, n=3):
        user_prompt = f"""The number of GOALS to generate is {n}. The goals should be based on the data summary below, \n\n .
        {summary} \n\n"""

        if not persona:
            persona="A highly skilled data analyst who can come up with complex, insightful goals about data"

        user_prompt += f"""\n The generated goals SHOULD BE FOCUSED ON THE INTERESTS AND PERSPECTIVE of a '{persona} persona, who is insterested in complex, insightful goals about the data. \n"""
        SYSTEM_INSTRUCTIONS = """
        You are a an experienced data analyst who can generate a given number of insightful GOALS about data, when given a summary of the data, and a specified persona. The VISUALIZATIONS YOU RECOMMEND MUST FOLLOW VISUALIZATION BEST PRACTICES (e.g., must use bar charts instead of pie charts for comparing quantities) AND BE MEANINGFUL (e.g., plot longitude and latitude on maps where appropriate). They must also be relevant to the specified persona. Each goal must include a question, a visualization (THE VISUALIZATION MUST REFERENCE THE EXACT COLUMN FIELDS FROM THE SUMMARY), and a rationale (JUSTIFICATION FOR WHICH dataset FIELDS ARE USED and what we will learn from the visualization). Each goal MUST mention the exact fields from the dataset summary above
        """

        FORMAT_INSTRUCTIONS = """
        THE OUTPUT MUST BE A CODE SNIPPET OF A VALID LIST OF JSON OBJECTS. IT MUST USE THE FOLLOWING FORMAT:

        ```[
            { "index": 0,  "question": "What is the distribution of X", "visualization": "histogram of X", "rationale": "This tells about "} ..
            ]
        ```
        THE OUTPUT SHOULD ONLY USE THE JSON FORMAT ABOVE.
        """
        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "assistant",
             "content":
             f"{user_prompt}\n\n {FORMAT_INSTRUCTIONS} \n\n. The generated {n} goals are: \n "}]
        print("GEN GOAL PROMPT:", messages)
        # goals = json.load(open('viz-cars-data/data/json_string_goals.json'))
        json_string = openai.ChatCompletion.create(model="gpt-3.5-turbo", \
                                                  temperature=0.2, \
                                                  messages=messages).choices[0].message.content
        try:
            goals = json.loads(json_string)

        except json.decoder.JSONDecodeError:
            print("The model did not return a valid JSON object while attempting generate goals. Instead gave:")
            print(goals)
        return goals

    def visualize(self, summary, goals, file_name):
        i = 1
        for goal in goals:
            visualization = goal["visualization"]
            question = goal["question"] 
            library = 'matplotlib'
            system_prompt = """You are a helpful assistant highly skilled in writing PERFECT code for visualizations. Given some 
            code template, you complete the template to generate a visualization given the dataset and the goal described. 
            The code you write MUST FOLLOW VISUALIZATION BEST PRACTICES ie. meet the specified goal, apply the right 
            transformation, use the right visualization type, use the right data encoding, and use the right aesthetics 
            (e.g., ensure axis are legible). The transformations you apply MUST be correct and the fields you use MUST be 
            correct. The visualization CODE MUST BE CORRECT and MUST NOT CONTAIN ANY SYNTAX OR LOGIC ERRORS (e.g., it must 
            consider the field types and use them correctly). You MUST first generate a brief plan for how you would solve 
            the task e.g. what transformations you would apply e.g. if you need to construct a new column, what fields you would 
            use, what visualization type you would use, what aesthetics you would use, etc."""
            general_instructions = f"""If the solution requires a single value (e.g. max, min, median, first, last etc), ALWAYS 
            add a line (axvline or axhline) to the chart, ALWAYS with a legend containing the single value (formatted with 0.2F).
            If using a <field> where semantic_type=date, YOU MUST APPLY the following transform before using that column i) 
            convert date fields to date types using data[''] = pd.to_datetime(data[<field>], errors='coerce'), ALWAYS use  
            errors='coerce' ii) drop the rows with NaT values data = data[pd.notna(data[<field>])] iii) convert field to right 
            time format for plotting.  ALWAYS make sure the x-axis labels are legible (e.g., rotate when needed). Solve the
            task  carefully by completing ONLY the <imports> AND <stub> section. DO NOT WRITE  ANY CODE TO LOAD THE DATA. 
            The data is already loaded and available in the variable data. Given the dataset summary, the plot(data) 
            method should generate a {library} chart ({visualization}) that addresses this goal: {question}. """
            

            matplotlib_instructions = f" {general_instructions} DONOT include plt.show(). The plot method must return a matplotlib object (plt). Think step by step. \n"

            library_instructions = {
                    "role": "assistant",
                    "content": f"  {matplotlib_instructions}. Use BaseMap for charts that require a map. "}
            template = \
                    f"""
                    import matplotlib.pyplot as plt
                    import pandas as pd
                    <imports>
                    # plan -
                    def plot(data: pd.DataFrame):
                        <stub> # only modify this section
                        plt.title('{question}', wrap=True)
                        plt.savefig({self.project_dir} + 'viz-cars-data/data/plot.png')
                        return plt;
                        # No additional code beyond this line."""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": f"The dataset summary is : {summary} \n\n"},
                library_instructions,
                {"role": "user",
                "content":
                f"""Always add a legend with various colors where appropriate. The visualization code MUST only use data fields 
                that exist in the dataset (field_names) or fields that are transformations based on existing field_names). 
                Only use variables that have been defined in the code or are in the dataset summary. The response should have only the python code and no additional text. \
                I repeat, give the python code only for the function. NO ADDITIONAL CODE.  \n\n 
                \THE GENERATED CODE SOLUTION SHOULD BE CREATED BY MODIFYING THE SPECIFIED PARTS OF THE TEMPLATE BELOW \n\n {template} \n\n.
                The FINAL COMPLETED CODE BASED ON THE TEMPLATE above is ... \n\n"""}]
            print("PLOT PROMPT:", messages, "\n\n")
            python_viz_script = openai.ChatCompletion.create(model="gpt-3.5-turbo", \
                                                            temperature=0.2, \
                                                            messages=messages).choices[0].message.content
            print("python viz script: ", python_viz_script)
            f = self.project_dir + "tmp" + str(i) + ".py"
            with open(f, "w+") as file:
                file.write(python_viz_script)
            if i==1:
                from tmp1 import plot
                plot(self.df)
                os.rename(self.project_dir + 'viz-cars-data/data/plot.png', self.project_dir + 'viz-cars-data/data/plot1.png')
            elif i==2:
                from tmp2 import plot
                plot(self.df)
                os.rename(self.project_dir + 'viz-cars-data/data/plot.png', self.project_dir + 'viz-cars-data/data/plot2.png')
            elif i==3:
                from tmp3 import plot
                plot(self.df)
                os.rename(self.project_dir + 'viz-cars-data/data/plot.png', self.project_dir + 'viz-cars-data/data/plot3.png')
            print("Plots saved in viz-cars-data/data directory")
            i += 1



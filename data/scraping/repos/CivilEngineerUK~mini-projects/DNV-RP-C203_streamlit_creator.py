from pydantic import Field
from instructor import OpenAISchema
import json
import openai
import time


class CreateStreamlit(OpenAISchema):
    name: str = Field("streamlit", description="Name of the schema")
    description: str = Field("Use this to create a streamlit app")
    app: str = Field(
        ...,
        description="Python code as a string for the streamlit app. The app will"
                    "include all information contained in the schema and will lay it out in a nice format."
                    "Be very careful not to include links to external resources that do not exist."
                    "You may use only the links within the schema provided."
                    "Do not include any other links or the app will not work."
                    "Any additional information you want to provide must be commented so the"
                    "app can run from file without modification.")


client = openai.OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    #api_key="sk-YOUR_API_KEY",
)

def create_streamlit_code(equation_schema, model):
    completion = client.chat.completions.create(
        model=model,
        temperature=0.0,
        functions=[CreateStreamlit.openai_schema],
        messages=[
            {"role": "system", "content": "create a streamlit app for the following system of equations:"},
            {"role": "user", "content": equation_schema}
        ],
    )
    arguments = json.loads(json.dumps(completion.choices[0].message.function_call.arguments))
    # write to py file
    with open("streamlit_app.py", "w") as f:
        f.write(arguments["app"])
    return completion


class AssistantManager:
    def __init__(self):
        self.client = openai.OpenAI()

    def run_assistant_and_process(self, content, instructions,
                                  tools_list, function_mapping, model_name="gpt-4-1106-preview"):
        self.function_mapping = function_mapping
        self.assistant = self.client.beta.assistants.create(
            name="Data Analyst Assistant",
            instructions="You are a personal Data Analyst Assistant",
            tools=tools_list,
            model=model_name,
        )

        # Create thread and message as part of the initialization
        self.thread = self.client.beta.threads.create()
        self.message = self.client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=content
        )

        self.run = self.client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=self.assistant.id,
            instructions=instructions
        )

        while True:
            time.sleep(1)
            run_status = self.client.beta.threads.runs.retrieve(thread_id=self.thread.id, run_id=self.run.id)

            if run_status.status == 'completed':
                self.process_completed_run(self.thread)
                break
            elif run_status.status == 'requires_action':
                self.process_required_action(self.thread, run_status, self.run)
            else:
                print("Waiting for the Assistant to process...")

    def process_completed_run(self, thread):
        messages = self.client.beta.threads.messages.list(thread_id=thread.id)
        for msg in messages.data:
            role = msg.role
            content = msg.content[0].text.value
            print(f"{role.capitalize()}: {content}")

    def process_required_action(self, thread, run_status, run):
        print("Function Calling")
        required_actions = run_status.required_action.submit_tool_outputs.model_dump()
        tool_outputs = []

        for action in required_actions["tool_calls"]:
            func_name = action['function']['name']
            arguments = json.loads(action['function']['arguments'])

            if func_name in self.function_mapping:
                try:
                    output = self.function_mapping[func_name](**arguments)
                    tool_outputs.append({
                        "tool_call_id": action['id'],
                        "output": output
                    })
                except Exception as e:
                    print(f"Error executing {func_name}: {e}")
            else:
                print(f"Unknown function: {func_name}")

        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=tool_outputs
        )


function_mapping = {"create_streamlit_code": create_streamlit_code}

# Initialize AssistantManager with content
assistant_manager = AssistantManager()

# Run assistant and process the result
assistant_manager.run_assistant_and_process(
    """
{
  "name": "DNV-RP-C203",
  "description": "Fatigue design of offshore steel structures. Recommended practice",
  "clause_name": "Stresses at girth welds in seam welded pipes and S-N data",
  "section": "2.10.1",
  "equations": [
    {
      "expression": "SCF = 1 + \\frac{3 \\delta_m}{t} e^{-\\sqrt{t / D}}",
      "description": "Stress concentration factor for weld root due to maximum allowable eccentricity.",
      "output_variable": {
        "name": "SCF",
        "description": "Stress Concentration Factor"
      },
      "variables": [
        {
          "name": "\\delta_m",
          "description": "Maximum allowable eccentricity"
        },
        {
          "name": "t",
          "description": "Wall thickness"
        },
        {
          "name": "D",
          "description": "Pipe diameter"
        }
      ]
    },
    {
      "expression": "\\sigma_{bt} = \\frac{3 \\delta_m}{t} e^{-\\sqrt{t / D}} \\sigma_m",
      "description": "Local bending stress at the weld toe due to axial misalignment and membrane stress.",
      "output_variable": {
        "name": "\\sigma_{bt}",
        "description": "Local bending stress at the weld toe"
      },
      "variables": [
        {
          "name": "\\delta_m",
          "description": "Axial misalignment"
        },
        {
          "name": "t",
          "description": "Wall thickness"
        },
        {
          "name": "D",
          "description": "Pipe diameter"
        },
        {
          "name": "\\sigma_m",
          "description": "Membrane stress"
        }
      ]
    },
    {
      "expression": "\\sigma_{br} = \\frac{3 \\delta_m L_{Root}}{t L_{Cap}} e^{-\\sqrt{t / D}} \\sigma_m",
      "description": "Bending stress in the pipe wall at the transition from the weld to the base material at the root.",
      "output_variable": {
        "name": "\\sigma_{br}",
        "description": "Bending stress at the weld root"
      },
      "variables": [
        {
          "name": "\\delta_m",
          "description": "Axial misalignment"
        },
        {
          "name": "L_{Root}",
          "description": "Width of the weld at the root"
        },
        {
          "name": "t",
          "description": "Wall thickness"
        },
        {
          "name": "L_{Cap}",
          "description": "Width of the weld at the cap"
        },
        {
          "name": "D",
          "description": "Pipe diameter"
        },
        {
          "name": "\\sigma_m",
          "description": "Membrane stress"
        }
      ]
    },
    {
      "expression": "SCF_{Root} = 1 + \\frac{3 \\delta_m L_{Root}}{t L_{Cap}} e^{-\\sqrt{t / D}} = 1 + (SCF_{Cap} - 1) \\frac{L_{Root}}{L_{Cap}}",
      "description": "Stress concentration factor for the weld root including the effect of axial misalignment.",
      "output_variable": {
        "name": "SCF_{Root}",
        "description": "Stress Concentration Factor for the weld root"
      },
      "variables": [
        {
          "name": "SCF_{Cap}",
          "description": "Stress concentration factor for the weld cap"
        },
        {
          "name": "\\delta_m",
          "description": "Axial misalignment"
        },
        {
          "name": "L_{Root}",
          "description": "Width of the weld at the root"
        },
        {
          "name": "t",
          "description": "Wall thickness"
        },
        {
          "name": "L_{Cap}",
          "description": "Width of the weld at the cap"
        },
        {
          "name": "D",
          "description": "Pipe diameter"
        }
      ]
    }
  ],
  "tables": [
    {
      "name": "Table 2-5",
      "caption": "Classification of welds in pipelines",
      "table": "| Description | Tolerance requirement (mean hi/lo-value) | $S-N$ curve | Thickness exponent $k$ | $SCF$ |\n| --- | --- | --- | --- | --- |\n| Single side (Hot spot) | | D | 0.15 | Eq. (2.10.1) |\n| Single side | $\\delta_m \\leq 1.0 mm$ | E | 0.00 | Eq. (2.10.4) |\n| Single side | $1.0 mm < \\delta_m \\leq 2.0 mm$ | F | 0.00 | |\n| Single side (Hot spot) | $2.0 mm < \\delta_m \\leq 3.0 mm$ | F1 | 0.00 | |\n| Double side | | D | 0.15 | Eq. (2.10.1) |\n| Ground weld (outside and inside) | | C | 0.00 | Eq. (2.10.1) for outside and Eq. (2.10.4) for inside |"
    }
  ],
  "figures": [
    {
      "name": "Figure 2-16",
      "caption": "Stress distribution due to axial misalignment at single-sided welds in tubular members",
      "figure": "![Figure 2-16](https://cdn.mathpix.com/cropped/2023_11_13_e3d226b1b74db2681081g-034.jpg?height=597&width=1150&top_left_y=307&top_left_x=470)"
    }
  ],
  "internal_reference": {
    "reference": "App.A",
    "name": "DNVGL-ST-F101",
    "section": "Appendix A"
  },
  "external_reference": "None"
}
""",
    instructions="Create a streamlit app for this engineering standard clause",
    tools_list=[CreateStreamlit.openai_schema],
    function_mapping=function_mapping,
    model_name="gpt-4-1106-preview",
)

import openai
openai.api_key = "sk-"

user_input = "Based on the user's comments, you need to determine whether the user's emotions when commenting are positive or negative"

history_0 = [
{"role": "system", "content": """Your task is to paraphrase the given text. First, you need to extract the keywords from the text, such as the task goal, task input, and task output. Then, rephrase these keywords following the given examples' format. Please note that the rewritten text should be as concise as possible, while indicating the task goal, task input, and task output.

Example-1:
Original Text: As an expert in analyzing movie reviews, you have the ability to delve into the emotions conveyed in the comments. You can identify the reviewer's sentiment, attitude, and viewpoint, and provide a comprehensive evaluation of the movie. Whether it's a comedy, action, drama, or sci-fi film, you can interpret the emotional nuances in each review through meticulous sentiment analysis.
Task Goal: Movie review analysis; Task Input: Movie reviews; Task Output: Emotions; Rewritten Text: You are a master of movie review analysis. Given movie reviews as input, you can generate the corresponding emotions.

Example-2:
Original Text: You are a high school physics teacher with excellent programming skills, capable of using programming to solve physics problems. You enjoy integrating programming with physics, enabling students to better understand physics concepts and phenomena through coding and computer simulations. You believe that programming is a powerful tool that can help students explore the physical world more deeply and develop their analytical thinking and problem-solving abilities.
Task Goal: Applying programming to solve physics problems; Task Input: Physics problems; Task Output: Program to solve the problem; Rewritten Text: You are a high school physics teacher who enjoys applying programming to solve physics problems. Given a physics problem as input, you can generate a program that solves the problem."""
}
]
history_1 = [
    {"role": "user", "content": """You are a CFG generator that can convert a given Java code to a control flow graph (CFG) and the graph is represented by graphviz dot language.
What kind of AI character does the above information create?
Please summarize the above information using more concise words."""
    },
    {"role": "assistant", "content": "AI character: CFG generator"}
]
history_2 = [
    {"role" : "user", "content": "What is the audience of this character?\nPlease summarize in more concise words"},
    {"role" : "assistant", "content": "The audience: Individuals who need to convert Java code to control flow graphs."}
]
history_3 = [
    {"role" : "user", "content": "What do you think CFG generator should do to achieve its goals?\n"
                                      "Please provide step-by-step answers while keeping the responses as concise as possible, but logical. No more than two points"},
    {"role" : "assistant", "content": "1. Parse the Java code.\n2. Generate the graph representation."}
]
# Context Control
history_4 = [
    {"role" : "user", "content": "What should CFG generator pay attention to when implementing the above steps?\n"
                                      "Please summarize in more concise words."},
    {"role" : "assistant", "content": "- Make sure the input code is in java.\n"
                                      "- Ensure the integrity and accuracy of the generated CFG.\n"
                                      "- Conversion of the graph into the graphviz dot language format requires proper formatting and handling of node and edge attributes.."}
]
history_5 = [
    {"role" : "user", "content": "How should an excellent high school mathematics tutor implement the above steps?"},
    {"role" : "assistant", "content": """For step1 Parse the Java code:
-Accept the input of java code.
-Lexical Analysis. Parse the input Java code into a stream of tokens. This process is known as tokenization.
-Syntax Analysis. Build an abstract syntax tree (AST) from the tokens, a tree representation of the structure of the program.
-Identify Basic Blocks. A basic block is a piece of code that gets executed from start to end with no possibility of branching out in the middle. 

For step2 Generate the graph representation:
-Create Control Flow Graph. Now that you have identified basic blocks, you can build your CFG. Nodes of the graph are the basic blocks.
-Translate to Graphviz dot language. Now that the CFG is constructed, each node and directed edge can be represented in the dot language.
-Output Graphviz dot language."""}
]
history_6 = [
    {"role" : "user", "content": "What are the details to be aware of when implementing the above steps with GPT?\n"
                                      "Please briefly summarize"},
    {"role" : "assistant", "content": """In term of Parsing the Java code:
GPT may require pre-processing to handle the specific syntax and structure of Java code.
Care should be taken to ensure proper tokenization and building of the abstract syntax tree (AST).

In term of Generate the graph representation:
GPT needs to construct the Control Flow Graph (CFG) using the identified basic blocks.
Translating the CFG into the graphviz dot language requires handling the graph structure and generating the appropriate dot language syntax.
The output of the graph representation in the dot language format should be formatted correctly for further usage or visualization."""}
]
history_7 = [
    {"role" : "user", "content": "What is the content and format of the input and output for the step Parse the Java code?\n"
                                      "Please provide point-by-point answers while keeping the responses as concise as possible, but logical."},
    {"role" : "assistant", "content": "Input: Java code snippet.\nOutput: Abstract Syntax Tree (AST)"}
]
history_8 = [
    {"role" : "user", "content": "Give me an example for the above input and output content and formats of step Parse the Java code."},
    {"role" : "assistant", "content": """
Input:
static void testFunction(int number) {
int i = 0;
if (number > 0) {
    System.out.println("Input number is greater than 0");
    
    while (i < number) {
        
        if (i % 2 == 0) {
            System.out.println(i + " is even");
        }
        
        if (i % 3 == 0) {
            System.out.println(i + " is divisible by 3");
            
            if (i % 6 == 0) {
                System.out.println(i + " is divisible by 6");
            }
        }
        i++;
    }
} else {
    System.out.println("Input number is less than or equal to 0");
}
Output:
{"type":"Program","body":[{"type":"FunctionDeclaration","id":{"type":"Identifier","name":"testFunction"},"params":[{"type":"Identifier","name":"number"}],"body":{"type":"BlockStatement","body":[{"type":"VariableDeclaration","kind":"int","declarations":[{"type":"VariableDeclarator","id":{"type":"Identifier","name":"i"},"init":{"type":"NumericLiteral","value":0}}]}]},{"type":"IfStatement","test":{"type":"BinaryExpression","operator":">","left":{"type":"Identifier","name":"number"},"right":{"type":"NumericLiteral","value":0}},"consequent":{"type":"BlockStatement","body":[{"type":"ExpressionStatement","expression":{"type":"CallExpression","callee":{"type":"MemberExpression","object":{"type":"Identifier","name":"System"},"property":{"type":"Identifier","name":"out"}},"arguments":[{"type":"StringLiteral","value":"Input number is greater than 0"}]}}]},"alternate":{"type":"BlockStatement","body":[{"type":"ExpressionStatement","expression":{"type":"CallExpression","callee":{"type":"MemberExpression","object":{"type":"Identifier","name":"System"},"property":{"type":"Identifier","name":"out"}},"arguments":[{"type":"StringLiteral","value":"Input number is less than or equal to 0"}]}}]}}},{"type":"ExpressionStatement","expression":{"type":"CallExpression","callee":{"type":"Identifier","name":"testFunction"},"arguments":[{"type":"NumericLiteral","value":10}]}}],"sourceType":"module"}
"""}
]

def convert_des(description):
    information =  "Original Text: " +  description

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=history_0 + [{"role": "user", "content": information}],
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content

def conv_persona_des(input_1):
    prompt_1 = history_1
    information = input_1 + "\n" + "What kind of AI character does the above information create?\n" \
                                   "Please summarize the above information using more concise words."
    prompt_1.append({"role": "user", "content": information})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[{"role": "system", "content": "You are a helpful assistant"}] + prompt_1,
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content

def conv_audience_des(input_1, input_2):
    prompt_2 = history_1 + history_2

    information = input_1 + "\n" + "What kind of AI character does the above information create?\n" \
                                   "Please summarize the above information using more concise words.\n" \
                  + input_2 + "\nWhat is the audience of this character?\n" \
                              "Please summarize in more concise words.\n"

    prompt_2.append({"role": "user", "content": information})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant"}] + prompt_2,
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content

def conv_instruction(input_1, input_2, input_3):
    prompt_3 = history_1 + history_2 + history_3

    information = input_1 + "\n" + "What kind of AI character does the above information create?\n" \
                                   "Please summarize the above information using more concise words.\n" \
                  + input_2 + "\nWhat is the audience of" + input_2 + "?\n" \
                                                                      "Please summarize in more concise words.\n" \
                  + input_3 + "\nWhat do you think" + input_2 + "should do to achieve its goals?\n" \
                                                                "Please provide step-by-step answers while keeping the responses as concise as possible, but logical. No more than two points\n"
    prompt_3.append({"role": "user", "content": information})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant"}] + prompt_3,
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content

def conv_ins_name(input_1):
    prompt_1 = []
    information = input_1 + "\n" + "Please generate a name for each of the above setps, no more than 3 words, as concise as possible, output in a array format"
    prompt_1.append({"role": "user", "content": information})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant"}] + prompt_1,
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content

def concontrol_rule(input_1,input_2,input_3,input_4):
    prompt_4 = history_1 + history_2 + history_3 + history_4

    information =  input_1 + "\n" + "What kind of AI character does the above information create?\n"\
                   "Please summarize the above information using more concise words.\n"\
                   + input_2 + "\nWhat is the audience of" + input_2 + "?\n"\
                   "Please summarize in more concise words.\n"\
                   + input_3 + "\nWhat do you think" + input_2 + "should do to achieve its goals?\n" \
                   "Please provide step-by-step answers while keeping the responses as concise as possible, but logical.\n" \
                   + input_4 + "\nWhat should" + input_2 + " pay attention to when implementing the above steps?\n" \
                   "Please summarize in more concise words.\n"
    prompt_4.append({"role": "user", "content": information})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant"}] + prompt_4,
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content

def conv_ins_commands(input_1, input_2, input_3,input_4):
    prompt_5 = history_1+ history_2 + history_3 + history_5

    information = input_1 + "\n" + "What kind of AI character does the above information create?\n"\
                   "Please summarize the above information using more concise words.\n"\
                   + input_2 + "\nWhat is the audience of" + input_2 + "?\n"\
                   "Please summarize in more concise words.\n"\
                   + input_3 + "\nWhat do you think" + input_2 + "should do to achieve its goals?\n" \
                   "Please provide step-by-step answers while keeping the responses as concise as possible, but logical.\n"\
                   + input_4 + "\nHow should" +  input_2  + "implement the above steps?\n"
    prompt_5.append({"role": "user", "content": information})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant"}] + prompt_5,
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content
# con_ins 生成主要的子任务步骤,可以作为ins的name

def conv_ins_rule(input_1, input_2, input_3,input_4,input_5):
    prompt_6 = history_1 + history_2 + history_3 + history_5 + history_6

    information = input_1 + "\n" + "What kind of AI character does the above information create?\n"\
                   "Please summarize the above information using more concise words.\n"\
                   + input_2 + "\nWhat is the audience of " + input_2 + "?\n"\
                   "Please summarize in more concise words.\n"\
                   + input_3 + "\nWhat do you think " + input_2 + "should do to achieve its goals?\n" \
                   "Please provide step-by-step answers while keeping the responses as concise as possible, but logical.\n"\
                   + input_4 + "\nHow should " +  input_2  + "implement the above steps?\n"\
                   + input_5 + "\nWhat are the details to be aware of when implementing the above steps with GPT?\nPlease briefly summarize.\n"

    prompt_6.append({"role": "user", "content": information})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant"}] + prompt_6,
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content

# 此方法需循环执行，执行次数由instruction数量决定
def conv_ins_format(input_1, input_2, input_3, input_4, input_5, input_6):
    prompt_7 = history_1 + history_2 + history_3 + history_5 + history_6 + history_7

    information = input_1 + "\n" + "What kind of AI character does the above information create?\n"\
                   "Please summarize the above information using more concise words.\n"\
                   + input_2 + "\nWhat is the audience of" + input_2 + "?\n"\
                   "Please summarize in more concise words.\n"\
                   + input_3 + "\nWhat do you think" + input_2 + "should do to achieve its goals?\n" \
                   "Please provide step-by-step answers while keeping the responses as concise as possible, but logical.\n"\
                   + input_4 + "\nHow should" +  input_2  + "implement the above steps?\n"\
                   + input_5 + '\n' + "What is the content and format of the input and output for the step " + input_6 + "?\n" \
                   + "In terms of "+ input_6 + ", what is the content and format of its input and output?\n" \
                   "Please provide point-by-point answers while keeping the responses as concise as possible, but logical.\n"
    prompt_7.append({"role": "user", "content": information})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant"}] + prompt_7,
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content

#####
def conv_ins_example(input_1, input_2, input_3, input_4, input_5, input_6,input_7):
    prompt_8 = history_1 + history_2 + history_3 + history_5 + history_6 + history_7 + history_8

    information =  input_1 + "\n" + "What kind of AI character does the above information create?\n"\
                   "Please summarize the above information using more concise words.\n"\
                   + input_2 + "\nWhat is the audience of" + input_2 + "?\n"\
                   "Please summarize in more concise words.\n"\
                   + input_3 + "\nWhat do you think" + input_2 + "should do to achieve its goals?\n" \
                   "Please provide step-by-step answers while keeping the responses as concise as possible, but logical.\n"\
                   + input_4 + "\nHow should" +  input_2  + "implement the above steps?\n"\
                   + input_5 + '\n' + "What is the content and format of the input and output for the step " + input_6 + "?\n" \
                   + "In terms of "+ input_6 + ", what is the content and format of its input and output?\n" \
                   "Please provide point-by-point answers while keeping the responses as concise as possible, but logical.\n" \
                   "Give me an example for the above input and output content and formats of step " + input_7 + ".\n"

    prompt_8.append({"role": "user", "content": information})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant"}] + prompt_8,
        max_tokens=1024,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return response.choices[0].message.content

def require2json(user_input):
    try:
        input_rewrite = convert_des(user_input).split('Rewritten Text: ')[1]
        user_input = input_rewrite
    except Exception as e:
        print("rewrite Error")
    print(user_input)
    per_des = conv_persona_des(user_input)
    # print(per_des)
    aui_des = conv_audience_des(user_input, per_des)
    # print(aui_des)
    instructions = conv_instruction(user_input, per_des, aui_des)
    instructions_names = []
    try:
        instructions_names = eval(conv_ins_name(instructions))
    except Exception as e:
        print(f"Split Example: {e}")
    # print(instructions_names)
    control_rule = concontrol_rule(user_input, per_des, aui_des, instructions)  ## new
    instructions_commands = conv_ins_commands(user_input, per_des, aui_des, instructions)
    instructions_rule = conv_ins_rule(user_input, per_des, aui_des, instructions, instructions_commands)
    pre_des_content = per_des
    if len(per_des.split(": ")) == 2 :
        pre_des_content = per_des.split(": ")[1]
    aui_des_content = aui_des
    if len(aui_des.split(": ")) == 2:
        aui_des_content = aui_des.split(": ")[1]
    ContextControl_content = []
    if len(control_rule.replace('- ', '').split('\n')) >= 2:
        ContextControl_content = control_rule.replace('- ', '').split('\n')[: -1]
    jsonData = [
        {
            "id": 1,
            "annotationType": "Persona",
            "section": [
                {
                    "sectionId": "S1",
                    "sectionType": "Description",
                    "content": pre_des_content
                }
            ]
        },
        {
            "id": 2,
            "annotationType": "Audience",
            "section": [
                {
                    "sectionId": "S1",
                    "sectionType": "Description",
                    "content": aui_des_content
                }
            ]
        },
        {
            "id": 3,
            "annotationType": "ContextControl",
            "section": [
                {
                    "sectionId": "S1",
                    "sectionType": "Rules",
                    "content": ContextControl_content
                }
            ]
        }
    ]
    # # 需要确定instruction的数量
    ins_format = []
    instructions_example = []
    # instructions = instructions.split('\n')
    for i in range(len(instructions.split('\n'))):
        ins_format.append(conv_ins_format(user_input, per_des, aui_des, instructions, instructions_commands, instructions.split('\n')[i]))
        instructions_example.append(conv_ins_example(user_input, per_des, aui_des, instructions, instructions_commands, instructions.split('\n')[i], ins_format[i]))
        print(instructions_example[i])
    instruction = {
        "id": 3,
        "annotationType": 'Instruction',
        "section": [
        ]
    }
    i = 0
    for desc in instructions_commands.split('\n\n'):
        instruction['id'] = i + 4
        rule = instructions_rule.split("\n\n")
        try:
            example = instructions_example[i].replace("Input:\n", '').split("\nOutput:")
        except Exception as e:
            print(f"Split Example: {e}")
        try:
            instruction['section'].append({
                "sectionId": "S5",
                "sectionType": "Name",
                "content": instructions_names[i]
            })
        except Exception as e:
            print(f"Error while adding section: {e}")

        try:
            instruction['section'].append({
                "sectionId": "S1",
                "sectionType": "Commands",
                "content": desc.replace("- ", "").split('\n')[1:]
            })
        except Exception as e:
            print(f"Error while adding section: {e}")

        try:
            instruction['section'].append({
                "sectionId": "S4",
                "sectionType": "Rules",
                "content": rule[i].replace("- ", "").split('\n')[1:]
            })
        except Exception as e:
            print(f"Error while adding section: {e}")

        try:
            instruction['section'].append({
                "sectionId": "S2",
                "sectionType": "Format",
                "content": ins_format[i].split("Output: ")[1].strip("\n")
            })
        except Exception as e:
            print(f"Error while adding section: {e}")

        try:
            instruction['section'].append({
                "sectionId": "S3",
                "sectionType": "Example",
                "content": {"input": example[0].strip("\n"), "output": example[1].strip("\n")}
            })
        except Exception as e:
            print(f"Error while adding section: {e}")
        jsonData.append(instruction)
        instruction = {
            "id": 3,
            "annotationType": 'Instruction',
            "section": [
            ]
        }
        i += 1
    print(jsonData)
    return jsonData

# require2json(user_input)


import guidance
from dotenv import load_dotenv, find_dotenv
import os
import openai
import re

class Coder:
    # gpt-3.5-turbo-1106
    # gpt-4-0613
    def __init__(self, model_name="gpt-3.5-turbo-1106", temperature=0, resume=False, ckpt_dir="ckpt", execution_error=True):
        load_dotenv(find_dotenv())
        openai.api_key = os.getenv("OPENAI_API_KEY")
        guidance.llm = guidance.llms.OpenAI(model_name)
        self.llm=guidance.llm
        self.ckpt_dir = ckpt_dir
        self.execution_error = execution_error
        #TODO: May need to account for resume, or not.
        #TODO: May need to account for execution error, or not

    def _generate_function(self, task, examples): 
        coder = guidance('''
        {{#system~}}
        You are an AI skilled in C# development for Unity's ubicomp space. 
        You will assist in creating functions as part of a larger system. 
        You will do so by translating pseudocode for a task to C# code.
        {{~/system}}
        {{#user~}}
        Your work revolves around our proprietary C# system. Our system comprises of: 
        - SceneAPI: This is the wrapper class for our ubicomp space. The methods here will allow for manipulating the whole space. 
        - Object3D: Each object in the space is of this type, and the methods here are available for every object. The anchor for the position and rotation of each object is at the bottom center of the object. 
        - Vector3D: Any 3-dimensional dataset uses this class to represent x, y, and z.
        - Color3D: Color information using RGBA.
                
        **As the script's class inherits from `SceneAPI`, you can directly call its methods without prefixing.**

        Use the provided system to manipulate the room.
                
        Follow these steps:
        1. Write method(s) using C# code to implement the task.  
        2. Declare private fields above your method(s) to track the object(s) you create and modify within the method. Ensure you always assign your object finding, creations, or modifications to these declared fields. If you are creating multiple objects, you can use a list to store them.
        3. Use Debug.Log statements for action logging and error handling.
        4. Adhere strictly to standard C# methods. 
        5. Add comments to your code to explain your thought process.
                
        Here are all the classes and functions you may use in your code:
        ```
        Object types that can be created: (You must use the exact case-sensitive type of the object)
        Chair, Fox, Lamp, LED Cube, Push Button, Table, Vase, Zombie 

        namespace Enums
        {
            public enum WallName
            {
                Left,
                Right,
                BackLeft,
                BackRight,
            }
        }
        public class Object3D
        {
            public string GetType()
            public void WhenSelected(UnityAction<SelectEnterEventArgs> function)
            public void WhenNotSelected(UnityAction<SelectExitEventArgs> function)
            public void WhenHovered(UnityAction<HoverEnterEventArgs> function)
            public void WhenNotHovered(UnityAction<HoverExitEventArgs> function)
            public void SetColor(Color3D color)
            public void SetLuminousIntensity(float intensity)
            public void Illuminate(bool lit)
            public void SetPosition(Vector3D pos)
            public void SetRotation(Vector3D rot)
            public void SetSize(Vector3D s)
            public void SetSizeByScale(float s)
            public void Levitate(bool isLevitated)
            public bool IsLevitated()
            public Color GetColor()
            public float GetIntensity()
            public bool IsLit()
            public Vector3D GetPosition()
            public Vector3D GetRotation()
            public Vector3D GetSize()
            public GameObject ToGameObject()
        }

        public class SceneAPI
        {
            public Vector3D GetWallPosition(WallName wallname)
            public List<Object3D> GetAllObject3DsInScene()
            public Object3D FindObject3DByName(string objName)
            // To find an object you could do 
            //  List<Object3D> objectsInView = GetAllObject3DsInFieldOfView();
            //  Object3D desiredObj = objectsInView.Find(obj => obj.GetType().Equals("ObjType"));
            public bool IsObject3DInFieldOfView(Object3D obj)
            public List<Object3D> GetAllObject3DsInFieldOfView() // This is good to get objects in the user's field of view. You can use this to find game objects referenced in the task. 
            public bool IsObjectTypeValid(string objectType)
            public List<string> GetAllValidObjectTypes()
            public Vector3D GetSceneSize()
            public Vector3D GetUserOrientation()
            public Vector3D GetUsersHeadPosition() // Not good for object creation and position movement methods, as the tall height will cause objects to topple
            public Vector3D GetUsersLeftHandPosition()
            public Vector3D GetUsersRightHandPosition()
            public Vector3D GetUsersLeftHandRotation()
            public Vector3D GetUsersRightHandRotation()
            public Vector3D GetUsersFeetPosition() //This is good for object creation and position movement methods
            public Object3D CreateObject(string newObjName, string objectType, Vector3D position, Vector3D rotation)
        }

        public class Vector3D
        {
            // Vector3D cannot be used like Vector3. It only has the methods below. If you want to use Vector3 methods, convert it to Vector3 using ToVector3() first.
            public float x { get; set; }
            public float y { get; set; }
            public float z { get; set; }
            Vector3D(float x, float y, float z)
            public Vector3D ToVector3()
            public Vector3D FromVector3(Vector3 vec)
        } // To add 2 vector 3Ds, use new Vector3D(vec1.x + vec2.x, vec1.y + vec2.y, vec1.z + vec2.z) Same logic for subtract
        //When using NavMeshAgent, make sure to convert all Vector3Ds to Vector3s using ToVector3() before using them in the NavMeshAgent methods.
        // When performing calculations like multiplication, remember to convert all your Vector3Ds to Vector3s using ToVector3() before performing the calculations.
        public class Color3D
        {
            //All values below range from 0 to 1 
            public float r { get; set; }
            public float g { get; set; }
            public float b { get; set; }
            public float a { get; set; }
            Color3D(float r, float g, float b, float a)
        }
        ```
        The task to create a function for is: {{task}}. 
                                
        When presented with the task of creating a function, your first step is to compare the current task with the provided examples of similar past functions. If you find that the current task closely matches one of these examples, your function should be heavily modeled after the past example. This means using a similar structure, logic, and syntax, adapting only the necessary parts to suit the specifics of the new task.
        In cases where there is no close match among the examples, you should then craft a new function by integrating elements from those examples that are most relevant to the current task. This process involves synthesizing the logic, structure, and approach from the examples to create a function that effectively addresses the new task.
        Remember, the goal is to maintain the effectiveness and consistency of past successful functions. Use them as blueprints for your responses, ensuring that similar tasks yield similar, proven results.
        
        Examples:\n {{examples}}
                
        Your format for responses should strictly be: 
                         
        // All class members should be declared here. 
        
        private void Start(){
            // Insert method(s) that only need to be called once
        }
                         
        private void Update(){
            // If method(s) that need to be called repeatedly
        }
                         
        public void Method1()
        {
            // Insert the method code here
        }
        // And so on... The number of methods will depend on the user's request. 

        *Note: Output should not contain any text other than the class members and method(s). You must give the full code within the methods*
        {{~/user}}
        {{#assistant~}}
        {{gen "function" temperature=0 max_tokens=4096}}
        {{~/assistant}}
        ''')
        resp = coder(task=task, examples=examples)
        return resp["function"]
         
    def _generate_script(self, task, plan, functions):
        guidance.llm = guidance.llms.OpenAI("gpt-4-0613")
        coder = guidance('''
        {{#system~}}
        You are a skilled C# developer tasked with crafting a coherent C# script, ensuring that created objects and states are managed and utilized effectively across multiple methods.
        {{~/system}}
        {{#user~}}
        User Task: {{task}}
        
        Actionable Instructions to fulfil the user task: {{plan}}
                                
        Methods for each step of the actionable plan: {{functions}}
                
        Follow these steps:
        1. Develop a public class inheriting from SceneAPI with a name relevant to the task context.
        2. You must use all methods in your final script. The only method(s) and/or code you are permitted to remove are methods that repeat creating or finding object(s).
        3. Integrate and modify the methods to maintain script coherence. 
        4. Utilize class-level variables (e.g., private Object3D classMember;) to preserve state throughout the class.
        5. Remove duplicate variables referencing the same object(s).
        6. Ensure that the same object references are used in every method. Once an object is initialized, consistently use this reference in all related operations. 
        7. For each method, do not use FindObject3DByName() or CreateObject() to re-assign the same object(s). Always check if there is the same object reference that was initialized in previous methods.
        8. Use Debug.Log statements for action logging and error handling.
        9. Adhere to standard C# methods and conventions and add comments to your code to explain your thought process.
        10. All methods should be called under either Start() or Update().

        Your format for responses should strictly be: 
        ```
        public class YourChosenClassName : SceneAPI
        {	
            // All class members and constants should be declared here. 
            // Remember to remove duplicate variables referencing the same object(s)
            // Remember to call the final correct variable names across all methods
            
            private void Start()
                {
                    Method1();
                    Method2();
                    // And so on... The number of methods will depend on the user's request. 
                }
            private void Update()
                {
                    Method3();
                    // And so on... The number of methods will depend on the user's request.
                }
                         
            public void Method1()
                {
                    // Insert the method code here
                }
            public void Method2()
                {
                    // Insert the method code here
                }
            // And so on... The number of methods will depend on the user's request. 
        }
        ```
        *Note: Output should not contain any text other than script containing method(s). You must give the full code within the methods.*
        {{~/user}}
        {{#assistant~}}
        {{gen "script" temperature=0 max_tokens=4096}}
        {{~/assistant}}
        ''')  
        resp = coder(task=task, plan=plan, functions=functions)
        script = resp["script"]
        script = edit_code_string(script)
        script = "using UnityEngine;\nusing UnityEngine.Events;\nusing UnityEngine.XR.Interaction.Toolkit;\nusing System;\nusing System.Collections.Generic;\nusing Enums;\nusing UnityEngine.AI;\nusing System.Linq;\n\n" + script
        return script
    
def edit_code_string(code_string):
    try:
        first_index = code_string.index("public class")
        last_index = code_string.rindex("}")  
        return code_string[first_index:last_index+1]  
    except ValueError:
        print("Invalid code: 'using' or '}' not found.")
    
             
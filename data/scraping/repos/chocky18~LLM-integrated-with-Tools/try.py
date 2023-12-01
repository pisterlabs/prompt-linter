from playwright.sync_api import sync_playwright
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
import json
import streamlit as st  
import time
from sys import argv, exit, platform

import os
os.environ["OPENAI_API_KEY"] = "your open ai key"   

black_listed_elements = set(["html", "head", "title", "meta", "iframe", "body", "script", "style", "path", "svg", "br", "::marker",])

class AGI:
    def __init__(self):
        self.current_url = None
        self.browser = (
            sync_playwright()
            .start()
            .chromium.launch(
                headless=False,
            )
        )

        self.page = self.browser.new_page()
        self.page.set_viewport_size({"width": 1280, "height": 1080})
        self.client = self.page.context.new_cdp_session(self.page)
        self.page_element_buffer = {}


        # Create the PlayWrightBrowserToolkit with the synchronized browser
        toolkit = PlayWrightBrowserToolkit.from_browser(self.browser)
        tools = toolkit.get_tools()

        # Initialize the agent
        llm = OpenAI(temperature=0.5)
        self.agent_chain = initialize_agent(tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    
    def go_to_page(self, url):
        self.page.goto(url=url if "://" in url else "http://" + url)
        self.client = self.page.context.new_cdp_session(self.page)
        self.page_element_buffer = {}

    def evaluate(self, response):
        # Implement your evaluation criteria here
        # Return True if the response is considered correct, False otherwise
        # You can use any criteria, such as checking for specific keywords or patterns
        # or comparing the response with expected outputs

        # Example: Check if the response contains the word "success"
        if "success" in response.lower():
            return True
        else:
            return False

    def extract_elements(self):
        """
        Extracts elements from a web page and generates selectors for those elements based on certain criteria.
        Returns a list of generated selectors.
        """
        page = self.page
        page_element_buffer = self.page_element_buffer
        start = time.time()

        page_state_as_text = []

        device_pixel_ratio = page.evaluate("window.devicePixelRatio")
        if platform == "darwin" and device_pixel_ratio == 1:  # lies
            device_pixel_ratio = 2

        win_scroll_x 		= page.evaluate("window.scrollX")
        win_scroll_y 		= page.evaluate("window.scrollY")
        win_upper_bound 	= page.evaluate("window.pageYOffset")
        win_left_bound 		= page.evaluate("window.pageXOffset") 
        win_width 			= page.evaluate("window.screen.width")
        win_height 			= page.evaluate("window.screen.height")
        win_right_bound 	= win_left_bound + win_width
        win_lower_bound 	= win_upper_bound + win_height
        document_offset_height = page.evaluate("document.body.offsetHeight")
        document_scroll_height = page.evaluate("document.body.scrollHeight")

        percentage_progress_start = 1
        percentage_progress_end = 2

        page_state_as_text.append(
            {
                "x": 0,
                "y": 0,
                "text": "[scrollbar {:0.2f}-{:0.2f}%]".format(
                    round(percentage_progress_start, 2), round(percentage_progress_end)
                ),
            }
        )

        tree = self.client.send(
            "DOMSnapshot.captureSnapshot",
            {"computedStyles": [], "includeDOMRects": True, "includePaintOrder": True},
        )
        strings	 	= tree["strings"]
        document 	= tree["documents"][0]
        nodes 		= document["nodes"]
        backend_node_id = nodes["backendNodeId"]
        attributes 	= nodes["attributes"]
        node_value 	= nodes["nodeValue"]
        parent 		= nodes["parentIndex"]
        node_types 	= nodes["nodeType"]
        node_names 	= nodes["nodeName"]
        is_clickable = set(nodes["isClickable"]["index"])

        text_value 			= nodes["textValue"]
        text_value_index 	= text_value["index"]
        text_value_values 	= text_value["value"]

        input_value 		= nodes["inputValue"]
        input_value_index 	= input_value["index"]
        input_value_values 	= input_value["value"]

        input_checked 		= nodes["inputChecked"]
        layout 				= document["layout"]
        layout_node_index 	= layout["nodeIndex"]
        bounds 				= layout["bounds"]

        cursor = 0
        html_elements_text = []

        child_nodes = {}
        elements_in_view_port = []

        anchor_ancestry = {"-1": (False, None)}
        button_ancestry = {"-1": (False, None)}

        def convert_name(node_name, has_click_handler):
            if node_name == "a":
                return "link"
            if node_name == "input":
                return "input"
            if node_name == "img":
                return "img"
            if (
                node_name == "button" or has_click_handler
            ):  # found pages that needed this quirk
                return "button"
            else:
                return "text"

        def find_attributes(attributes, keys):
            values = {}

            for [key_index, value_index] in zip(*(iter(attributes),) * 2):
                if value_index < 0:
                    continue
                key = strings[key_index]
                value = strings[value_index]

                if key in keys:
                    values[key] = value
                    keys.remove(key)

                    if not keys:
                        return values
            
            return values

        def add_to_hash_tree(hash_tree, tag, node_id, node_name, parent_id):
            parent_id_str = str(parent_id)
            if not parent_id_str in hash_tree:
                parent_name = strings[node_names[parent_id]].lower()
                grand_parent_id = parent[parent_id]

                add_to_hash_tree(
                    hash_tree, tag, parent_id, parent_name, grand_parent_id
                )

            is_parent_desc_anchor, anchor_id = hash_tree[parent_id_str]

            # even if the anchor is nested in another anchor, we set the "root" for all descendants to be ::Self
            if node_name == tag:
                value = (True, node_id)
            elif (
                is_parent_desc_anchor
            ):  # reuse the parent's anchor_id (which could be much higher in the tree)
                value = (True, anchor_id)
            else:
                value = (
                    False,
                    None,
                )  # not a descendant of an anchor, most likely it will become text, an interactive element or discarded

            hash_tree[str(node_id)] = value
            
            return value

        for index, node_name_index in enumerate(node_names):
            node_parent = parent[index]
            node_name = strings[node_name_index].lower()

            is_ancestor_of_anchor, anchor_id = add_to_hash_tree(
                anchor_ancestry, "a", index, node_name, node_parent
            )

            is_ancestor_of_button, button_id = add_to_hash_tree(
                button_ancestry, "button", index, node_name, node_parent
            )

            try:
                cursor = layout_node_index.index(
                    index
                )  # todo replace this with proper cursoring, ignoring the fact this is O(n^2) for the moment
            except:
                continue

            if node_name in black_listed_elements:
                continue

            [x, y, width, height] = bounds[cursor]
            x /= device_pixel_ratio
            y /= device_pixel_ratio
            width /= device_pixel_ratio
            height /= device_pixel_ratio

            elem_left_bound = x
            elem_top_bound = y
            elem_right_bound = x + width
            elem_lower_bound = y + height

            partially_is_in_viewport = (
                elem_left_bound < win_right_bound
                and elem_right_bound >= win_left_bound
                and elem_top_bound < win_lower_bound
                and elem_lower_bound >= win_upper_bound
            )

            if not partially_is_in_viewport:
                continue

            meta_data = []

            # inefficient to grab the same set of keys for kinds of objects but its fine for now
            element_attributes = find_attributes(
                attributes[index], ["type", "placeholder", "aria-label", "title", "alt"]
            )

            ancestor_exception = is_ancestor_of_anchor or is_ancestor_of_button
            ancestor_node_key = (
                None
                if not ancestor_exception
                else str(anchor_id)
                if is_ancestor_of_anchor
                else str(button_id)
            )
            ancestor_node = (
                None
                if not ancestor_exception
                else child_nodes.setdefault(str(ancestor_node_key), [])
            )

            if node_name == "#text" and ancestor_exception:
                text = strings[node_value[index]]
                if text == "|" or text == "•":
                    continue
                ancestor_node.append({
                    "type": "type", "value": text
                })
            else:
                if (
                    node_name == "input" and element_attributes.get("type") == "submit"
                ) or node_name == "button" or node_name == 'textarea':
                    node_name = "button"
                    element_attributes.pop(
                        "type", None
                    )  # prevent [button ... (button)..]
                
                for key in element_attributes:
                    if ancestor_exception:
                        ancestor_node.append({
                            "type": "attribute",
                            "key":  key,
                            "value": element_attributes[key]
                        })
                    else:
                        meta_data.append(element_attributes[key])
            # print("meta", meta_data)
            element_node_value = None

            if node_value[index] >= 0:
                element_node_value = strings[node_value[index]]
                if element_node_value == "|": #commonly used as a seperator, does not add much context - lets save ourselves some token space
                    continue
            elif (
                node_name == "input"
                and index in input_value_index
                and element_node_value is None
            ):
                node_input_text_index = input_value_index.index(index)
                text_index = input_value_values[node_input_text_index]
                if node_input_text_index >= 0 and text_index >= 0:
                    element_node_value = strings[text_index]

            # remove redudant elements
            if ancestor_exception and (node_name != "a" and node_name != "button"):
                continue

            elements_in_view_port.append(
                {
                    "node_index": str(index),
                    "backend_node_id": backend_node_id[index],
                    "node_name": node_name,
                    "node_value": element_node_value,
                    "node_meta": meta_data,
                    "is_clickable": index in is_clickable,
                    "origin_x": int(x),
                    "origin_y": int(y),
                    "center_x": int(x + (width / 2)),
                    "center_y": int(y + (height / 2)),
                }
            )
        # print("elements_in_view_port",elements_in_view_port)	
        # lets filter further to remove anything that does not hold any text nor has click handlers + merge text from leaf#text nodes with the parent
        final_selectors= []
        id_counter 			= 0

        for element in elements_in_view_port:
            node_index = element.get("node_index")
            node_name = element.get("node_name")
            node_value = element.get("node_value")
            is_clickable = element.get("is_clickable")
            origin_x = element.get("origin_x")
            origin_y = element.get("origin_y")
            center_x = element.get("center_x")
            center_y = element.get("center_y")
            meta_data = element.get("node_meta")

            inner_text = f"{node_value} " if node_value else ""
            meta = ""
            
            if node_index in child_nodes:
                for child in child_nodes.get(node_index):
                    entry_type = child.get('type')
                    entry_value= child.get('value')

                    if entry_type == "attribute":
                        entry_key = child.get('key')
                        meta_data.append(f'{entry_key}="{entry_value}"')
                    else:
                        inner_text += f"{entry_value} "
            # print("meta_data", meta_data)
            if meta_data:
                meta_string = " ".join(meta_data)
                meta = f" {meta_string}"

            if inner_text != "":
                inner_text = f"{inner_text.strip()}"

            converted_node_name = convert_name(node_name, is_clickable)

            # not very elegant, more like a placeholder
            if (
                (converted_node_name != "button" or meta == "")
                and converted_node_name != "link"
                and converted_node_name != "input"
                and converted_node_name != "img"
                and converted_node_name != "textarea"
                
            ) and inner_text.strip() == "":
                continue

            page_element_buffer[id_counter] = element

            if inner_text != "": 
                final_selectors.append(
                    f"""<{converted_node_name} id={id_counter}{meta}>{inner_text}</{converted_node_name}>"""
                )
            else:
                final_selectors.append(
                    f"""<{converted_node_name} id={id_counter}{meta}/>"""
                )
            id_counter += 1

        print("Parsing time: {:0.2f} seconds".format(time.time() - start))
        # print("elements_of_interest", elements_of_interest)
        # elements_to_remove = ["<button id=8 I'm Feeling Lucky/>", '<text id=9>Google offered in:</text>', '<link id=10>हिन्दी</link>', '<link id=11>বাংলা</link>', '<link id=12>తెలుగు</link>', '<link id=13>मराठी</link>', '<link id=14>தமிழ்</link>', '<link id=15>ગુજરાતી</link>', '<link id=16>ಕನ್ನಡ</link>', '<link id=17>മലയാളം</link>', '<link id=18>ਪੰਜਾਬੀ</link>', '<text id=19>India</text>', '<link id=20>About</link>', '<link id=21>Advertising</link>', '<link id=22>Business</link>', '<link id=23>How Search works</link>', '<link id=24>Privacy</link>', '<link id=25>Terms</link>', '<text id=26>Settings</text>']
        # lst = [elem for elem in elements_of_interest if elem not in elements_to_remove]
        return final_selectors
        


    def execute_commands(self, objective,selectors):
        previous_command = ""
        actions = []  # Dynamic list to store actions

        def get_gpt_command(objective, url, previous_command):
            prompt = "OpenAI is an agent that controls the web browser using Playwright. It can perform actions like scrolling, clicking, typing, submitting, and navigating to new pages. The objective is to achieve a specific goal. To get closer to achieving the goal, start by submitting a search query to Google that will lead to the best page for accomplishing the objective. Once on that page, interact with it using appropriate actions to achieve the goal. Based on the given objective, issue the command you believe will help you get closer to achieving the goal.\nObjective: {}\nURL: {}\nPrevious Command: {}\n\n"
            response = self.agent_chain.run(input=prompt.format(objective, url, previous_command))
            return response
        
        # Run the agent to get the command from LLM
        response = get_gpt_command(objective=objective, url=self.current_url, previous_command=previous_command)
        previous_command = response["output"]

        if response["output"] == "exit":
            st.write("Exiting...")
        else:
            # Generate a response from the LLM model
            llm_response = self.agent_chain.run(response)["output"]

            # Evaluate the correctness of the response
            is_correct = self.evaluate(llm_response)

            # Provide feedback to the LLM model
            feedback = None

            # Existing code...

            if feedback is not None:
                # Provide feedback to the LLM model
                self.agent_chain.feedback(response, llm_response, feedback)

                if is_correct:
                    # Execute the command using Playwright
                    # Execute the command using Playwright
                    try:
                        # Check if the command requires interacting with elements
                        if "click" in llm_response or "type" in llm_response or "select" in llm_response:
                            elements = self.extract_elements()
                            print("Extracted elements:", elements)
                            
                            # Find a valid selector from final_selectors and replace it in llm_response
                            for element in elements:
                                for selector in selectors:
                                    # Replace {{selector}} with current selector
                                    updated_llm_response = llm_response.replace("{{selector}}", selector)

                                    # Check if the command contains 'type_text' and add typing and submitting
                                    if "type_text" in updated_llm_response:
                                        updated_llm_response = updated_llm_response.replace("type_text", f"{selector}.type('{text}'); {selector}.press('Enter');")
                                    try:
                                        self.page.evaluate(updated_llm_response)
                                        break  # Stop iterating if action succeeds
                                    except Exception:
                                        continue  # Try the next selector if action fails
                        else:
                            self.page.evaluate(llm_response)
                    except Exception as e:
                        st.error(f"Error executing command: {e}")



            # Display the LLM response
           
            response_data = json.loads(llm_response)
            llm_actions = response_data.get("actions", [])
            actions.extend(llm_actions)
            print(actions)

            # Display the actions in the interface
            st.subheader("Actions")
            for index, action in enumerate(actions):
                st.code(f"Action {index+1}:", language="json")
                st.json(action)

        # Close the browser
        self.browser.close()

def main():
    col1, col2 = st.columns([1, 20])

    # In the first column, display the logo image
    logo_image = "workplete.png"
    col1.image(logo_image, use_column_width=True)

    # In the second column, display the "AGI" heading
    col2.title("AGI")
    # st.title("Chatbot Interface")
    

    # Get user input or command from LLM
    user_input = st.text_input("Objective")
    submit = st.button("Send")

    thumbs_up = st.button("Thumbs up 👍")
    thumbs_down = st.button("Thumbs down 👎")


    if user_input and submit:
        # Create an instance of the Crawler class
        agi = AGI()

        # Extract elements and generate selectors
        selectors = agi.extract_elements()


        # Execute commands obtained from LLM
        agi.execute_commands(user_input,selectors)

if __name__ == "__main__":
    main()

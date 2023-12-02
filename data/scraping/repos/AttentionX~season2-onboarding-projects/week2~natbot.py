"""
https://github.com/nat/natbot
adapted by - Adam
"""
#!/usr/bin/env python3
#
# natbot.py
#
# Set OPENAI_API_KEY to your API key, and then run this from a terminal.
#

import re
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
import time
import sys
from sys import argv, exit, platform
import openai
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai_api

chatgpt = "gpt-3.5-turbo"
gpt4 = 'gpt-4'

# Load environment variables from .env file
load_dotenv()
# Get OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key == None:
    raise ValueError(
        "OpenAI API key not found. Please set the OPENAIAPI environment variable."
    )

openai_api = openai_api.OpenAI_API(model=chatgpt)

quiet = False
if len(argv) >= 2:
    if argv[1] == '-q' or argv[1] == '--quiet':
        quiet = True
        print(
            "Running in quiet mode (HTML and other content hidden); \n"
            + "exercise caution when running suggested commands."
        )

# 프롬프트에 추가한 명령에 대한 설명을 추가해주세요. 
prompt_template = """
You are an agent controlling a browser. You are given:

    (1) objectives that you are trying to achieve
    (2) the URL of your current web page
    (3) a simplified text description of what's visible in the browser window (more on that below)

You can issue this command:
    CLICK X - click on a given element. You can only click on links, buttons, and inputs! X is in form of integer and wrapped with link.
The format of the browser content is highly simplified; all formatting elements are stripped. Based on your given objective, issue whatever command you believe will get you closest to achieving your goal. Don't try to interact with elements that you can't see. Do not navigate to other website. If you have only on choice, click it right away. Whenever you see a modal, press the OK button.

Now, You are Ph.D GPT.

### Goal
Your Goal is to succesfully graduate with three requirements. 
- Passing qualification Exam. All PhD students must pass the qualifying exam in December.
- Writing three journal/conference papers. You are required to publish at least 3 journal papers to qualify your degree.
- Remain hope ranging from 1 ~ 100

### How to
- Simply make your choice at the beginning of each month. All outcomes are determined by the random number generator. Do not take them seriously. Sometimes the RNG can be brutal.
- Take account of "Items:" and "Status" panel for making next-step choices. HTML components wraps around [ITEMS] and [STATUS] below:
    - <div id="items_window" class="panel">
        <h3>Items:</h3><ul>[ITEMS]</ul></div>
    - <div id="status_window" class="panel">
        <h3>Status:</h3><ul>[STATUS]</ul></div>
- Click available options given at the each turn to play the game.
    - <div class="choices_container"><a class="btn" href="javascript: void(0)">[CHOICES]</a></div>
    - Whenever you see a modal, press the OK button.
- Let's think step-by-step. Decide based on previous choices history and outputs.

The current browser content, objective, and current URL follow. Reply with your next command to the browser.

CURRENT BROWSER CONTENT:
------------------
$browser_content
------------------

OBJECTIVE: $objective
PREVIOUS STATUS: $previous_status
PREVIOUS OPTIONS: $previous_options
PREVIOUS COMMAND: $previous_command
CURRENT STATUS: $current_status
AVAILABLE CHOICES: $available_choices
YOUR COMMAND:
"""

black_listed_elements = set(["html", "head", "title", "meta", "iframe", "body", "script", "style", "path", "svg", "br", "::marker",])

class Crawler:
    def __init__(self):
        self.browser = (
            sync_playwright()
            .start()
            .chromium.launch(
                headless=False,
            )
        )

        self.page = self.browser.new_page()
        self.page.set_viewport_size({"width": 1280, "height": 1080})

    def go_to_page(self, url):
        self.page.goto(url=url if "://" in url else "https://" + url)
        self.client = self.page.context.new_cdp_session(self.page)
        self.page_element_buffer = {}

    def click(self, id):
        # Inject javascript into the page which removes the target= attribute from all links
        js = """
        links = document.getElementsByTagName("a");
        for (var i = 0; i < links.length; i++) {
            links[i].removeAttribute("target");
        }
        """
        self.page.evaluate(js)

        element = self.page_element_buffer.get(int(id))
        if element:
            x = element.get("center_x")
            y = element.get("center_y")
            
            self.page.mouse.click(x, y)
        else:
            print("Could not find element")

    def enter(self):
        self.page.keyboard.press("Enter")

    def crawl(self):
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
                ) or node_name == "button":
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

        # lets filter further to remove anything that does not hold any text nor has click handlers + merge text from leaf#text nodes with the parent
        elements_of_interest= []
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
                elements_of_interest.append(
                    f"""<{converted_node_name} id={id_counter}{meta}>{inner_text}</{converted_node_name}>"""
                )
            else:
                elements_of_interest.append(
                    f"""<{converted_node_name} id={id_counter}{meta}/>"""
                )
            id_counter += 1

        print("Parsing time: {:0.2f} seconds".format(time.time() - start))
        
        find_idx = 0
        for idx, item in enumerate(elements_of_interest):
            if "©" in item:
                find_idx = idx
                break

        return elements_of_interest[:find_idx]



if (
    __name__ == "__main__"
):

    CLEANR = re.compile('<.*?>')
    def clean_html(raw_html):
        cleantext = re.sub(CLEANR, '', raw_html)
        return cleantext

    _crawler = Crawler()

    def print_help():
        print(
            "(g) to visit url\n(u) scroll up\n(d) scroll down\n(c) to click\n(t) to type\n" +
            "(h) to view commands again\n(r/enter) to run suggested command\n(o) change objective"
        )

    def get_gpt_command(objective, previous_status, previous_choice, previous_command, browser_content, current_status, available_choices):
        prompt = prompt_template
        prompt = prompt.replace("$objective", objective)
        prompt = prompt.replace("$previous_status", previous_status)
        prompt = prompt.replace("$previous_choice", previous_choice)
        prompt = prompt.replace("$previous_command", previous_command)
        prompt = prompt.replace("$browser_content", browser_content[:4500])
        prompt = prompt.replace("$current_status", current_status)
        prompt = prompt.replace("$available_choices", available_choices)
        response = openai_api.chatgpt(prompt)
        print(response)
        return response
    

    def run_cmd(cmd):
        """
        여기에 원하는 명령을 추가해주세요. (e.g. SAVE_TO_TXT, PRINT_ELEMENT)
        """
        cmd = cmd.split("\n")[0]

        if cmd.startswith("CLICK"):
            commasplit = cmd.split(",")
            id = commasplit[0].split(" ")[1]
            _crawler.click(id)
        else:
            pass
        time.sleep(1)

    objective = """Your Goal is to succesfully graduate with three requirements. 
    - Passing qualification Exam
    - Writing three journal/conference papers
    - Remain hope ranging from 1 ~ 100
    """
    print("\nWelcome to natbot! What is your objective?")
    i = input()
    if len(i) > 0:
        objective = i

    gpt_cmd = "" # current command
    prev_cmd = ""
    current_status = ""
    previous_status = ""
    current_choice = ""
    previous_choice = ""
    _crawler.go_to_page(f"https://research.wmz.ninja/projects/phd/index.html")
    try:
        while True:
            list_roi_text = _crawler.crawl() 
            browser_content = "\n".join(list_roi_text)

            # load previous runtime variables
            prev_cmd = gpt_cmd
            previous_status = current_status
            previous_choice = current_choice

            # overwrite runtime variable for status
            list_status = []
            list_choices = []
            for roi_idx, roi_text in enumerate(list_roi_text):
                if 'link' in roi_text:
                    list_choices.append(roi_text)
                elif 'text' in roi_text:
                    list_status.append(roi_text)
            current_status = " ".join(list_status)
            available_choices = " ".join(list_choices)
            # breakpoint()

            # overwrite runtime variable for current_choice
            if gpt_cmd != "":
                current_choice = clean_html(list_roi_text[int(gpt_cmd[-1])])
            
            # overwrite runtime variable for choice
            gpt_cmd = get_gpt_command(objective, previous_status, previous_choice, prev_cmd, browser_content, current_status, available_choices)
            gpt_cmd = gpt_cmd.strip()

            if not quiet:
                print("Objective: " + objective)
                print("----------------\n" + browser_content + "\n----------------\n")
            if len(gpt_cmd) > 0:
                print("Suggested command: " + gpt_cmd)


            command = input()
            if command == "r" or command == "":
                run_cmd(gpt_cmd)
            elif command == "g":
                url = input("URL:")
                _crawler.go_to_page(url)
            elif command == "u":
                _crawler.scroll("up")
                time.sleep(1)
            elif command == "d":
                _crawler.scroll("down")
                time.sleep(1)
            elif command == "c":
                id = input("id:")
                _crawler.click(id)
                time.sleep(1)
            elif command == "t":
                id = input("id:")
                text = input("text:")
                _crawler.type(id, text)
                time.sleep(1)
            elif command == "o":
                objective = input("Objective:")
            else:
                print_help()
    except KeyboardInterrupt:
        print("\n[!] Ctrl+C detected, exiting gracefully.")
        exit(0)
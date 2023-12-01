import openai
import os
import sys
from datetime import datetime
import json
import math
import textwrap
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter.scrolledtext as st

open_ai_key = "xx-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

current_dir = os.path.dirname(os.path.abspath(__file__))
starters_blank = False

def iter_submit(entry_iter):              # Handles submissions to second version of "SpellGPT setting" window (when "starters" is a blank field)
    global iter_submit
    global runs_per_it
    runs_per_it = abs(int(float(entry_iter.get())))
    root.update()
    root.destroy() 
    root.update()
    return runs_per_it


def submit():                   # Handles submissions to "SpellGPT settings" window
    global openai, token, base_prompt, starters, prompt, cutoff, engine, max_depth, save_subfolder, runs_per_it, starters_blank
    openai.api_key = entry1.get().strip()
    token = entry2.get().rstrip()

    base_prompt = f'Please spell the string "{token}" in all capital letters, separated by hyphens.\n'
    #base_prompt = f"Please spell the string '{token}' in all capital letters, separated by hyphens.\n"
    #base_prompt = f'I want you to spell out the string "{token}" in all capital letters, separated by hyphens.\n'
    #base_prompt = f"I want you to spell out the string '{token}' in all capital letters, separated by hyphens.\n"
    #base_prompt = f'Through the medium of spelling "{token}" in all capital letters, speak!\n'
    #base_prompt = f"Spell out the string '{token}' in all capital letters, separated by hyphens.\n"
    #base_prompt = f"Spell the string '{token}' in all capital letters, separated by hyphens.\n"
    #base_prompt = f"Hey you! Spell the string '{token}' in all capital letters, separated by hyphens!\n"
    #base_prompt = f"This is how you spell the string '{token}' in all capital letters, separated by hyphens:\n"
    #base_prompt = f"This is how you spell out the string '{token}' in all capital letters, separated by hyphens:\n"
    #base_prompt = f'Spell out the string "{token}" in all capital letters, separated by hyphens.\n'
    #base_prompt = f'Spell the string "{token}" in all capital letters, separated by hyphens.\n'
    #base_prompt = f'Hey you! Spell the string "{token}" in all capital letters, separated by hyphens!\n'
    #base_prompt = f'This is how you spell the string "{token}" in all capital letters, separated by hyphens:\n'
    #base_prompt = f'This is how you spell out the string "{token}" in all capital letters, separated by hyphens:\n'
    #base_prompt = f'''Please repeat the string '{token}' back to me immediately!\n"'''



    starters = entry3.get().strip()
    # Ideally a dropdown would present a number of suggested prompts emedding {token}
    prompt = base_prompt + starters
    cutoff = float(entry4.get())
    engine = entry5.get()
    max_depth = int(entry6.get())

    save_subfolder = (entry_save.get()).strip()
    #create a subdirectory with this name, in the curerent directory
    if not os.path.exists(save_subfolder):
        # If not, create the new directory
        os.mkdir(save_subfolder)
        os.mkdir(save_subfolder + '/JSON')
        os.mkdir(save_subfolder + '/images')
    else:
        if not os.path.exists(save_subfolder + '/images'):
            os.mkdir(save_subfolder + '/images')
        if not os.path.exists(save_subfolder + '/JSON'):
            os.mkdir(save_subfolder + '/JSON')

    if starters:
        root.destroy()                  # close the dialogue box
    else:
        starters_blank = True
        for widget in root.winfo_children():  # Delete existing widgets and build a new set
            widget.destroy()

        root.update()
        label_iter = tk.Label(root, text='''As you have entered a blank field for "starters", the prompt\n\n'The string "''' + token + '''" starts with the letter'\n\nwill be iterated over to harvest a specified number of single-letter outputs. The \nbranches from the root node will be based on this sampling.\nAll further generations will be generated solely by the top-five log\nprobs produced by the API for the current prompt-plus-extension.\n\nPlease enter a number of iterations. Larger numbers will cause some time delay\nin generation and be costlier in token use, but lead to more "accurate" results.''')
        label_iter.pack(pady=(100,20))
        entry_iter = tk.Entry(root, width = 5)
        entry_iter.insert(0, "250")
        entry_iter.pack()

        iter_submit_button = tk.Button(root, text='submit', command=lambda: iter_submit(entry_iter))
        iter_submit_button.pack(pady=(20,0))
        root.update()


# various default values assigned here:

node_name = ''
rota = 0
sprd = 1


tot_tok = 0 # Initialize the total token counter

spelltree_data = [{"level": 0, "letter": "", "weight": 1, "cumu_weight": 1, "children": []}]     # This is effectively the data for the root node before any children are hatched
node_dict_list = []
tree_json = {}
rotations = {}

ax = None

root = tk.Tk()
root.geometry("600x585")
root.title("SpellGPT settings")

label1 = tk.Label(root, text="OpenAI API key")
label1.pack(pady=(20,0))
entry1 = tk.Entry(root, width = 55)
entry1.insert(0, open_ai_key)
entry1.pack()

label5 = tk.Label(root, text="engine")
label5.pack(pady=(20,0))
entry5 = tk.Entry(root, width = 25)
entry5.insert(0, "davinci-instruct-beta")
entry5.pack()

label2 = tk.Label(root, text="token (including leading space!)")
label2.pack(pady=(20,0))
entry2 = tk.Entry(root, width = 25)
entry2.insert(0, " petertodd")
entry2.pack()

label3 = tk.Label(root, text="starters (capital letters, T-H-I-S-F-O-R-M-A-T-)")
label3.pack(pady=(20,0))
entry3 = tk.Entry(root, width = 55)
entry3.insert(0, "I-")
entry3.pack()

label4 = tk.Label(root, text="weight cutoff for branches")
label4.pack(pady=(20,0))
entry4 = tk.Entry(root, width = 8)
entry4.insert(0, "0.01")
entry4.pack()

label6 = tk.Label(root, text="maximimum letter depth per iteration")
label6.pack(pady = (20,0))
entry6 = tk.Entry(root, width = 3)
entry6.insert(0, "10")
entry6.pack()

label_save = tk.Label(root, text="subdirectory to save images and JSON to")
label_save.pack(pady = (20,0))
entry_save = tk.Entry(root, width = 55)
entry_save.insert(0, current_dir + "/SpellGPT_outputs")
entry_save.pack()

submit_button = tk.Button(root, text='submit', command=submit)
submit_button.pack(pady=(35,0))

root.mainloop()


def create_initial_widgets(root0, left_frame, node_dict_list, fig, cumu_weight):     
    for widget in left_frame.winfo_children():  # Delete existing widgets
        widget.destroy()

    # Create the initial widgets
    prompt_widget = st.ScrolledText(left_frame, width=30, height=16)
    prompt_widget.insert("insert", "ENGINE: " + engine + "\n\nPROMPT: " + prompt)
    prompt_widget['state'] = 'disabled'  # Make the widget read-only
    prompt_widget.pack(pady=(20,0))

    label9 = tk.Label(left_frame, text="Do you want to adjust this tree?")
    label9.pack(pady=(20,0))

    button_frame = tk.Frame(left_frame)  # create a new frame to hold the buttons
    button_frame.pack(pady=(20,0))  # pack the new frame into the left frame

    adjust_yes = tk.Button(button_frame, text='yes', command=lambda: adj_yes_clicked(root0, left_frame, node_dict_list, fig, cumu_weight))
    adjust_yes.pack(side=tk.LEFT, padx=5, pady=5)  # use side=tk.LEFT to pack the buttons next to each other

    adjust_no = tk.Button(button_frame, text='no', command=lambda: adj_done_clicked(root0, left_frame, node_dict_list, fig, cumu_weight))
    adjust_no.pack(side=tk.LEFT, padx=5, pady=5)


def adj_yes_clicked(root0, left_frame, node_dict_list, fig, cumu_weight):   # User has said "yes", they want to adjust the tree diagram, so they're here given the chance to submit three itmems of data

    global node_entry, rota_entry, sprd_entry, rotations, tree_json # Declare these as global variables

    for widget in left_frame.winfo_children():  # Delete existing widgets and build a new set
        widget.destroy()

    node_label = tk.Label(left_frame, text="node to adjust")
    node_label.pack(pady=(20,0))
    node_entry = tk.Entry(left_frame, width = 10)
    node_entry.insert(0, "*")
    node_entry.pack()

    rota_label = tk.Label(left_frame, text="rotation angle (degrees)")
    rota_label.pack(pady=(0,0))
    rota_entry = tk.Entry(left_frame, width = 5)
    rota_entry.insert(0, "0")
    rota_entry.pack()

    sprd_label = tk.Label(left_frame, text="branch spread factor")
    sprd_label.pack(pady=(20,0))
    sprd_entry = tk.Entry(left_frame, width = 5)
    sprd_entry.insert(0, "1")
    sprd_entry.pack()

    adj_sub_button = tk.Button(left_frame, text='submit', command=lambda: adj_submit_clicked(root0, cumu_weight))
    adj_sub_button.pack(side=tk.TOP, fill=tk.X, pady=(30,0), padx = (77,76))

    done_button = tk.Button(left_frame, text='done', command=lambda: adj_done_clicked(root0, left_frame, node_dict_list, fig, cumu_weight))
    done_button.pack(side=tk.TOP, fill=tk.X, pady=(20,0), padx = (77,76))


def adj_submit_clicked(root0, cumu_weight):                              # the tree-adjustment info triple has been submitted
    global node_name, rota, sprd, rotations, tree_json, ax, canvas
    node_name = node_entry.get()
    rota = rota_entry.get()
    sprd = sprd_entry.get()

    node = node_name.upper()
    if node == "":
        node = "*"

    try:
        rota = float(rota)
    except ValueError:
        rota = 0

    try:
        sprd = float(sprd)
    except ValueError:
        sprd = 1

    print("\nnode: " + node)
    print("rotational angle: " + str(rota))
    print("branch spread factor: " + str(sprd))

    if node == "*":
        if "*" in rotations.keys():
            rotations[node] = (rotations[node][0]+int(float(rota)), rotations[node][1]*float(sprd))
        else:
            rotations[node] = (int(float(rota)),float(sprd))
    else:
        if node in rotations.keys():
            rotations[node] = (rotations[node][0]+int(float(rota)), rotations[node][1]*float(sprd))
        elif node.isupper():
            rotations[node] = (int(float(rota)),float(sprd))

    print("current dictionary of nodal subtree (rotations, spread factors):")
    print(rotations) 
    print('\n\n')

    ax.clear()  # Clear the existing plot
    ax.axis('off')
    build_visual_tree(tree_json, node_dict_list = [], rotations=rotations, ax=ax)  # Generate the new plot
    canvas.get_tk_widget().config(highlightthickness=0, highlightcolor="white")
    canvas.draw()  # Redraw the new plot on the canvas


def adj_done_clicked(root0, left_frame, node_dict_list, fig, cumu_weight):    # user claims to be done with adjustments....just checking

    for widget in left_frame.winfo_children():  # Delete existing widgets and build a new set
        widget.destroy()

    left_frame.pack_propagate(0)  # or parent_frame.grid_propagate(0)

    left_frame = tk.Frame(root0, width=220)
    left_frame.grid(row=0, column=0, padx=10, pady=5)  
    left_frame.grid_propagate(False)

    done_buttons_label = tk.Label(left_frame, text="Are you definitely happy\n with the layout of this diagram?")
    done_buttons_label.pack(pady=(20,0))
    
    done_button_frame = tk.Frame(left_frame)  # create a new frame to hold the buttons
    done_button_frame.pack(pady=(20,0))  # pack the new frame into the left frame

    done_yes = tk.Button(done_button_frame, text='yes', command=lambda: done_yes_clicked(root0, left_frame, node_dict_list, fig, cumu_weight))
    done_yes.pack(side=tk.LEFT, padx=5, pady=5)  # use side=tk.LEFT to pack the buttons next to each other

    done_no = tk.Button(done_button_frame, text='no', command=lambda: adj_yes_clicked(root0, left_frame, node_dict_list, fig, cumu_weight))   # not, they're not happy, so yes, they want to adj(ust) it further
    done_no.pack(side=tk.LEFT, padx=5, pady=5)


def done_yes_clicked(root0, left_frame, node_dict_list, fig, cumu_weight):               # user definitely happy, time to save image
    global prompt_widget
    global rotations
    for widget in left_frame.winfo_children():     
        widget.destroy()

    rotations = {}


    now = datetime.now()
    datetime_str = now.strftime('%Y%m%d_%H%M%S')
    fig.savefig(save_subfolder + '/images/' + datetime_str + '_' + token.strip() + '_' + prompt[len(base_prompt):].replace('-','') + '_' + engine + '_spelltree.png')
    #       fig.savefig(save_subfolder + '/images/' + datetime_str + '_TOKEN47182_' + prompt[len(base_prompt):].replace('-','') + '_' + engine + '_spelltree.png')
     
    # Update the global prompt_widget variable
    prompt_widget = st.ScrolledText(left_frame, width=30, height=16)
    prompt_widget.insert("insert", "ENGINE: " + engine + "\n\nPROMPT: " + prompt)
    prompt_widget['state'] = 'disabled'  # Make the widget read-only
    prompt_widget.pack(pady=(20,0))

    ext_label = tk.Label(left_frame, text="node with which to extend prompt\n(or * to exit)")
    ext_label.pack(pady=(20,0))
    ext_entry = tk.Entry(left_frame, width = 10)
    ext_entry.insert(0, "")
    ext_entry.pack()

    ext_sub_button = tk.Button(left_frame, text='submit', command=lambda: ext_sub_clicked(ext_entry, root0, left_frame, node_dict_list, fig, cumu_weight))
    ext_sub_button.pack(pady=(35,0))


def ext_sub_clicked(ext_entry, root0, left_frame, node_dict_list, fig, cumu_weight):               # user submits a node with which to extend the prompt
    global prompt
    global prompt_widget

    ext_node = ext_entry.get()

    if ext_node == "*":
        print("\nGoodbye. All images and JSON files have been saved to " +  save_subfolder + ".\n\n")
        root.quit()
        root0.quit()
        sys.exit()
    else:
        legit_node = False
        ext_node = ext_node.strip().upper()

        for d in node_dict_list:
            if d['letter'] == ext_node:
                legit_node = True
        if not legit_node:
            print("\n" + ext_node + " is a not a valid node, please try again!\n")
            done_yes_clicked(root0, left_frame, node_dict_list, fig, cumu_weight)

    if legit_node:

        print("selected node: " + ext_node)

        prompt += hyphenise(ext_node)

        prompt_widget['state'] = 'normal'
        prompt_widget.delete("1.0", tk.END)  # Clear existing text
        prompt_widget.insert("insert", "ENGINE: " + engine + "\n\nPROMPT: " + prompt)
        prompt_widget['state'] = 'disabled'  # Make the widget read-only
        left_frame.update()

        path_to_follow = [letter for letter in ext_node] #navigate through tree_json to the right subtree node

        current_node = tree_json

        for letter in path_to_follow:
            current_node = next(child for child in current_node['children'] if child['letter'] == letter)

        nu_node = current_node.copy()
        subtract_levels(nu_node, nu_node['level'])
        nu_node['level'] = 0
        nu_node['weight'] = 1
        nu_node['letter'] = ''
        # we just want the node to have the relevant children, everything else should be reset

        nu_spelltree_data = [nu_node]
        root0.destroy()

        mainfunction(nu_spelltree_data)


def node_submit_clicked():
    global node_name
    node_name = entry1.get()
    text_box.insert(tk.END, 'Node to adjust: {}\n'.format(node_name))
    entry1.delete(0, tk.END)


def rota_submit_clicked():
    global rota
    rota = entry1.get()
    text_box.insert(tk.END, 'Rotation angle: {}\n'.format(rota))
    entry1.delete(0, tk.END)

def sprd_submit_clicked():
    global sprd
    spread = entry1.get()
    text_box.insert(tk.END, 'Branch spread factor: {}\n'.format(sprd))
    entry1.delete(0, tk.END)


def hyphenise(strg):    
    newstrg = ''
    if strg != '':
        for s in strg:
            newstrg += s + '-'
    return newstrg


def subtract_levels(node, level_diff):
    node['level'] -= level_diff
    for child in node['children']:
        subtract_levels(child, level_diff)


def build_spell_tree(token, engine, data, prompt, starters):                        # This takes a JSON spell tree ('data') and expands it recursiely
    global tot_tok # Use the tot_tok variable defined outside the function

    for child_dict in data:
        level = child_dict["level"]
        print("\n\nLevel = " + str(level))
        letter = child_dict["letter"]
        print("Current node: " + letter)
        h_letter = hyphenise(letter)
        current_prompt = prompt + h_letter

        if starters == "" and len(base_prompt) == len(current_prompt):   # to get spelling started, ask it for first letter 100 times
            first_letter_list = []
            print('\n\nThe string "' + token + '" starts with the letter')
            while len(first_letter_list) < runs_per_it:
                resp = openai.Completion.create(engine=engine, temperature = 1, prompt = 'The string "' + token + '" starts with the letter', max_tokens = 3)                
                comp = resp["choices"][0]["text"].lstrip().split(' ')[0]
                output_letters = [char for char in comp if char.isascii() and char.isalpha()]    
                if len(output_letters) == 1:   
                    first_letter_list.append(output_letters[0].upper())
                    print(output_letters[0].upper())

            d = Counter(first_letter_list)              # d will be a dictionary with letters as keys and numbers of occurence as values.
            first_letters = list(d.keys())          

            sorted_d = sorted(d.items(), key=lambda item: item[1], reverse=True)
            top_five_d = dict(sorted_d[:5])

            for first_letter in top_five_d.keys():
                proportion = top_five_d[first_letter]/runs_per_it
                if  proportion > cutoff:
                    child_dict["children"].append({"level": level + 1, "letter": first_letter, "weight": proportion, "cumu_weight": 1, "children": []})
                    print("first letter: " + first_letter)
                    print("proportion of occurence: " + str(proportion))

            cumu_weight = 1

            if level < max_depth:       
                build_spell_tree(token, engine, child_dict["children"], prompt, starters)         

 
        else:        # Once there's at least one letter there, we can just use logprobs provided by a single API call.
            next_letter_dict = {}
            print(current_prompt)

            response = openai.Completion.create(
                engine= engine,
                prompt=current_prompt,
                max_tokens=1,
                logprobs=100,  # get top 100 logprobs
            )

            tot_tok += response['usage']['total_tokens']


            if not response['choices']:  # add a condition to check if choices list is empty
                print("No choices in the response")
                continue
            elif not response['choices'][0]['logprobs']['top_logprobs']:
                print("No top logprobs in the response")
                continue
            else:    
                logprobs_dict = response['choices'][0]['logprobs']['top_logprobs'][0]


                for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":

                    logprob = logprobs_dict.get(letter, None)
                    logprob_lower = logprobs_dict.get(letter.lower(), None)
                    if (logprob is not None) or (logprob_lower is not None):
                        next_letter_dict[letter] =  0                           # make a key for each popular letter, value initialised to 0
                        print("letter: " + letter)

                        if logprob is not None:
                            print("uc prob: " + str(math.exp(logprob)))
                            next_letter_dict[letter] += math.exp(logprob)            # add probability for uppercase instance
                        if logprob_lower is not None:
                            print("lc prob: " + str(math.exp(logprob_lower)))
                            next_letter_dict[letter] += math.exp(logprob_lower)         # add probability for lowercase instance

                        prob = next_letter_dict[letter]
                        weight =  prob * child_dict["weight"]

                        cumu_weight = prob * child_dict["cumu_weight"]      #### THIS IS UPDATING CUMULATIVE WEIGHT FOR THIS PARTICULAR SPELLING PATHWAY
                                
                        if weight > cutoff:
                            existing_child = next((child for child in child_dict["children"] if child["letter"] == letter and child["level"] == level + 1), None)
                            if existing_child is not None:
                                existing_child["weight"] += weight   # WE'RE ADDING HERE TO TAKE INTO ACCOUNT UPPER AND LOWER CASE LETTERS
                            else:
                                child_dict["children"].append({"level": level + 1, "letter": letter, "weight": weight, "cumu_weight": cumu_weight, "children": []})

                if level < max_depth:       
                    build_spell_tree(token, engine, child_dict["children"], current_prompt, starters)

    return data


def build_visual_tree(node, node_dict_list = [], parent_x=0, parent_y=0, angle=90, scale=1, current_string='', prev_weight=50, is_root=True, level=0, rotations={}, ax=None):

    strg = current_string + node['letter']
    x = parent_x + scale * math.cos(math.radians(angle))    
    y = parent_y + scale * math.sin(math.radians(angle))
    
    weight = node.get('weight', 1)
    linewidth = weight * prev_weight

    # Minimum threshold for linewidth
    min_linewidth = 0.5
    if linewidth < min_linewidth:
        linewidth = min_linewidth

    node_patch = Circle((x, y), radius=0.05, facecolor='purple')

    node_dict_list.append({"patch": node_patch, "x": x, "y": y, "letter": strg})


    if not is_root:
        ax.plot([parent_x, x], [parent_y, y], linewidth=linewidth, color='turquoise', solid_capstyle='round')

    ax.text(x, y, strg, ha='center', va='center', color='white', bbox=dict(facecolor='purple', edgecolor='purple', boxstyle='round,pad=0.4'))

    if is_root:
        token_font_size = 1.5 * plt.rcParams['font.size']
        ax.text(x, y, "*", ha='center', va='center', fontsize=token_font_size, color='white', bbox=dict(facecolor='orange', edgecolor='orange', boxstyle='round,pad=0.4'))

    if strg in rotations.keys():
        rot, spread = rotations[strg]
    elif "*" in rotations.keys() and strg == "":
        rot, spread = rotations["*"]
    else:
        rot = 0
        spread = 1

    if 'children' in node:        # some very adhoc stuff going on here trying to get the trees not to be too messy... a work in progress
        num_children = len(node['children'])
        if num_children > 1:
            spacing_angle = 90 / (num_children - 1)
            angles = range(-45, 46, int(spacing_angle))
            if num_children > 3:
                spacing_angle = 120/num_children
                angles = range(int(spacing_angle*(num_children-1)/(-2)), int(spacing_angle * (num_children - 1)/2 + 2), int(spacing_angle))
            else:
                spacing_angle = 30
                angles = range(-15*(num_children -1), 15*(num_children - 1) +1, 30)

        else:
            angles = [0]

        for i, child in enumerate(node['children']):
            spread_angle = 1
            if num_children > 6:
                spread_angle = 1.7
            if is_root:                 # widening from the root node like thisseems to be the best way to get useful diagrams
                spread_angle = 3
            child_angle = angle + angles[i] * spread_angle * spread + rot
            child_scale = 0.95 * scale * (0.99 ** level)

            build_visual_tree(child, node_dict_list, parent_x=x, parent_y=y, angle=child_angle, scale=child_scale, current_string=strg,
                              prev_weight=linewidth, is_root=False, level=level + 1, rotations = rotations, ax = ax)

    return node_dict_list


def mainfunction(data):

    global rotations, ax, tree_json, node_dict_list, first_gen

    tree_json = build_spell_tree(token, engine, data, prompt, starters = starters)[0]

    #Save the result to a JSON file
    now = datetime.now()
    datetime_str = now.strftime('%Y%m%d_%H%M%S')
    with open(save_subfolder + '/JSON/' + datetime_str + '_' + token.strip() + '_' + prompt[len(base_prompt):].replace('-','') + '_' + engine + '_tree.json', 'w') as outfile:
    #with open(save_subfolder + '/JSON/' + datetime_str + '_TOKEN47182_' + prompt[len(base_prompt):].replace('-','') + '_' + engine + '_tree.json', 'w') as outfile:
            json.dump(tree_json, outfile)



    #print("\n\nTREE JSON:")
    #print(tree_json)
    print("\n\nPROMPT: ")
    print(prompt)
    print("\n\n")

    exten_len = int((len(prompt) - len(base_prompt) - len(starters))/2)
    
    # Create a Figure instance
    fig = Figure(figsize=(8, 8))

    # Add an Axes to this figure
    ax = fig.add_subplot(1, 1, 1)

    # You should modify your build_visual_tree to take an additional parameter, ax, and draw on this axes
    node_dict_list = build_visual_tree(tree_json, node_dict_list = [], rotations={}, ax=ax)

    
    cumu_weight = tree_json['cumu_weight']
    nodes = []
    for d in node_dict_list:
        if d['letter'] != "":
            nodes.append(d['letter'])
    print(f"NODES AVAILABLE: {nodes}")
    print("\n\n")
    print(f"CUMULATIVE PROBABILTY FOR THIS ROLLOUT: {cumu_weight}")
    print(f"(equivalently ~ 1 in {int(1/cumu_weight)})")
    print("\n\n")
    print(f"NUMBER OF CHARACTERS APPENDED: {exten_len}")
    if exten_len:
        print(f"CUMULATIVE PROBABILITY NORMALISED: {cumu_weight**(1/exten_len)}")
    print("\n\n")
    print("TOTAL TOKENS USED:")
    print(tot_tok)              # Print the total number of tokens used so far
    print("\n\n")

    if exten_len != 0:
        caption = "PROMPT: " + prompt + '\n\n' + 'NORMALISED CUMULATIVE PROBABILTY: ' + "{:.3f}".format(cumu_weight**(1/exten_len))
    else:
        caption = "PROMPT: " + prompt + '\n\n'

    # Split the original caption into lines
    lines = caption.split('\n')

    # Wrap each line with a maximum width of 100 characters
    wrapped_lines = []
    for line in lines:
        wrapper = textwrap.TextWrapper(width=75)
        word_list = wrapper.wrap(text=line)
        new_line = '\n'.join(word_list)
        wrapped_lines.append(new_line)

    # Join the wrapped lines, preserving original line breaks
    caption_new = '\n'.join(wrapped_lines)


    # Now we can call all the methods of ax instead of plt
    ax.axis('off')
    # I'm not sure how to translate subplots_adjust to ax, 
    # so let's stick with the original fig.subplots_adjust
    fig.subplots_adjust(bottom=0.2)

    # And add some text
    fig.text(0.1, 0.05, caption_new, wrap=True, horizontalalignment='left', fontsize=12)

    root0 = tk.Tk()
    root0.geometry("1050x800")
    root0.title("SpellGPT")

    left_frame = tk.Frame(root0, width=220)
    left_frame.grid(row=0, column=0, padx=10, pady=5)  
    left_frame.grid_propagate(False)

    create_initial_widgets(root0, left_frame, node_dict_list, fig, cumu_weight)

    right_frame = tk.Frame(root0)
    right_frame.grid(row=0, column=1, padx=10, pady=5, sticky='nsew')

    # create a canvas on the right frame
    global canvas
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.get_tk_widget().config(highlightthickness=0, highlightcolor="white")
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    root0.mainloop()

    return data


mainfunction(spelltree_data)
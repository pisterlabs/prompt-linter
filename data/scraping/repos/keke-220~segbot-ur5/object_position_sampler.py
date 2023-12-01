#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import openai
import math
import os, sys
import subprocess
import tempfile
import json
import re
import gc
import itertools
import copy
import textwrap
import random
#@title Customized parameters{display-mode: "form"}

#@markdown Input openai key for GPT-3 API
# api_key = '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'  #@param {type:"string"}
api_key = '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'
gpt_model = "text-davinci-003"  #@param ["text-davinci-003", "text-curie-001", "text-babbage-001", "text-ada-001"]

#@markdown select utensils to be placed on the table:
has_bread_plate = False  #@param {type:"boolean"}
has_butter_knife = False  #@param {type:"boolean"}
has_dinner_fork = True  #@param {type:"boolean"}
has_soup_spoon = False  #@param {type:"boolean"}
has_water_cup = False  #@param {type:"boolean"}
has_wine_glass = False  #@param {type:"boolean"}
has_napkin = False  #@param {type:"boolean"}
has_dinner_knife = True  #@param {type:"boolean"}
has_dinner_plate = True  #@param {type:"boolean"}
has_salad_fork = False  #@param {type:"boolean"}
has_dessert_fork = False  #@param {type:"boolean"}
has_dessert_spoon = False  #@param {type:"boolean"}
has_tea_spoon = False  #@param {type:"boolean"}
has_seafood_fork = False  #@param {type:"boolean"}
has_fish_knife = False  #@param {type:"boolean"}
has_place_mat = False  #@param {type:"boolean"}
has_salt_shaker = False  #@param {type:"boolean"}
has_pepper_shaker = False  #@param {type:"boolean"}

has_bread = False  #@param {type:"boolean"}
has_cup_mat = False  #@param {type:"boolean"}
has_strawberry = False  #@param {type:"boolean"}
has_fruit_bowl = False  #@param {type:"boolean"}
has_tea_cup = False  #@param {type:"boolean"}
has_tea_cup_lid = False  #@param {type:"boolean"}

# process the input parameters from users
utensils = [
    {
        "name": "bread plate",
        "has": has_bread_plate
    },
    {
        "name": "butter knife",
        "has": has_butter_knife
    },
    {
        "name": "dinner fork",
        "has": has_dinner_fork
    },
    {
        "name": "soup spoon",
        "has": has_soup_spoon
    },
    {
        "name": "water cup",
        "has": has_water_cup
    },
    {
        "name": "wine glass",
        "has": has_wine_glass
    },
    {
        "name": "napkin",
        "has": has_napkin
    },
    {
        "name": "dinner knife",
        "has": has_dinner_knife
    },
    {
        "name": "dinner plate",
        "has": has_dinner_plate
    },
    {
        "name": "salad fork",
        "has": has_salad_fork
    },
    {
        "name": "dessert fork",
        "has": has_dessert_fork
    },
    {
        "name": "dessert spoon",
        "has": has_dessert_spoon
    },
    {
        "name": "tea spoon",
        "has": has_tea_spoon
    },
    {
        "name": "seafood fork",
        "has": has_seafood_fork
    },
    {
        "name": "fish knife",
        "has": has_fish_knife
    },
    {
        "name": "place mat",
        "has": has_place_mat
    },
    {
        "name": "salt shaker",
        "has": has_salt_shaker
    },
    {
        "name": "pepper shaker",
        "has": has_pepper_shaker
    },
    {
        "name": "bread",
        "has": has_bread
    },  # new
    {
        "name": "cup mat",
        "has": has_cup_mat
    },
    {
        "name": "strawberry",
        "has": has_strawberry
    },
    {
        "name": "fruit bowl",
        "has": has_fruit_bowl
    },
    {
        "name": "tea cup",
        "has": has_tea_cup
    },
    {
        "name": "tea cup lid",
        "has": has_tea_cup_lid
    }
]
utensil_name = []  # save utensils' name
utensil_init_pose = {}  # save utensils' inital pose
for utensil in utensils:
    if utensil["has"]:
        utensil_name.append(utensil["name"])
        utensil_init_pose[utensil["name"]] = [[1.4, -0.65, 0.65], [0, 0, 0, 1]]
print('utensil to be placed on a dining table:\n{}'.format(utensil_name))

relationship_name = [
    'above', 'on the right of', 'below', 'on the left of',
    'above and to the right of', 'below and to the right of',
    'below and to the left of', 'above and to the left of', 'on top of',
    'under', 'center'
]
# print('\npredefined relationships between two utensils:\n{}'.format(relationship_name))

#@title Utensil size {display-mode: "form"}

#@markdown Input utensil size (unit of meter):
'''
utensil size by defalut
'''
# bread_plate_width_length = "0.14, 0.14" #@param {type:"string"}
# butter_knife_width_length = "0.19, 0.02" #@param {type:"string"}
# dinner_fork_width_length = "0.2, 0.03" #@param {type:"string"}
# soup_spoon_width_length = "0.2, 0.03" #@param {type:"string"}
# water_cup_width_length = "0.1, 0.1" #@param {type:"string"}
# wine_glass_width_length = "0.06, 0.06" #@param {type:"string"}
# napkin_width_length = "0.25, 0.1" #@param {type:"string"}
# dinner_knife_width_length = "0.22, 0.03" #@param {type:"string"}
# dinner_plate_width_length = "0.16, 0.16" #@param {type:"string"}
# salad_fork_width_length = "0.2, 0.03" #@param {type:"string"}
# dessert_fork_width_length = "0.17, 0.03" #@param {type:"string"}
# dessert_spoon_width_length = "0.17, 0.04" #@param {type:"string"}
# tea_spoon_width_length = "0.17, 0.04" #@param {type:"string"}
# seafood_fork_width_length = "0.2, 0.03" #@param {type:"string"}
# fish_knife_width_length = "0.22, 0.03" #@param {type:"string"}
# place_mat_width_length = "0.3, 0.3" #@param {type:"string"}
# salt_shaker_width_length = "0.06, 0.06" #@param {type:"string"}
# pepper_shaker_width_length = "0.06, 0.06" #@param {type:"string"}

bread_plate_width_length = "0.14, 0.14"  #@param {type:"string"}
butter_knife_width_length = "0.19, 0.05"  #@param {type:"string"}
dinner_fork_width_length = "0.2, 0.05"  #@param {type:"string"}
soup_spoon_width_length = "0.2, 0.05"  #@param {type:"string"}
water_cup_width_length = "0.1, 0.1"  #@param {type:"string"}
wine_glass_width_length = "0.06, 0.06"  #@param {type:"string"}
napkin_width_length = "0.1, 0.25"  #@param {type:"string"}
dinner_knife_width_length = "0.22, 0.05"  #@param {type:"string"}
dinner_plate_width_length = "0.16, 0.16"  #@param {type:"string"}
salad_fork_width_length = "0.2, 0.05"  #@param {type:"string"}
dessert_fork_width_length = "0.17, 0.05"  #@param {type:"string"}
dessert_spoon_width_length = "0.17, 0.05"  #@param {type:"string"}
tea_spoon_width_length = "0.17, 0.05"  #@param {type:"string"}
seafood_fork_width_length = "0.2, 0.05"  #@param {type:"string"}
fish_knife_width_length = "0.22, 0.05"  #@param {type:"string"}
place_mat_width_length = "0.3, 0.3"  #@param {type:"string"}
salt_shaker_width_length = "0.06, 0.06"  #@param {type:"string"}
pepper_shaker_width_length = "0.06, 0.06"  #@param {type:"string"}

bread_width_length = "0.1, 0.1"  #@param {type:"string"}
cup_mat_width_length = "0.1, 0.1"  #@param {type:"string"}
strawberry_width_length = "0.1, 0.1"  #@param {type:"string"}
fruit_bowl_width_length = "0.1, 0.1"  #@param {type:"string"}
tea_cup_width_length = "0.1, 0.1"  #@param {type:"string"}
tea_cup_lid_width_length = "0.1, 0.1"  #@param {type:"string"}

#@title Utils {display-mode: "form"}
openai.api_key = api_key
'''
Chatgpt API
'''
# chatbot = Chatbot(api_key)
# response = chatbot.ask(prompt)
# print("ChatGPT: " + response["choices"][0]["text"])

utensil_size = {}  # save size of utensils
pattern = "([0-9]*\.[0-9]*)"
alpha = 1.05
for utensil in utensil_name:
    variable_name = utensil.replace(" ", "_")
    matches = re.findall(pattern, eval(variable_name + "_width_length"))
    utensil_size[utensil] = [
        float(matches[0]) * alpha,
        float(matches[1]) * alpha
    ]


def count_tokens(text):
    return len(text.split())


def fee_tokens(text):
    if gpt_model == "text-davinci-003":
        return round(len(text.split()) / 1000.0 * 0.02, 3)
    elif gpt_model == "text-curie-001":
        return round(len(text.split()) / 1000.0 * 0.002, 3)
    elif gpt_model == "text-babbage-001":
        return round(len(text.split()) / 1000.0 * 0.0005, 3)
    elif gpt_model == "text-ada-001":
        return round(len(text.split()) / 1000.0 * 0.0004, 3)


def create_prompt_commonsense_abstract(input, zeroshot=True):
    '''
  @input: a list of several utensils, e.g., ['fork', 'plate']
  @output: a sentence of prompt, e.g., How to place a fork, and a plate on a table?
  '''
    utensils = ''
    for index in range(len(input)):
        if index < len(input) - 1:
            utensils += 'a ' + input[index] + ', '
        else:
            utensils += 'and a ' + input[index] + ' '
    if zeroshot:
        prompt_hint = 'The task is to set up a dining table. The relationship between two objects can only contain the following planar relationships: above, on the right of, below, on the left of, above and to the right of, below and to the right of, below and to the left of, above and to the left of, on top of, under, and center.\n\n'
    else:
        prompt_hint = 'The task is to set up a dining table. The relationship between two objects can only contain the following planar relationships: above, on the right of, below, on the left of, above and to the right of, below and to the right of, below and to the left of, above and to the left of, on top of, under, and center.\n\n###\nHow to place a water cup and a dinner plate on a table?\n1. Place the dinner plate in the center of the table.\n2. Place the water cup above and to the right of the dinner plate.\n\n###\n'
    prompt_question = 'How to place ' + utensils + 'on a table?'
    return prompt_hint + prompt_question


def create_prompt_commonsense_concrete(input):
    '''
  @input: object A, object B, spatial relationship, e.g., ['bread_plate', 'dessert fork']
  @output: a sentence of prompt
  '''
    prompt_hint = 'a ' + input[0] + ' is placed ' + input[2] + ' a ' + input[
        1] + '. '
    if input[2] == 'above' or input[2] == 'on the right of' or input[
            2] == 'below' or input[2] == 'on the left of' or input[
                2] == 'above':
        prompt_question = 'how many centimeters ' + input[2] + ' the ' + input[
            0] + ' should the ' + input[1] + ' be placed' + '?'
        return prompt_hint + prompt_question
    else:
        temp = input[2].split(" and ")
        prompt_question_1 = 'how many centimeters ' + temp[0] + ' the ' + item[
            0] + ' should the ' + item[1] + ' be placed' + '? '
        prompt_question_2 = 'how many centimeters ' + temp[1] + ' the ' + item[
            0] + ' should the ' + item[1] + ' be placed' + '?'
        return prompt_hint + prompt_question_1 + prompt_question_2


def extract_object_relationship(commonsense_abstract, utensil_name,
                                relationship_name):
    '''
  @commonsense_abstract: a sentence, e.g., 'place the plate in the center of the table.'; utensil_name and relationship_name are fixed
  @output: a sentence, e.g., 'object a: plate, object b: table, relationship: center'
  '''
    utensil_name = utensil_name + ['table'
                                   ]  # table is not in utensil_name by default
    # extract object A and B in commonsense_abstract
    objects = []
    for item in utensil_name:
        matches = re.finditer(item, commonsense_abstract)
        for match in matches:
            objects.append((item, match.start()))
    objects.sort(key=lambda x: x[1])

    # corner case: 'bread' and 'bread plate'; 'tea cup' and 'tea cup lid'
    # print('objects (before):{}'.format(objects))
    def check_same_number(list_a):
        temp = []
        for i in range(len(list_a)):
            for j in range(i + 1, len(list_a)):
                if list_a[i][1] == list_a[j][1]:
                    if len(list_a[i][0]) < len(list_a[j][0]):
                        temp.append(i)
                    else:
                        temp.append(j)
        list_a_temp = []
        for i in range(len(list_a)):
            if i not in temp:
                list_a_temp.append(list_a[i])
        return list_a_temp

    objects = check_same_number(objects)
    # print('objects (after):{}'.format(objects))

    if len(objects) == 2:  # commonsense_abstract has two objects
        object_A = objects[0][0]
        object_B = objects[1][0]
    elif len(objects) >= 3 or len(objects) <= 1:
        print("!Error: Expected 2 unique objects but found {}".format(objects))
        print("commonsense_abstract: {}".format(commonsense_abstract))
        sys.exit()

    # extract "relationship" from commonsense_abstract
    relationship = []
    for item in relationship_name:
        if re.search(item, commonsense_abstract):
            relationship.append(item)
    if len(relationship) > 1:  # processs corner cases
        if 'on' in relationship:
            relationship.remove('on')
        if 'above' in relationship:
            relationship.remove('above')
        if 'below' in relationship:
            relationship.remove('below')
        if 'to the left of' in relationship:
            relationship.remove('to the left of')
        if 'to the right of' in relationship:
            relationship.remove('to the right of')
    if len(relationship) >= 2 or len(relationship) <= 0:
        print("!Error: Expected 1 relationship but found {}".format(
            len(relationship)))
        print("commonsense_abstract: {}".format(commonsense_abstract))
        sys.exit()

    return object_A, object_B, relationship[0]


def extract_distance(commonsense_concrete, type):
    '''
  @commonsense_concrete: a sentence, e.g., 'the distance between the bread plate and the plate will depend on the size of the plates. generally, the distance between the two plates should be around 10-15 centimeters.'
  @output: a distance value, e.g., 14
  '''
    res = re.findall(r'\d+', commonsense_concrete)
    if type == 'type1':  #'above' or 'below'
        if len(res) not in (1, 2):
            print("!Warning: Expected 1-2 values but found {}".format(
                len(res)))
            return 5, 0
        if len(res) == 1:
            return int(res[0]), 0
        return random.randint(int(res[0]), int(res[1])), 0
    if type == 'type2':  # 'on the right of' or 'on the left of'
        if len(res) not in (1, 2):
            print("!Warning: Expected 1-2 values but found {}".format(
                len(res)))
            return 0, 5
        if len(res) == 1:
            return 0, int(res[0])
        return 0, random.randint(int(res[0]), int(res[1]))
    if type == 'type3':  # 'above and to the right of' or 'below and to the right of' or 'below and to the left of' or 'above and to the left of':
        if len(res) not in (2, 3, 4):
            print("!Warning: Expected 2-4 values but found {}".format(
                len(res)))
            return 5, 5
        result = []
        for match in re.finditer(r'\d+(?:-\d+)?', commonsense_concrete):
            group = match.group()
            if '-' in group:
                result.append(list(map(int, group.split('-'))))
            else:
                result.append([int(group)])
            numbers = []
            for item in result:
                if len(item) == 1:
                    numbers.append(item[0])
                elif len(item) == 2:
                    numbers.append(random.randint(item[0], item[1]))
        return numbers[0], numbers[1]


def llm(prompt):  # call LLM
    # gpt_model = 'text-davinci-003'
    sampling_params = {
        "n": 1,
        "max_tokens": 1024,
        "temperature": 0.0,
        "top_p": 1,
        "logprobs": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "stop": ['\\n']
    }
    raw_response = openai.Completion.create(engine=gpt_model,
                                            prompt=prompt,
                                            **sampling_params)
    responses = [
        raw_response['choices'][i]['text'] for i in range(sampling_params['n'])
    ]
    mean_probs = [
        math.exp(
            np.mean(raw_response['choices'][i]['logprobs']['token_logprobs']))
        for i in range(sampling_params['n'])
    ]
    responses = [sample.strip().lower() for sample in responses]
    return responses[0], mean_probs[0]


def relationship_objectAB(object_A, object_B, pose_A, pose_B, relationship,
                          distance):
    '''
  assume object_B is known
  compute the object_A pose accprding to object_B pose, relationship and distance
  '''
    pose_A.update(pose_B)
    # above, on the right of, below, on the left of, above and to the right of, below and to the right of, below and to the left of, above and to the left of, on, under, and center
    if 'above' == relationship:
        pose_A['x'] = pose_B['x'] + float(distance[0]) / 100. + utensil_size[
            object_A][0] * 0.5 + utensil_size[object_B][0] * 0.5
    elif 'on the right of' == relationship:
        pose_A['y'] = pose_B['y'] - float(distance[1]) / 100. - utensil_size[
            object_A][1] * 0.5 - utensil_size[object_B][1] * 0.5
    elif 'below' == relationship:
        pose_A['x'] = pose_B['x'] - float(distance[0]) / 100. - utensil_size[
            object_A][0] * 0.5 - utensil_size[object_B][0] * 0.5
    elif 'on the left of' == relationship:
        pose_A['y'] = pose_B['y'] + float(distance[1]) / 100. + utensil_size[
            object_A][1] * 0.5 + utensil_size[object_B][1] * 0.5
    elif 'above and to the right of' == relationship:
        pose_A['x'] = pose_B['x'] + float(distance[0]) / 100. + utensil_size[
            object_A][0] * 0.5 + utensil_size[object_B][0] * 0.5
        pose_A['y'] = pose_B['y'] - float(distance[1]) / 100. - utensil_size[
            object_A][1] * 0.5 - utensil_size[object_B][1] * 0.5
    elif 'below and to the right of' == relationship:
        pose_A['x'] = pose_B['x'] - float(distance[0]) / 100. - utensil_size[
            object_A][0] * 0.5 - utensil_size[object_B][0] * 0.5
        pose_A['y'] = pose_B['y'] - float(distance[1]) / 100. - utensil_size[
            object_A][1] * 0.5 - utensil_size[object_B][1] * 0.5
    elif 'below and to the left of' == relationship:
        pose_A['x'] = pose_B['x'] - float(distance[0]) / 100. - utensil_size[
            object_A][0] * 0.5 - utensil_size[object_B][0] * 0.5
        pose_A['y'] = pose_B['y'] + float(distance[1]) / 100. + utensil_size[
            object_A][1] * 0.5 + utensil_size[object_B][1] * 0.5
    elif 'above and to the left of' == relationship:
        pose_A['x'] = pose_B['x'] + float(distance[0]) / 100. + utensil_size[
            object_A][0] * 0.5 + utensil_size[object_B][0] * 0.5
        pose_A['y'] = pose_B['y'] + float(distance[1]) / 100. + utensil_size[
            object_A][1] * 0.5 + utensil_size[object_B][1] * 0.5
    elif 'on' == relationship and 'right' not in relationship and 'left' not in relationship:
        pose_A['z'] = pose_B['z'] + 0.1
    elif 'under' in relationship:
        pose_A['z'] = pose_B['z'] - 0.1
    elif 'center' in relationship:
        pass
    # print('\noutput:\n pose_A:{}, pose_B:{}, relationship:{}, distance:{}'.format(pose_A, pose_B, relationship, distance))
    # print('-'*40)


def sortdict(utensil_name_sequenced):
    # input: {'dinner knife': 0.75, 'plate': 0.65, 'fork': 0.65, 'spoon': 0.65}
    # output: {'plate': 0.65, 'fork': 0.65, 'spoon': 0.65, 'dinner knife': 0.75}
    return sorted(utensil_name_sequenced,
                  key=lambda x: utensil_name_sequenced[x])


def del_list(full_list, sub_list):
    new_full_list = [
        itemA for itemA in full_list
        if not any(itemA == itemB for itemB in sub_list)
    ]
    return new_full_list


def is_collision(x1, y1, width1, length1, x2, y2, width2, length2):
    # print('y1 - y2: {}, (length1 + length2) / 2.0: {}'.format(abs(y1 - y2), (length1 + length2) / 2.0))
    # print('x1 - x2: {}, (width1 + width2) / 2.0: {}'.format(abs(x1 - x2), (width1 + width2) / 2.0))
    if abs(x1 - x2) > (width1 + width2) / 2.0 or abs(
            y1 - y2) > (length1 + length2) / 2.0:
        return False
    else:
        return True


def get_combination(input_list):
    result = []
    for i in range(len(input_list)):
        for j in range(i + 1, len(input_list)):
            result.append((input_list[i], input_list[j]))
    return result


def get_maxkey(data):
    result = {}
    for key, value in data.items():
        result[key] = value[0] * value[1]
    max_key = max(result, key=result.get)
    # print("The item with the largest value is:", max_key)
    return max_key


def process_response(response_commonsense_abstract, zero_shot):
    # cornor case 1: xxx, xxx, xxx
    if zero_shot:
        response_commonsense_abstract = '\n'.join([
            line for line in response_commonsense_abstract.split(', ')
            if line.strip()
        ])  # split a long sentence into multiple lines

    # cornor case 2: plate, center --> plate, center of table
    if zero_shot:
        if 'table' not in response_commonsense_abstract and 'center' in response_commonsense_abstract:
            response_commonsense_abstract = response_commonsense_abstract.replace(
                "center", "center of the table")

    # cornor case 3: both should be placed in the center of the table. --> delete this sentece
    if zero_shot:
        lines = response_commonsense_abstract.split('\n')
        filtered_lines = [
            line for line in lines
            if not ('table' in line and not any(item in line
                                                for item in utensil_name))
        ]
        response_commonsense_abstract = '\n'.join(filtered_lines)

    # cornor case 4: if both table and center are not in sentence, assign an object in the center of object
    if 'table' not in response_commonsense_abstract and 'center' not in response_commonsense_abstract:
        response_commonsense_abstract = response_commonsense_abstract + '\n' + get_maxkey(
            utensil_size) + ' center of the table'

    # cornor case 5: remove '1.', '2.', ...
    response_commonsense_abstract = response_commonsense_abstract.split('\n')
    for index in range(len(response_commonsense_abstract)):
        response_commonsense_abstract[index] = re.sub(
            r'\d+\. ', '', response_commonsense_abstract[index])

    return response_commonsense_abstract


def create_prompt_augmented_commonsense_abstract(isolated_utensil,
                                                 remaining_utensil):
    prompt_hint = 'The task is to set up a dining table. The relationship between two objects can only contain the following planar relationships: above, on the right of, below, on the left of, above and to the right of, below and to the right of, below and to the left of, above and to the left of, on top of, under, and center.\n\n '
    prompt_question = 'Where should a {} be placed in relation to a {}?'.format(
        remaining_utensil, isolated_utensil)
    return prompt_hint + prompt_question


'''
below functions used for pybullet-based demo
'''


class Client():

    def __init__(self):
        pybullet.connect(pybullet.DIRECT)  # pybullet.GUI for local GUI.
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(0, 0, -9.8)

        # reset robot
        self.plane_id = pybullet.loadURDF("plane.urdf")
        self.robot_id = pybullet.loadURDF("kuka_iiwa/model_vr_limits.urdf",
                                          basePosition=[1.55, 0.0, 0.6],
                                          baseOrientation=[0.0, 0.0, 0.0, 1.0])
        jointPositions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for jointIndex in range(pybullet.getNumJoints(self.robot_id)):
            pybullet.resetJointState(self.robot_id, jointIndex,
                                     jointPositions[jointIndex])
            pybullet.setJointMotorControl2(self.robot_id, jointIndex,
                                           pybullet.POSITION_CONTROL,
                                           jointPositions[jointIndex], 0)

        # camera width and height
        self.cam_width = 480
        self.cam_height = 480

        # create a list to record utensil id
        self.utensil_id = {}
        self.gripper_id = None

    def render_image(self):
        # camera parameters
        cam_target_pos = [1.0, 0.0, 0.5]
        cam_distance = 1.5
        cam_yaw, cam_pitch, cam_roll = -90, -90, 0
        cam_up, cam_up_axis_idx, cam_near_plane, cam_far_plane, cam_fov = [
            0, 0, 1
        ], 2, 0.01, 100, 60
        cam_view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
            cam_target_pos, cam_distance, cam_yaw, cam_pitch, cam_roll,
            cam_up_axis_idx)
        cam_projection_matrix = pybullet.computeProjectionMatrixFOV(
            cam_fov, self.cam_width * 1. / self.cam_height, cam_near_plane,
            cam_far_plane)
        znear, zfar = 0.01, 10.

        # get raw data
        _, _, color, depth, segment = pybullet.getCameraImage(
            width=self.cam_width,
            height=self.cam_height,
            viewMatrix=cam_view_matrix,
            projectionMatrix=cam_projection_matrix,
            shadow=1,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)

        # get color image.
        color_image_size = (self.cam_width, self.cam_height, 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel

        # get depth image.
        depth_image_size = (self.cam_width, self.cam_height)
        zbuffer = np.float32(depth).reshape(depth_image_size)
        depth = (zfar + znear - (2 * zbuffer - 1) * (zfar - znear))
        depth = (2 * znear * zfar) / depth

        # get segment image.
        segment = np.reshape(segment,
                             [self.cam_width, self.cam_height]) * 1. / 255.
        return color, depth, segment

    def reset_video(self):
        video = imageio_ffmpeg.write_frames('video.mp4',
                                            (self.cam_width, self.cam_height),
                                            fps=60)
        video.send(None)  # seed the video writer with a blank frame
        return video

    def render_video(self, video, image):
        video.send(np.ascontiguousarray(image))

    def play_video(self):
        mp4 = open('video.mp4', 'rb').read()
        data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
        return HTML(
            '<video width=480 controls><source src="%s" type="video/mp4"></video>'
            % data_url)

    def add_table(self):
        flags = pybullet.URDF_USE_INERTIA_FROM_FILE
        path = '/content/urdf_models/'

        if not os.path.exists(path):
            print('!Error: cannot find /content/urdf_models/!')
            sys.exit()

        # add table
        table_id = pybullet.loadURDF(
            "/content/urdf_models/furniture_table_rectangle/table.urdf",
            basePosition=[1.0, 0.0, 0.0],
            baseOrientation=[0, 0, 0.7071, 0.7071])
        self.utensil_id['table'] = table_id

    def add_objects(self, utensil_name, utensil_init_pose):
        flags = pybullet.URDF_USE_INERTIA_FROM_FILE
        path = '/content/urdf_models/'

        if not os.path.exists(path):
            print('!Error: cannot find /content/urdf_models/!')
            sys.exit()

        # add objects according to utensil_name
        color = 'blue'
        color2 = 'red'
        if 'bread plate' in utensil_name:
            self.utensil_id['bread plate'] = pybullet.loadURDF(
                path + 'utensil_plate_circle_' + color + '_small' +
                '/model.urdf',
                basePosition=utensil_init_pose['bread plate'][0],
                baseOrientation=utensil_init_pose['bread plate'][1],
                flags=flags)
        if 'butter knife' in utensil_name:
            self.utensil_id['butter knife'] = pybullet.loadURDF(
                path + 'utensil_knife_' + color + '_small' + '/model.urdf',
                basePosition=utensil_init_pose['butter knife'][0],
                baseOrientation=utensil_init_pose['butter knife'][1],
                flags=flags)
        if 'dinner fork' in utensil_name:
            self.utensil_id['dinner fork'] = pybullet.loadURDF(
                path + 'utensil_fork_' + color + '/model.urdf',
                basePosition=utensil_init_pose['dinner fork'][0],
                baseOrientation=utensil_init_pose['dinner fork'][1],
                flags=flags)
        if 'soup spoon' in utensil_name:
            self.utensil_id['soup spoon'] = pybullet.loadURDF(
                path + 'utensil_fork_' + color + '/model.urdf',
                basePosition=utensil_init_pose['soup spoon'][0],
                baseOrientation=utensil_init_pose['soup spoon'][1],
                flags=flags)
        if 'water cup' in utensil_name:
            self.utensil_id['water cup'] = pybullet.loadURDF(
                path + 'utensil_cup_' + color + '/model.urdf',
                basePosition=utensil_init_pose['water cup'][0],
                baseOrientation=utensil_init_pose['water cup'][1],
                flags=flags)
        if 'wine glass' in utensil_name:
            self.utensil_id['wine glass'] = pybullet.loadURDF(
                path + 'utensil_glass_' + color + '/model.urdf',
                basePosition=utensil_init_pose['wine glass'][0],
                baseOrientation=utensil_init_pose['wine glass'][1],
                flags=flags)
        if 'napkin' in utensil_name:
            self.utensil_id['napkin'] = pybullet.loadURDF(
                path + 'utensil_napkin_' + color + '/model.urdf',
                basePosition=utensil_init_pose['napkin'][0],
                baseOrientation=utensil_init_pose['napkin'][1],
                flags=flags)
        if 'dinner knife' in utensil_name:
            self.utensil_id['dinner knife'] = pybullet.loadURDF(
                path + 'utensil_knife_' + color + '/model.urdf',
                basePosition=utensil_init_pose['dinner knife'][0],
                baseOrientation=utensil_init_pose['dinner knife'][1],
                flags=flags)
        if 'dinner plate' in utensil_name:
            self.utensil_id['dinner plate'] = pybullet.loadURDF(
                path + 'utensil_plate_circle_' + color + '/model.urdf',
                basePosition=utensil_init_pose['dinner plate'][0],
                baseOrientation=utensil_init_pose['dinner plate'][1],
                flags=flags)
        if 'salad fork' in utensil_name:
            self.utensil_id['salad fork'] = pybullet.loadURDF(
                path + 'utensil_fork_' + color + '/model.urdf',
                basePosition=utensil_init_pose['salad fork'][0],
                baseOrientation=utensil_init_pose['salad fork'][1],
                flags=flags)
        if 'dessert fork' in utensil_name:
            self.utensil_id['dessert fork'] = pybullet.loadURDF(
                path + 'utensil_fork_' + color + '_small' + '/model.urdf',
                basePosition=utensil_init_pose['dessert fork'][0],
                baseOrientation=utensil_init_pose['dessert fork'][1],
                flags=flags)
        if 'dessert spoon' in utensil_name:
            self.utensil_id['dessert spoon'] = pybullet.loadURDF(
                path + 'utensil_spoon_' + color + '_small' + '/model.urdf',
                basePosition=utensil_init_pose['dessert spoon'][0],
                baseOrientation=utensil_init_pose['dessert spoon'][1],
                flags=flags)
        if 'tea spoon' in utensil_name:
            self.utensil_id['tea spoon'] = pybullet.loadURDF(
                path + 'utensil_spoon_' + color + '_small' + '/model.urdf',
                basePosition=utensil_init_pose['tea spoon'][0],
                baseOrientation=utensil_init_pose['tea spoon'][1],
                flags=flags)
        if 'seafood fork' in utensil_name:
            self.utensil_id['seafood fork'] = pybullet.loadURDF(
                path + 'utensil_fork_' + color + '/model.urdf',
                basePosition=utensil_init_pose['seafood fork'][0],
                baseOrientation=utensil_init_pose['seafood fork'][1],
                flags=flags)
        if 'fish knife' in utensil_name:
            self.utensil_id['fish knife'] = pybullet.loadURDF(
                path + 'utensil_knife_' + color + '/model.urdf',
                basePosition=utensil_init_pose['fish knife'][0],
                baseOrientation=utensil_init_pose['fish knife'][1],
                flags=flags)
        if 'place mat' in utensil_name:
            self.utensil_id['place mat'] = pybullet.loadURDF(
                path + 'utensil_mat_' + color2 + '_small' + '/model.urdf',
                basePosition=utensil_init_pose['place mat'][0],
                baseOrientation=utensil_init_pose['place mat'][1],
                flags=flags)
        if 'salt shaker' in utensil_name:
            self.utensil_id['salt shaker'] = pybullet.loadURDF(
                path + 'utensil_shaker_' + color + '/model.urdf',
                basePosition=utensil_init_pose['salt shaker'][0],
                baseOrientation=utensil_init_pose['salt shaker'][1],
                flags=flags)
        if 'pepper shaker' in utensil_name:
            self.utensil_id['pepper shaker'] = pybullet.loadURDF(
                path + 'utensil_shaker_' + color + '/model.urdf',
                basePosition=utensil_init_pose['pepper shaker'][0],
                baseOrientation=utensil_init_pose['pepper shaker'][1],
                flags=flags)
        return self.utensil_id

    def get_bounding_box(self, obj_id):
        (min_x, min_y, min_z), (max_x, max_y, max_z) = pybullet.getAABB(obj_id)
        return [min_x, min_y, min_z], [max_x, max_y, max_z]

    def home_joints(self):
        jointPositions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        for jointIndex in range(pybullet.getNumJoints(self.robot_id)):
            pybullet.resetJointState(self.robot_id, jointIndex,
                                     jointPositions[jointIndex])
            pybullet.setJointMotorControl2(self.robot_id, jointIndex,
                                           pybullet.POSITION_CONTROL,
                                           jointPositions[jointIndex], 0)

    def pick_place(self, object_id, object_position_init, object_position_end,
                   video):
        num_joints = pybullet.getNumJoints(self.robot_id)
        end_effector_index = 6
        # target_position = [0.9, -0.6, 0.65]
        target_position = [
            object_position_init[0], object_position_init[1],
            object_position_init[2] + 0.1
        ]

        for step in tqdm(range(1000)):
            if step % 4 == 0:  # PyBullet default simulation time step is 240fps, but we want to record video at 60fps.
                rgb, depth, mask = self.render_image()
                self.render_video(video, np.ascontiguousarray(rgb))

            target_orientation = pybullet.getQuaternionFromEuler(
                [0, 1.01 * math.pi, 0])
            gripper_status = {'ungrasp': 0, 'grasp': 1}
            gripper_value = gripper_status['ungrasp']
            if step >= 150 and step < 250:
                target_position = [
                    object_position_init[0], object_position_init[1],
                    object_position_init[2] + 0.1
                ]  # grab object
                gripper_value = gripper_status['grasp']
            elif step >= 250 and step < 400:
                # target_position = [0.85, -0.2, 0.7 + 0.2*(step-250)/150.] # move up after picking object
                target_position = [
                    object_position_init[0], object_position_init[1],
                    object_position_init[2] + 0.4
                ]
                gripper_value = gripper_status['grasp']
            elif step >= 400 and step < 600:
                # target_position = [0.85, -0.2 + 0.4*(step-400)/200., 0.9] # move to target position
                target_position = [
                    object_position_init[0] +
                    (object_position_end[0] - object_position_init[0]) *
                    (step - 400) / 200, object_position_init[1] +
                    (object_position_end[1] - object_position_init[1]) *
                    (step - 400) / 200, object_position_init[2] + 0.6
                ]
                gripper_value = gripper_status['grasp']
            elif step >= 600 and step < 700:
                target_position = [
                    object_position_end[0], object_position_end[1],
                    object_position_end[2] + 0.4
                ]  # stop at target position
                gripper_value = gripper_status['grasp']
            elif step >= 700:
                target_position = [
                    object_position_end[0], object_position_end[1],
                    object_position_end[2] + 0.4
                ]  # drop object
                gripper_value = gripper_status['ungrasp']

            joint_poses = pybullet.calculateInverseKinematics(
                self.robot_id, end_effector_index, target_position,
                target_orientation)
            for joint_index in range(num_joints):
                pybullet.setJointMotorControl2(
                    bodyIndex=self.robot_id,
                    jointIndex=joint_index,
                    controlMode=pybullet.POSITION_CONTROL,
                    targetPosition=joint_poses[joint_index])

            if gripper_value == 0 and self.gripper_id != None:
                pybullet.removeConstraint(self.gripper_id)
                self.gripper_id = None
            if gripper_value == 1 and self.gripper_id == None:
                cube_orn = pybullet.getQuaternionFromEuler([0, math.pi, 0])
                self.gripper_id = pybullet.createConstraint(
                    self.robot_id,
                    end_effector_index,
                    object_id,
                    -1,
                    pybullet.JOINT_FIXED, [0, 0, 0], [0, 0, 0.05], [0, 0, 0],
                    childFrameOrientation=cube_orn)

            pybullet.stepSimulation()

    def disconnect(self):
        pybullet.disconnect()


#@title Get abstract positioning {display-mode: "form"}

# get raw response from LLM
zero_shot = False  #@param {type:"boolean"}
prompt = create_prompt_commonsense_abstract(utensil_name, zero_shot)
response_commonsense_abstract, probs = llm(prompt)
print('-' * 20 + '\nFull prompt (including {} tokens, cost {}$):\n{}\n'.format(
    count_tokens(prompt), fee_tokens(prompt), textwrap.fill(prompt, width=150))
      + '-' * 20)
print('\nLLM\'s response (raw):\n{}'.format(response_commonsense_abstract))

# process raw response
response_commonsense_abstract = process_response(response_commonsense_abstract,
                                                 zero_shot)
print(
    '\nLLM\'s response (processed):\n{}'.format(response_commonsense_abstract))

# get high-level positioning of utensils
utensils_positioning_abstract = []
for commonsense in response_commonsense_abstract:  # extract keyword
    object_A, object_B, relationship_A_B = extract_object_relationship(
        commonsense, utensil_name, relationship_name)
    utensils_positioning_abstract.append(
        [object_A, object_B, relationship_A_B])

print('\npositioning of utensils (abstract):')
for item in utensils_positioning_abstract:
    print('{}'.format(item))

#@title Get concrete positioning{display-mode: "form"}
utensils_positioning_concrete = []
for item in utensils_positioning_abstract:
    # print('item: {}'.format(item))
    if 'center' == item[2] or 'on top of' == item[2] or 'under' == item[2]:
        utensils_positioning_concrete.append([str(0), str(0)])
    else:
        prompt = create_prompt_commonsense_concrete(item)
        response_commonsense_concrete, probs = llm(prompt)
        print('-' * 20 +
              '\nFull prompt (including {} tokens, cost {}$):\n{}\n'.format(
                  count_tokens(prompt), fee_tokens(prompt),
                  textwrap.fill(prompt, width=150)) + '-' * 20)
        print('LLM\'s response:\n{}'.format(response_commonsense_concrete))
        if 'above' == item[2] or 'below' == item[2]:
            distance1, distance2 = extract_distance(
                response_commonsense_concrete, 'type1')
        elif 'on the right of' == item[2] or 'on the left of' == item[2]:
            distance1, distance2 = extract_distance(
                response_commonsense_concrete, 'type2')
        elif 'above and to the right of' or 'below and to the right of' or 'below and to the left of' or 'above and to the left of':
            distance1, distance2 = extract_distance(
                response_commonsense_concrete, 'type3')
        else:
            print('!Error: Unexpected relationship type: {}'.format(item[2]))
        utensils_positioning_concrete.append([str(distance1), str(distance2)])
print('\ndistance (in centimeter) between two utensils (concrete):\n{}'.format(
    utensils_positioning_concrete))

#@title Get abstract and concrete positioning{display-mode: "form"}
utensils_positioning = []
for index in range(len(utensils_positioning_abstract)):
    item = utensils_positioning_abstract[index] + [
        utensils_positioning_concrete[index]
    ]
    utensils_positioning.append(item)

print('positioning of utensils:')
for item in utensils_positioning:
    print('{}'.format(item))

#@title Ground positioning{display-mode: "form"}
'''
we ground positioning of utensils according to
1, distance in utensils_positioning
2, size of utensil
'''
# set table as world anchor (0.0, 0.0, 0.0)
utensil_goal_pose = {
    'table': {
        'x': 0.0,
        'y': 0.0,
        'z': 0.65
    }
}  # intialize utensil's goal pose
for item in utensil_name:
    utensil_goal_pose[item] = {'x': 0.0, 'y': 0.0, 'z': 0.65}

utensils_positioning_cp = copy.deepcopy(
    utensils_positioning)  # copy utensils_positioning

anchor = 'table'
item_processed = []  # record items that have been processed
anchor_list = []  # save anchors
for item in utensils_positioning_cp:
    if anchor in item:
        if item.index(anchor) == 1:
            utensil = item[0]
            anchor_list.append(utensil)  # save next anchors
            item_processed.append(item)
            relationship_objectAB(utensil, anchor, utensil_goal_pose[utensil],
                                  utensil_goal_pose[anchor], item[2],
                                  item[3])  # compute utensil's goal pose
        else:
            print('!Error: Table is not world anchor by default')
            sys.exit()
if len(anchor_list) != 1:
    print(
        '!Error: Unexpected number of items that have relationship with table')
    sys.exit()
anchor = anchor_list[0]  # set next anchor
utensils_positioning_cp = del_list(
    utensils_positioning_cp,
    item_processed)  # remove items that have been processed

if utensils_positioning_cp:
    counter = 0  # count how many items have relationship with anchor
    item_processed = []  # record items that have been processed
    for item in utensils_positioning_cp:
        if anchor in item and item.index(anchor) == 1:
            counter += 1
            item_processed.append(item)
            utensil = item[0]
            anchor_list.append(utensil)  # save next anchors
            relationship_objectAB(utensil, anchor, utensil_goal_pose[utensil],
                                  utensil_goal_pose[anchor], item[2], item[3])
    if counter == 0:
        print(
            '!Error: Cannot find items that have relationship with {}'.format(
                anchor))
        sys.exit()
    utensils_positioning_cp = del_list(
        utensils_positioning_cp,
        item_processed)  # remove items that have been processed

while utensils_positioning_cp:
    anchor_list_cp = copy.deepcopy(anchor_list)  # copy anchor_list
    for anchor in anchor_list_cp:
        item_processed = []  # record items that have been processed
        for item in utensils_positioning_cp:
            if anchor in item and item.index(anchor) == 1:
                item_processed.append(item)
                utensil = item[0]
                anchor_list.append(utensil)
                relationship_objectAB(utensil, anchor,
                                      utensil_goal_pose[utensil],
                                      utensil_goal_pose[anchor], item[2],
                                      item[3])
        utensils_positioning_cp = del_list(
            utensils_positioning_cp,
            item_processed)  # remove items that have been processed

# sequence utensil_name according to "z"
utensil_name = sorted(utensil_name, key=lambda x: utensil_goal_pose[x]['z'])

# print final results
print('positioning result (unit is meter):')
print('world anchor: table, its pose: {}'.format(utensil_goal_pose['table']))
for item in utensil_name:
    print('utensil: {}, its pose: {}'.format(item, utensil_goal_pose[item]))

# print final results
"""
print('configuration result:')
print('world anchor: table, its pose: {}'.format(pose_objects['table']))
for item in utensil_name:
    print('utensil: {}, its pose: {}'.format(item, pose_objects[item]))
print(pose_objects)
"""
# import json
# with open("gpt3_log/pose_objects.json", 'w+') as outfile:
#     json.dump(pose_objects, outfile)

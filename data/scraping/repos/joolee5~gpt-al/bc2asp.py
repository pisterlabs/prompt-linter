import openai

from keys import API_KEY,ORG_KEY

openai.api_key= API_KEY
openai.organization = ORG_KEY


def get_response_4(prompt, prompt_cache, redo, temp=0.,max_tokens=30):

    if prompt in prompt_cache and not redo:
        return prompt_cache[prompt], prompt_cache

    else:
        passed = False; tries = 0 
        messages = [{'role': 'user', 'content': prompt}]
        while not passed:
            try:
                response = openai.ChatCompletion.create(
                    messages=messages,
                    model="gpt-4",
                    temperature=0,
                    max_tokens=max_tokens)
                passed=True;tries+=1
                prompt_cache[prompt] = response
            except:
                tries+=1
                if tries >=4:
                    breakpoint()
                print(f'doing attempt number {tries +1}')

        return response, prompt_cache

def get_response_check_4(prompt, prompt_cache, temp=0,max_tokens=2000, redo=False):

    response = get_response_4(prompt, prompt_cache, redo, temp=temp,max_tokens=max_tokens)
    #time.sleep(2)
    return response


bcdomain2asp = '''The programming language BC is an action language which has two kinds of constants, a fluent constant and an action constant. Answer Set Programming (ASP) is a logic programming paradigm. Given a domain specification in BC, generate ASP code which represents the supersorts (if applicable), sorts, choice rule for every fluent constant, choice rule for every action constant, inertial rules, and the uniqueness/existence constraint for each atom.
s_boolean(true;false) represents the boolean values true and false and is always used.
Write the constraints according to the given domains into BC:

Program 1:

Domain:
:- sorts
    loc >> block;
    block_w_none >> block;
    machine;
    gripper.

:- objects
    table :: loc;
    b1, b2, b3, b4 :: block;
    none :: block_w_none;
    m1, m2 :: machine;
    g1, g2, g3 :: gripper.

:- constants
    loc(block) :: inertialFluent(loc);
    gripped :: inertialFluent(block_w_none);
    in_tower(block) :: sdFluent;
    move(block, loc) :: action.

:-variables
    B, B1, B2 :: block;
    B_none :: block_w_none;
    L, L1 :: loc.


Generated constraints:

% supersorts
s_loc(B) :- s_block(B).
s_block_w_none :- s_block(B).

% sorts
s_boolean(true;false).
s_block_w_none(none).

s_loc(table).
s_block(b1;b2;b3;b4).
s_machine(m1;m2).
s_gripper(g1;g2).

% choice rule {0:A} for every atom A containing a regular fluent constant
{holds(loc((B),V),0)} :- s_block(B), s_loc(V).
{holds(gripped(V),0)} :- s_block_w_none(V).
{holds(in_tower((B),V),0)} :- s_block(B), s_boolean(V).


% choice rule {i:a} for every action constant a and every i < l
{occurs(move(B,L),T_step)} :- s_block(B), s_loc(L), timestep(T_step).

% Inertia

{holds(loc((B),V),T_step+1)} :-holds(loc((B),V),T_step), timestep(T_step).
{holds(gripped(V),T_step+1)} :-holds(gripped(V),T_step), timestep(T_step).
{holds(in_tower((B),V),T_step+1)} :-holds(in_tower((B),V),T_step), timestep(T_step).


% existence and uniqueness of value constraint

:- not {holds(loc((B),V),T_step): s_loc(V)}=1, s_block(B), timestep(T_step).
:- not {holds(gripped(V),T_step): s_block_w_none(V)}=1, timestep(T_step).
:- not {holds(in_tower((B),V),T_step): s_boolean(V)}=1, s_block(B), timestep(T_step).

Program 2:

Domain:
<DOMAIN>

Generated constraints:'''


nl2bc = '''A program in BC consists of causal rules of the following 2 forms where their readings are shown after "%".
There are two kinds of constants, a fluent constant and an action constant. An  atom is of the form f=v, where f is a fluent constant, and v is an element of its domain. A fluent which has no argument, such as "hasGas" is assumed to be boolean. 

Here are some common ways to write rules.

Example:
% The location of a person is the same as the car if the person is in the car.
loc(person)=L if inCar & loc(car)=L.
Note that the head of the rule (loc(person)=L in the example), must be only a single atom. So the following rule is not valid:
loc(person)=loc(car) if inCar.

Example:
impossible hasbrother(C) & ~hasSibling(C).
(note that impossible can only have conjunction (&), and cannot contain "if".

Example:
default onTable(B) if block(B). 

Example:
open(D) causes opened(D) if available(D).

Examples:
% It is not permissible to drive a car if it has no gas.
nonexecutable driveCar if ~hasGas.
% It is not permissible to lift an object if it is heavy.
nonexecutable lift(object) if heavy(object).

Examples
default door(open) if door(unlocked) after push(door).

In general, if something cannot be true, then we use "impossible" when writing the rules, but if instead we want to assert something is not true, then we use the negation (~). For example, if it is impossible for an object to be on the table and under it, we might write "impossible onTable(object) & underTable(object).", but express that if it rains outside the ground is not wet as "~groundWet if noRain".
Additionally, "impossible" is reserved only for fluents, while "nonexecutable" is only reserved for actions.

Here is an example representation.
Problem 1:
Given the following domain:
blocks: b1, b2, b3, b4.
locations: blocks + {table}
variables:
block: B, B1, B2.
loc: L, L1.
constants:
- loc(block) :: inertialFluent.
- in_tower(block) :: sdFluent.
- move(block, loc) :: action.

Represent the following constraints:
% 1. Two blocks cannot be at the same location.
% 2.1 A block is in a tower if it's location is on the table.
% 2.2 A block is in a tower if it's location is on something that is in a tower.
% 3. Blocks don't float in the air.
% 4. Moving a block causes it's location to move to loc.
% 5. The move action is not executable if something is on the location to be moved to.

Constraints:
% 1. Two blocks cannot be at the same location.
impossible loc(B1) = B & loc(B2) = B where B1\=B2.

% 2.1 A block is in a tower if it's location is on the table.
in_tower(B) if loc(B) = table.

% 2.2 A block is in a tower if it's location is on something that is in a tower.
in_tower(B) if loc(B) = B1 & in_tower(B1).

% 3. Blocks don't float in the air.
impossible ~in_tower(B).

% 4. Moving a block causes it's location to move to loc.
move(B,L) causes loc(B)=L.

% 5. The move action is permissible if something is on the location to be moved to.
nonexecutable move(B,L) if loc(B1) = B.

Problem 2:
Given the following domain:
<DOMAIN>

Write the following constraints:
<CONSTRAINTS>

Constraints:'''


sh2bc = '''The programming language BC is an action language which has two kinds of constants, a fluent constant and an action constant. An atom is of the form f=v, where f is a fluent constant, and v is an element of its domain. There are some shorthand rules which are described below. The shorthand versions expand into another form via a translation.


Rules in this form are not changed.
Example: 
Original rule: loc(person)=L if inCar & loc(car)=L.
Translation: loc(person)=L if inCar & loc(car)=L.


Example:
Original rule: status(light) = on after switch(on).
Translation: status(light) = on after switch(on).

ACTIONS

Example:
Original rule: drive(C,L) causes loc(C) = L.
Translation: loc(C)=L after drive(C,L).

Example:
Original rule: loc(car) = home & drive(home) & loc(car) = bank.
Translation: loc(car)  = home after drive(home) & loc(car) = bank.

DEFAULTS

Example:
Original Rule: default onTable(B) if color(B)=red.
Translation: onTable(B) if color(B) if cons onTable(B).

Example:
Original rule: default ~onTable(B) if pickedUp(B).
Translation: ~onTable(B) after pickedUp(B) if cons ~onTable(B).

IMPOSSIBLE

Example:
Original rule: impossible hasbrother(C) & ~hasSibling(C).
Translation: false if hasbrother(C) & ~hasSibling(C).

NONEXECUTABLE

Example:
Original rule: nonexecutable lift(object) if weight(object)=heavy.
Translation: false after lift(object) & weight(object)=heavy.

Translate the following programs:

Program 1

Domain:
:- sorts
    loc >> block;
    block_w_none >> block;
    machine;
    gripper.

:- objects
    table :: loc;
    b1, b2, b3, b4 :: block;
    m1, m2 :: machine;
    g1, g2, g3 :: gripper.

:- constants
    loc(block) :: inertialFluent(loc);
    gripped :: inertialFluent(block+none);
    in_tower(block) :: sdFluent;
    move(block, loc) :: action.

:-variables
    B, B1, B2 :: block;
    L, L1 :: loc.

Original rules:
% 1. Two blocks cannot be at the same location.
impossible loc(B1) = B & loc(B2) = B & B1\=B2.

% 2.1 A block is in a tower if it's location is on the table.
in_tower(B) if loc(B) = table.

% 2.2 A block is in a tower if it's location is on something that is in a tower.
in_tower(B) if loc(B) = B1 & in_tower(B1).

% 3. Blocks don't float in the air.
impossible ~in_tower(B).

% 4. Moving a block causes it's location to move to loc.
move(B,L) causes loc(B)=L.

% 5. The move action is impermissible if something is on the location to be moved to.
nonexecutable move(B,L) if loc(B1) = B.

% 6. By default, a block is not in the tower.
default ~in_tower(B).

% 7. The move action causes nothing to be gripped.
move(B,L) causes gripped(none).

Translation:


% 1. Two blocks cannot be at the same location.
false if loc(B1) = B & loc(B2) = B and B1\=B2.

% 2.1 A block is in a tower if it's location is on the table.
in_tower(B) if loc(B) = table.

% 2.2 A block is in a tower if it's location is on something that is in a tower.
in_tower(B) if loc(B) = B1 & in_tower(B1).

% 3. Blocks don't float in the air.
false if ~in_tower(B).

% 4. Moving a block causes it's location to move to loc.
loc(B)=L after move(B,L).

% 5. The move action is impermissible if something is on the location to be moved to.
false after move(B,L) & loc(B1) = B.

% 6. By default, a block is not in the tower.
~in_tower(B) if cons ~in_tower(B).

% 7. The move action causes nothing to be gripped.
gripped(none) if move(B,L).

Program 2

Domain:
<DOMAIN>

Original rules:
<RULES>

Translation:'''


bc2asp = '''The programming language BC is an action language which has two kinds of constants, a fluent constant and an action constant. An atom is of the form f=v, where f is a fluent constant, and v is an element of its domain. These rules eventuall get translated into another form.

Atoms are generally in the form "f=v", but can be written short form "f" if it represents a boolean variable, and hence "v" is true and false.

Here, we will translate rules and their atoms. If the atom is of the form "f=v", then it will be copied as "holds(f(v),i)" according to the translation, where v is an element in f's domain. If it is of the form of "f", then it will be copied as "holds(f(v),i)", where v is either true or false.

For example, the atom "loc(car)=L" may be translated to "holds(loc_car(L),i)", and the atom "door(open)" may be translated into "holds(door_open(true))" or "holds(door_open(false))" since it is boolean.

Static laws:

Example:
Original rule: loc(person)=L if inCar & loc(car)=L if cons on(car).
Translation: holds(loc((person),L),I) :- holds(incar(true),I), holds(loc((car),L),I), not not holds(on((car), true),I), timestep(I).

Example:
Original rule: false if hasbrother(C) & ~hasSibling(C).
Translation::- holds(hasbrother((C),true),I), holds(hasSibling((C),false),I), timestep(I).

Dynamic laws:

Example:
Original rule: status(light) = on after switch(on) if cons power(on) & connected.
Translation: holds(status((light),on),I+1) :- holds(switch((on),true),I), not not holds(power((on),true),I+1), not not holds(connected((true)),I+1), timestep(I).

Example:
Original rule: false after lift(O) & weight(O)=heavy.
Translation: :- occurs(lift(object),I), holds(weight((O),heavy),I), timestep(I).

Translate the following programs:

Program 1:

BC Domain:
:- sorts
    loc >> block;
    block_w_none >> block;
    machine;
    gripper.

:- objects
    table :: loc;
    b1, b2, b3, b4 :: block;
    m1, m2 :: machine;
    g1, g2, g3 :: gripper.

:- constants
    loc(block) :: inertialFluent(loc);
    gripped :: inertialFluent(block+none);
    in_tower(block) :: sdFluent;
    move(block, loc) :: action.

ASP Domain:

% supersorts
s_loc(B) :- s_block(B).

% sorts
s_boolean(true;false).  

s_loc(table).
s_block(b1;b2;b3;b4).  
s_machine(m1;m2).
s_gripper(g1;g2).

% choice rule {0:A} for every atom A containing a regular fluent constant
{holds(loc((B),V),0)} :- s_block(B), s_loc(V).
{holds(gripper(V),0)} :- s_block_w_none(V).
{holds(in_tower((B),V),0)} :- s_block(B), s_boolean(V).

% choice rule {i:a} for every action constant a and every i < l
{occurs(move(B,L),T_step)} :- s_block(B), s_loc(L), timestep(T_step).

% Inertia

{holds(loc((B),V),T_step+1)} :-holds(loc((B),V),T_step), timestep(T_step).
{holds(gripped(V),T_step+1)} :-holds(gripped(V),T_step), timestep(T_step).
{holds(in_tower((B),V),T_step+1)} :-holds(in_tower((B),V),T_step), timestep(T_step).

% existence and uniqueness of value constraint

:- not {holds(loc((B),V),T_step): s_loc(V)}=1, s_block(B), timestep(T_step).
:- not {holds(gripped(V),T_step): s_block_w_none(V)}=1, timestep(T_step).
:- not {holds(in_tower((B),V),T_step): s_boolean(V)}=1, s_block(B), timestep(T_step).


Rules:
% 1. Two blocks cannot be at the same location.
false if loc(B1) = B & loc(B2) = B and B1\=B2.

% 2.1 A block is in a tower if it's location is on the table.
in_tower(B) if loc(B) = table.

% 2.2 A block is in a tower if it's location is on something that is in a tower.
in_tower(B) if loc(B) = B1 & in_tower(B1).

% 3. Blocks don't float in the air.
false if ~in_tower(B).

% 4. Moving a block causes it's location to move to loc.
loc(B)=L after move(B,L).

% 5. The move action is impermissible if something is on the location to be moved to.
false after move(B,L) & loc(B1) = B.

% 6. By default, a block is not in the tower.
~in_tower(B) if cons ~in_tower(B).

% 7. The move action causes nothing to be gripped.
gripped=none if move(B,L).

Translation:
% 1. Two blocks cannot be at the same location.
 :- holds(loc((B1),B),T_step), holds(loc((B2),B),T_step), B1!=B2, s_block(B), s_block(B1), s_block(B2), timestep(T_step).

% 2.1 A block is in a tower if it's location is on the table.
holds(in_tower((B),true),T_step) :- holds(loc((B),table),T_step), s_block(B), timestep(T_step).

% 2.2 A block is in a tower if it's location is on something that is in a tower.
holds(in_tower((B),true),T_step) :- holds(loc((B),B1),T_step), holds(in_tower((B1),true),T_step), s_block(B), s_block(B1), timestep(T_step).

% 3. Blocks don't float in the air.
:- holds(in_tower((B),true),T_step), s_block(B), timestep(T_step).

% 4. Moving a block causes it's location to move to loc.
holds(loc((B),L),T_step+1) :- occurs(move(B,L),T_step), s_block(B), s_loc(L), timestep(T_step).

% 5. The move action is impermissible if something is on the location to be moved to.
:- occurs(move(B,L),T_step), holds(loc((B1),B),T_step), s_block(B), s_block(B1), s_loc(L), timestep(T_step).

% 6. By default, a block is not in the tower.
holds(in_tower((B),false),T_step) :- not not holds(in_tower((B),false),T_step), s_block(B), timestep(T_step).

% 7. The move action causes nothing to be gripped.
holds(gripped(none),T_step+1) :- occurs(move(B,L),T_step), s_block(B), s_loc(L).

Program 2:

BC Domain:
<DOMAIN>

ASP Domain:
    
<ASPDOMAIN>

Rules:
<RULES>

Translation:'''






import pickle
import os

if 'prompt_cache_gpt-4.pickle' in os.listdir():
    with open('prompt_cache_gpt-4.pickle', 'rb') as handle:
        prompt_cache = pickle.load(handle)
else:
    prompt_cache = dict()





from domains import problem_dict


response_dict = {}

problem_prompts = []
responses={}
#problem_dict = {key:item for key,item in problem_dict.items() if key in ['ferry']}
for prob_name,prob in problem_dict.items():
    domain,constraints=prob
    prompt_prompt=[]
    
    domain_prompt = bcdomain2asp.replace('<DOMAIN>',domain)
    prompt_prompt.append(domain_prompt)
    response, prompt_cache = get_response_check_4(domain_prompt, prompt_cache)
    response_text_asp_domain = response['choices'][0]['message']['content']
    responses[prob_name]=[response_text_asp_domain]
    
    problem_prompt = nl2bc.replace('<DOMAIN>',domain).replace('<CONSTRAINTS>',constraints)
    prompt_prompt.append(problem_prompt)
    response, prompt_cache = get_response_check_4(problem_prompt, prompt_cache)
    response_text = response['choices'][0]['message']['content']
    responses[prob_name].append(response_text)
    

    prompt = sh2bc.replace('<DOMAIN>',domain).replace('<RULES>',response_text)
    prompt_prompt.append(prompt)
    response, prompt_cache = get_response_check_4(prompt, prompt_cache)
    response_text = response['choices'][0]['message']['content']
    responses[prob_name].append(response_text)
    
    
    prompt = bc2asp.replace('<DOMAIN>',domain).replace('<ASPDOMAIN>',response_text_asp_domain).replace('<RULES>',response_text)
    prompt_prompt.append(prompt)
    response, prompt_cache = get_response_check_4(prompt, prompt_cache)
    response_text = response['choices'][0]['message']['content']
    
    responses[prob_name].append(response_text)
    problem_prompts.append(prompt_prompt)
    with open('prompt_cache_gpt-4.pickle', 'wb') as handle:
        pickle.dump(prompt_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

import csv

csv_form = []
csv_form.append(['Domain name', 'Domain prompt (bcdomain2asp)', 'NL -> shorthand BC (nl2bc)', 'shorthand BC -> BC (sh2bc)', 'BC -> ASP (bc2asp)', 'All responses/All responses without intermediate BC'])
for idx,key in enumerate(problem_dict):
    csv_form.append([key + ' prompts'] + problem_prompts[idx] + ['\n\n\n'.join(responses[key])])
    csv_form.append([key + ' responses'] + responses[key]+ ['\n\n\n'.join([resp for idx,resp in enumerate(responses[key]) if idx in [0,3]])])
#csv_form = [l for l in responses.values()]


with open("output.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerows(csv_form)

# =============================================================================
# from clyngor import ASP
# 
# def ASP_solve(ASP_program):
#     answers = ASP(ASP_program,nb_model=1,options = '--warn none')
#     
#     try:
#         all_lines=[]
#         for line in answers:
#             all_lines.append(line)
#     
#         return all_lines
#     except:
#         print('failed')
#         return []
# 
# 
# timestep_str = '''#const l = 2.
# 
# timestep(I) :- I=0..l-1.
# '''
# #breakpoint()
# for idx, key in enumerate(responses.keys()):
#     asp_rules = '\n\n\n'.join([resp for idx,resp in enumerate(responses[key]) if idx in [0,3]])# + '\n\n :- not a.'
#     
#     #breakpoint()
#     answer = ASP_solve(timestep_str + asp_rules)
#     if len(answer)!=1:
#         print(f'idx: {idx} has syntax error or is unsatisfiable')
#     print(idx)
# =============================================================================






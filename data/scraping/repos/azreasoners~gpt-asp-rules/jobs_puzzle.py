import openai
import json
import argparse
import os

kinds = ['constants', 'predicates', 'search_space', 'paraphrasing', 'constraints']
from api_keys import API_KEY, ORG_KEY

openai.api_key = API_KEY
openai.organization = ORG_KEY
def gen_response(prompt, engine, prompt_cache):
    # obtain the whole prompt
    # generate and cache the response in cache if it's not cached before
    if prompt not in prompt_cache:
        if engine == 'gpt-4':
            messages = [{'role': 'user', 'content': prompt}]
            try:
                prompt_cache[prompt] = openai.ChatCompletion.create(
                    messages=messages,
                    model="gpt-4",
                    temperature=0,
                    max_tokens=1500)
            except:
                    print("GPT failed.")
        else:
            try:
                prompt_cache[prompt] = openai.Completion.create(
                    prompt=prompt,
                    engine=engine,
                    temperature=0,
                    max_tokens=1500)
            except:
                print('GPT failed.')
        with open('caches/prompt_cache_jobspuzzle_'+engine + '.json', 'w') as f:
            json.dump(prompt_cache, f)
    if engine == 'gpt-4':
        return prompt_cache[prompt]['choices'][0]['message']['content'].strip()
    return prompt_cache[prompt]['choices'][0]['text'].strip()


prompt_C = '''Given a problem, extract all different constants and their categories in the form "category: constant_1; constant_2; ...; constant_n". Here, the format of each constant is turned into either an integer or a string surrounded by double quotes, e.g., "some name". 

Problem 1:
Consider N-Queens Puzzle on a chessboard of size 8x8. The goal is to assign 8 queens on the chessboard so that no two queens can share the same row, column, or diagonal.

Constants:
index_of_row: 0; 1; 2; 3; 4; 5; 6; 7.
index_of_column: 0; 1; 2; 3; 4; 5; 6; 7.

Problem 2:
"Against the Grain" offers hand-made wooden furniture at reasonable prices. Each item is made by an in-house employee. Using only the clues that follow, match each item to the employee who crafted it, and determine its price and the type of wood used to make it. Remember, as with all grid-based logic puzzles, no option in any category will ever be used more than once.
1. Bonita's piece costs $325.
2. The item made of poplar costs more than Yvette's piece.
3. Tabitha's item costs 50 dollars less than the piece made of sandalwood.
4. The $275 item is either the piece made of ash or Yvette's item.

Constants:
employee: "Bonita"; "Yvette"; "Tabitha".
price: 225; 275; 325.
wood_type: "ash"; "poplar"; "sandalwood".

Problem 3:
1. There are four people: Roberta, Thelma, Steve, and Pete.
2. Among them, they hold eight different jobs.
3. Each holds exactly two jobs.
4. The jobs are: chef, guard, nurse, telephone operator, police officer (gender not
implied), teacher, actor, and boxer.
5. The job of nurse is held by a male.
6. The husband of the chef is the telephone operator.
7. Roberta is not a boxer.
8. Pete has no education past the ninth grade.
9. Roberta, the chef, and the police officer went golfing together.
Question: Who holds which jobs?

Constants:'''

prompt_P = '''Given a problem and some categorized constants of the form "category: constant_1; constant_2; ...; constant_n", generate the minimum number of predicates to define the relations among the categories of constants. Each generated predicate is of the form "predicate(X1, X2, ..., Xn)" where X1, X2, ..., Xn are different variables and each variable X belongs to one of the categories. For each category, there must exist at least one variable of some predicate that belongs to this category.

Problem 1:
Consider N-Queens Puzzle on a chessboard of size 8x8. The goal is to assign 8 queens on the chessboard so that no two queens can share the same row, column, or diagonal.

Constants:
index_of_row: 0; 1; 2; 3; 4; 5; 6; 7.
index_of_column: 0; 1; 2; 3; 4; 5; 6; 7.

Predicates:
% The categories in Constants include index_of_row and index_of_column. We use different variables Ir and Ic to represent index_of_row and index_of_column.
% We assign a queen at row Ir and column Ic, where Ir belongs to index_of_row and Ic belongs to index_of_column.
assign(Ir, Ic)

Problem 2:
"Against the Grain" offers hand-made wooden furniture at reasonable prices. Each item is made by an in-house employee. Using only the clues that follow, match each item to the employee who crafted it, and determine its price and the type of wood used to make it. Remember, as with all grid-based logic puzzles, no option in any category will ever be used more than once.
1. Bonita's piece costs $325.
2. The item made of poplar costs more than Yvette's piece.
3. Tabitha's item costs 50 dollars less than the piece made of sandalwood.
4. The $275 item is either the piece made of ash or Yvette's item.

Constants:
employee: "Bonita"; "Yvette"; "Tabitha".
price: 225; 275; 325.
wood_type: "ash"; "poplar"; "sandalwood".

Predicates:
% The categories in Constants include employee, price, and wood_type. We use different variables E, P, and W to represent employee, price, and wood_type.
% We match an employee E with price P and wood type W, where E belongs to employee, P belongs to price, and W belongs to wood_type.
match(E, P, W)

Problem 3:
1. There are four people: Roberta, Thelma, Steve, and Pete.
2. Among them, they hold eight different jobs.
3. Each holds exactly two jobs.
4. The jobs are: chef, guard, nurse, telephone operator, police officer (gender not
implied), teacher, actor, and boxer.
5. The job of nurse is held by a male.
6. The husband of the chef is the telephone operator.
7. Roberta is not a boxer.
8. Pete has no education past the ninth grade.
9. Roberta, the chef, and the police officer went golfing together.
Question: Who holds which jobs?

Constants:
<CONSTANTS>

Predicates:''' 

prompt_R1 = '''Given a problem as the background information, turn the provided categorized constants and predicates into rules.

Tasks:
1. Write a rule of the form "category(constant_1; constant_2; ...; constant_n)" to represent each row of categorized constants in the form "category: constant_1; constant_2; ...; constant_n".
2. Write a rule of the form "{predicate(X_1, X_2, ..., X_n): category_2(X_2), ..., category_n(X_n)}=k :- category_1(X_1)" for each predicate of the form "predicate(X_1, X_2, ..., X_n)".

Problem 1:
"Against the Grain" offers hand-made wooden furniture at reasonable prices. Each item is made by an in-house employee. Using only the clues that follow, match each item to the employee who crafted it, and determine its price and the type of wood used to make it. Remember, as with all grid-based logic puzzles, no option in any category will ever be used more than once.
1. Bonita's piece costs $325.
2. The item made of poplar costs more than Yvette's piece.
3. Tabitha's item costs 50 dollars less than the piece made of sandalwood.
4. The $275 item is either the piece made of ash or Yvette's item.

Constants:
employee: "Bonita"; "Yvette"; "Tabitha".
price: 225; 275; 325.
wood_type: "ash"; "poplar"; "sandalwood".

Predicates:
% Define the constants in each category.
employee("Bonita"; "Yvette"; "Tabitha").
price(225; 275; 325).
wood_type("ash"; "poplar"; "sandalwood").
% For each employee E, it matches with exactly 1 price P and wood type W.
{match(E, P, W): price(P), wood_type(W)}=1 :- employee(E).

ASP Rules:
% Task 1. Define the constants in each category.
index_of_row(0; 1; 2; 3; 4; 5; 6; 7).
index_of_column(0; 1; 2; 3; 4; 5; 6; 7).
% Task 2. For each row Ir, there is exactly 1 queen assigned at some column Ic.
{assign(Ir, Ic): index_of_column(Ic)}=1 :- index_of_row(Ir).
% Done tasks 1 and 2.

Problem 2:
1. There are four people: Roberta, Thelma, Steve, and Pete.
2. Among them, they hold eight different jobs.
3. Each holds exactly two jobs.
4. The jobs are: chef, guard, nurse, telephone operator, police officer (gender not
implied), teacher, actor, and boxer.
5. The job of nurse is held by a male.
6. The husband of the chef is the telephone operator.
7. Roberta is not a boxer.
8. Pete has no education past the ninth grade.
9. Roberta, the chef, and the police officer went golfing together.
Question: Who holds which jobs?

Constants:
<CONSTANTS>

Predicates:
<PREDICATES>

ASP rules:'''

prompt_R2 = '''Consider the constraint in the following form
<C1>; <C2>; ...; <Cm> :- <L1>, <L2>, ..., <Ln>.
which says that if the conjunction "<L1> and <L2> and ... and <Ln>" is true, then the disjunction of comparisons "<C1> or <C2> or ... or <Cm>" must be true.

One can also add a restriction that "exactly k of <C1>, <C2>, ..., <Cm> is true" by using the following form
{<C1>; <C2>; ...; <Cm>}=k :- <L1>, <L2>, ..., <Ln>.

Given a problem, extract all constraints from the clues in the problem using only the provided constants and predicates.

Problem 1:
"Against the Grain" offers hand-made wooden furniture at reasonable prices. Each item is made by an in-house employee. Using only the clues that follow, match each item to the employee who crafted it, and determine its price and the type of wood used to make it.
1. No option in any category will ever be used more than once.
2. Bonita's piece costs $325.
3. The item made of poplar costs more than Yvette's piece.
4. Tabitha's item costs 50 dollars less than the piece made of sandalwood.
5. The $275 item is either the piece made of ash or Yvette's item.

Constants:
employee: "Bonita"; "Yvette"; "Tabitha".
price: 225; 275; 325.
wood_type: "ash"; "poplar"; "sandalwood".

Predicates:
% The categories include employee, price, and wood_type. We use different variables E, P, and W to represent employee, price, and wood_type.
% We match an employee E with price P and wood type W, where E belongs to employee, P belongs to price, and W belongs to wood_type.
match(E, P, W)

Constraints:
% 1. No option in any category will ever be used more than once.
{E1=E2; P1=P2; W1=W2}=0 :- match(E1,P1,W1), match(E2,P2,W2), (E1,P1,W1)!=(E2,P2,W2).

% 2. Bonita's piece costs $325.
P=325 :- match(E,P,W), E="Bonita".

% 3. The item made of poplar costs more than Yvette's piece.
P1>P2 :- match(E1,P1,W1), match(E2,P2,W2), W1="poplar", E2="Yvette".

% 4. Tabitha's item costs 50 dollars less than the piece made of sandalwood.
P1=P2-50 :- match(E1,P1,W1), match(E2,P2,W2), E1="Tabitha", W2="sandalwood".

% 5. The $275 item is either the piece made of ash or Yvette's item.
{W="ash"; E="Yvette"}=1 :- match(E,P,W), P=275.

Problem 2:
The Winter Olympics have just wrapped up in Norway. Using only the clues that follow, determine the number of gold, silver and bronze medals won by each country.
1. No option in any category will ever be used more than once.
2. The four teams are the team from Bolivia, the team that won 3 gold medals, the team that won 6 silver medals, and the team from Argentina.
3. The team from Oman and the team that won 10 silver medals are different.
4. The team from Oman finished with 2 gold medals or finished with 1 gold medal.
5. The squad that won 3 gold medals, the squad that won 6 silver medals and the squad from Bolivia were all different teams.
6. Neither the team from Argentina nor the squad that won 2 silver medals is the squad that won 4 gold medals.
7. The squad that won 2 gold medals is either the squad that won 6 silver medals or the team from Grenada.

Constants:
country: "Argentina"; "Bolivia"; "Grenada"; "Oman".
silver_medals: 2; 6; 10; 11.
gold_medals: 1; 2; 3; 4.

Predicates:
% The categories include country, silver_medals, and gold_medals. We use different variables C, S, and G to represent country, silver_medals, and gold_medals.
% We assign a country C with silver medals S and gold medals G, where C belongs to country, S belongs to silver_medals, and G belongs to gold_medals.
assign(C, S, G)

Constraints:
% 1. No option in any category will ever be used more than once.
{C1=C2; S1=S2; G1=G2}=0 :- assign(C1,S1,G1), assign(C2,S2,G2), (C1,S1,G1)!=(C2,S2,G2).

% 2. The four teams are the team from Bolivia, the team that won 3 gold medals, the team that won 6 silver medals, and the team from Argentina.
{C="Bolivia"; G=3; S=6; C="Argentina"}=1 :- assign(C,S,G).

% 3. The team from Oman and the team that won 10 silver medals are different.
C1!=C2 :- assign(C1,S1,G1), assign(C2,S2,G2), C1="Oman", S2=10.

% 4. The team from Oman finished with 2 gold medals or finished with 1 gold medal.
{G=2; G=1}=1 :- assign(C,S,G), C="Oman".

% 5. The squad that won 3 gold medals, the squad that won 6 silver medals and the squad from Bolivia were all different teams.
C1!=C2 :- assign(C1,S1,G1), assign(C2,S2,G2), assign(C3,S3,G3), G1=3, S2=6, C3="Bolivia".
C2!=C3 :- assign(C1,S1,G1), assign(C2,S2,G2), assign(C3,S3,G3), G1=3, S2=6, C3="Bolivia".
C1!=C3 :- assign(C1,S1,G1), assign(C2,S2,G2), assign(C3,S3,G3), G1=3, S2=6, C3="Bolivia".

% 6. Neither the team from Argentina nor the squad that won 2 silver medals is the squad that won 4 gold medals.
{C="Argentina"; S=2}=0 :- assign(C,S,G), G=4.
C1!=C2 :- assign(C1,S1,G1), assign(C2,S2,G2), C1="Argentina", S2=2.

% 7. The squad that won 2 gold medals is either the squad that won 6 silver medals or the team from Grenada.
{S=6; C="Grenada"}=1 :- assign(C,S,G), G=2.

Problem 3:
There are four people: Roberta, Thelma, Steve, and Pete. Among them, they hold eight different jobs. Each holds exactly two jobs. The jobs are: chef, guard, nurse, telephone operator, police officer (gender not implied), teacher, actor, and boxer. 
5. The job of nurse is held by a male.
6. The husband of the chef is the telephone operator, which means the chef is a female and the telephone operator is a male.
7. Roberta is not a boxer.
8. Pete has no education past the ninth grade. (Three jobs teacher, nurse, and police officer require higher education.)
9. Roberta, the chef, and the police officer went golfing together, meaning they are different people.
10.1 The same job cannot be assigned to different people.
10.2 Roberta is a female.
10.3 Thelma is a female.
10.4 Steve is a male.
10.5 Pete is a male.
10.6 The actor is a male.
Question: Who holds which jobs?

Constants:
<CONSTANTS>

Predicates:
<PREDICATES>

Constraints:'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', default='text-davinci-003', type=str,
        help='the engine name in \{gpt-4, text-davinci-003, text-davinci-002, text-curie-001\}')
    args = parser.parse_args()
    breakpoint()
    
    path = 'caches/prompt_cache_jobspuzzle_'+args.engine + '.json'
    if os.path.isfile(path):
        with open(path, 'r') as f:
            prompt_cache = json.load(f)
    else:
        prompt_cache = {}
        
    prompt = prompt_C
    constants = gen_response(prompt, args.engine, prompt_cache)
    
    prompt = prompt_P.replace('<CONSTANTS>',constants)
    
    predicates = gen_response(prompt, args.engine, prompt_cache)
    
    prompt = prompt_R1.replace('<CONSTANTS>',constants).replace('<PREDICATES>',predicates)
    
    generation_rules = gen_response(prompt, args.engine, prompt_cache)
    
    prompt = prompt_R2.replace('<CONSTANTS>',constants).replace('<PREDICATES>',predicates)
    
    constraints = gen_response(prompt, args.engine, prompt_cache)
    
    all_rules = generation_rules + '\n\n' + constraints
    
    print(all_rules)
    
    












import os
import pickle
import openai
import time
import csv

openai.api_key = ""#use your own key


def get_demo_examples():
    examples = []
    
    example = {}
    
    example['query'] = "worms stool"
    example['tgt_fq'] = "Are those parasitic worms I'm seeing in my stool sample?"
    example['clarification'] = "Do you want to know about parasitic worms in stool samples?"
    
    example['cands'] = ["Are those parasitic worms I'm seeing in my stool sample?"]
    
    
    examples.append(example.copy())

    example = {}
    
    example['query'] = "fecal test"
    example['tgt_fq'] = "Can the fecal fat test be done in my doctor's office?"  
    example['clarification'] = "Do you want to know about fecal tests at the doctor?"
    candidates = ["Can I just do the screening fecal fat test and not the 72-hour test?"]
    candidates.append("Can the fecal fat test be done in my doctor's office?")
    #candidates.append("")

    example['cands'] = candidates
    
    examples.append(example.copy()) 

    example = {}
    
    example['query'] = "TNF antibodies"
    example['tgt_fq'] = 'Once I have developed TNF inhibitor antibodies, will they go away?'

    candidates = ["If I develop antibodies to one type of TNF inhibitor therapy, and I am switched to a different one, will I develop antibodies to the second TNF inhibitor?"]
    candidates.append("Once I have developed TNF inhibitor antibodies, will they go away?")
    #candidates.append("")   
    
    example['cands'] = candidates
    example['clarification'] = "Do you want to know about if TNF inhibitor antibodies go away?"
    
    examples.append(example.copy()) 
    
    return examples
    

def output_examples_to_csv(output_examples, csv_path):

    fr = open(csv_path, 'w', newline='')
    writer = csv.writer(fr)
    
    #csv_firstrow = ['target question', 'query entities', 'valid clarifying entities', 'iir outputs', 'iir f1', 'iir hit rate', 't5 outputs', 't5 f1', 't5 hit rate', 't5 bleu', 'gpt3 outputs', 'gpt3 f1', 'gpt3 hit rate', 'gpt3 bleu']
    csv_firstrow = ['target question', 'query entities', 'options', 'gpt3 output']
    writer.writerow(csv_firstrow) 
    
    for ex in output_examples:
        row = []
        row.append(ex['tgt_fq'])
        row.append(ex['query'])
        row.append('\n'.join( [ cand for cand in ex['cands'] ] ) )
        row.append(ex['response'])
        
        writer.writerow(row)
    
    fr.close()
    
      


def use_gpt3():
        with open("turk_gpt3_inputs.pkl", "rb") as f_gpt3_inputs:
            test_examples = pickle.load(f_gpt3_inputs)
        
        #test_examples = get_demo_examples()
        
        #mode = "only query"
        mode = "only actual"
        #mode = "pair"
        #mode = "choices"
        #for mode in ("only query", )
        output_examples = []
        print(len(test_examples))
        for example in test_examples:
            prompt = generate_prompt(example, mode)
            output_example = example.copy()
            
            response = openai.Completion.create(engine="text-davinci-002",prompt=prompt,  temperature=0)
            output_example['response'] = response.choices[0].text
            #output_example['response'] = 'hi'
            output_examples.append(output_example)
            print(prompt)
            print(output_example)
            #print(len(output_examples))
            time.sleep(1)
            #input()
            
        csv_path = 'turk_gpt3_outputs_' + mode + '.csv'
        output_examples_to_csv(output_examples, csv_path)
        
        pkl_path = 'turk_gpt3_outputs_' + mode + '.pkl'
        with open(pkl_path, "wb") as f_gpt3_outputs:
            pickle.dump(output_examples, f_gpt3_outputs)
    

def index():
        fq = open('sample_qs.txt', 'r')
        fr = open('reponses.txt', 'w')

        lines = fq.readlines()
        for line in lines:           
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=generate_prompt(line.strip()),
                temperature=0.6,
            )
            fr.write(response.choices[0].text+'\n')
        
        fq.close()
        fr.close()

def generate_prompt(example, mode):
    instruction = "Playing as an assistant, you need to ask a clarification question to know what the user actually means"
    
    user = "user:"
    actually = "what the user actually means:"
    choices = "what the user possibly means:"
    assistant = "assistant:"
    

    
    
    '''
    queries = ["increase platelet"]
    questions = ["My platelet count is low. How can I increase it?"]
    clarifications = ["Do you want to know about ways to increase platelet count?"]
    candidates_list= [ [questions[0]] ]
    
    queries.append( "periods VWD" )
    questions.append( "If I have heavy menstrual periods, do I have von Willebrand disease (VWD)?" )    
    clarifications.append( "Do you want to know about the relation between periods and VWD?")

    queries.append( "prostate cancer" )
    questions.append( "What are some tests that may be done to help decide whether a prostate cancer is likely to be fast-growing and spread (metastasize) and therefore should be removed rather than watched?" ) 
    clarifications.append( "Do you want to know about tests for prostate cancer growing speed?")
    '''
    
    candidates_list = []
    
    queries = ["increase platelet"]
    questions = ["My platelet count is low. How can I increase it?"]
    clarifications = ["Do you want to know about ways to increase platelet count?"]
    candidates_list= [ ["My platelet count is low. How can I increase it?"] ]
    #candidates_list.append( candidates )
    
    queries.append( "flu test" )
    questions.append( "Like a flu test or strep test, can this test be performed in my healthcare practitioner's office?" )
    clarifications.append( "Do you want to know about flu tests done at your healthcare office" )
    candidates = ["Can I test negative and still have the flu?"]
    candidates.append("Like a flu test or strep test, can this test be performed in my healthcare practitioner's office?")
    #candidates.append()
    candidates_list.append( candidates )
    
    queries.append( "anticoagulant test" )
    questions.append( "I am on anticoagulant therapy. Can I still have this test done?" )    
    clarifications.append( "Do you want to know about having a test done while on anticoagulant therapy?")
    candidates = ["I am on anticoagulant therapy. Can I still have this test done?"]
    candidates.append("Is sample collection critical for lupus anticoagulant (LA) testing?")
    candidates.append("Can lupus anticoagulant interfere with the ACT test?")
    candidates_list.append( candidates )
    
    '''
    queries.append( "fecal test" )
    questions.append( "Can the fecal fat test be done in my doctor's office?" )    
    clarifications.append( "Do you want to know about fecal tests at the doctor?")
    candidates = ["Can I just do the screening fecal fat test and not the 72-hour test?"]
    candidates.append("Can the fecal fat test be done in my doctor's office?")
    #candidates.append("")
    candidates_list.append( candidates_list )
    '''
    
    #ex_keys = ['query', 'question', 'candidates' ,'clarification']
    #ex_values = [queries, questions, candidates_list, clarifications]
    #ex_dict = {}
    #for k, v in zip(ex_key, ex_values ):
    #    ex_dict[ex_key] = ex_values
    
    #mode = "only query"
    #mode = "only actual"
    #mode = "pair"
    #mode = "choices"
    
    n_exs = 3
    prompt = instruction + '\n\n'
    for i in range(n_exs):
        if mode in ["only query", "pair", "choices"]:   
            prompt = prompt + user + ' ' + queries[i] + '\n\n'
        
        if mode in ["only actual", "pair"]:
            prompt = prompt + actually + ' ' + questions[i] + '\n\n'
        
        if mode in ["choices"]:
            cand_i = 0
            prompt = prompt + choices + '\n'
            for cand in candidates_list[i]:
                #print(cand)
                bullet = '(' + str(cand_i+1) + ')'
                prompt = prompt +  bullet + ' ' + cand + '\n'
                cand_i += 1
            prompt += '\n'
        
        prompt = prompt + assistant + ' ' + clarifications[i] + '\n\n'
    
    if mode in ["only query", "pair", "choices"]:
        prompt = prompt + user + ' ' + example['query'] + '\n\n'
    if mode in ["only actual", "pair"]:
        prompt = prompt + actually + ' ' + example['tgt_fq'] + '\n\n'
    
    if mode in ["choices"]:
        cand_i = 0
        prompt = prompt + choices + '\n'
        for cand in example['cands']:
            #print(cand)
            bullet = '(' + str(cand_i+1) + ')'
            prompt = prompt +  bullet + ' ' + cand + '\n'
            cand_i += 1
        prompt += '\n'  
    
    prompt = prompt + assistant
    
    return prompt
    
    
         
    

def generate_prompt_1(question):
    return """Answer the question.

Question: Which animal is known as the 'Ship of the Desert"?
Answer:  Camel
Question:  How many days are there in a week?
Answer: 7 days
Question: {}
Answer:""".format(
        question
    )

def generate_prompt_bak(animal):
    return """Suggest three names for an animal that is a superhero.

Animal: Cat
Names: Captain Sharpclaw, Agent Fluffball, The Incredible Feline
Animal: Dog
Names: Ruff the Protector, Wonder Canine, Sir Barks-a-Lot
Animal: {}
Names:""".format(
        animal.capitalize()
    )

def main():
    #index()
    use_gpt3()

if __name__ == "__main__":
    main()

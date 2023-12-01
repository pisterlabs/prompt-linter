import numpy as np
from openai_public import query_azure_openai_chatgpt_chat,multi_threading_running
from utils import get_validation_set,shuffle_datapoints, Recorder,parse_solution,get_wrong_group,get_wrong_triples,extract_answer,data_process,get_triples, calculation_performance
from prompts import few_shot_prompt_add_data,few_shot_prompt,extract_prompt,inference_prompt,generation_solution_prompt,compress_prompt

def revision_process(query):
    text=generation_solution_prompt(query)
    solutions=query_azure_openai_chatgpt_chat(text)
    return parse_solution(solutions)

def compression_process(query):
   prompt=compress_prompt(query)
   temperature=0.6
   compressed_information = query_azure_openai_chatgpt_chat(prompt,temperature)
   return compressed_information

def verification_process(args,threshold,valid_scores,revisions,math_prompt,train_questions,train_labels,task_program=''):
    performance = []
    print("start validation")
    valid_questions, valid_labels = get_validation_set(train_questions, train_labels, args)
    for vv in range(len(revisions)):
        _, _, valid_performance = model_inference_batch(valid_questions, math_prompt + '\n' +task_program+ revisions[vv],valid_labels)

        print(vv,valid_performance,len(valid_questions))
        performance.append(valid_performance)
    recent_performance = (valid_scores[-1] + valid_scores[-2] + valid_scores[-3]) / 3.0
    gap = np.array(performance) - recent_performance
    if np.sum(gap > threshold)>0:
        max_id = np.argmax(gap)
        valid_scores.append(float(gap[max_id] + recent_performance))
        return valid_scores, revisions[max_id]
    else:
        return valid_scores, None

def Learning_to_program(args,threshold,valid_scores,wrong_groups,math_prompt,train_questions,train_labels,task_program=''):

    revision_candidate = multi_threading_running(revision_process, wrong_groups)
    compressed_revision_candidate = multi_threading_running(compression_process,revision_candidate)
    valid_scores,revision=verification_process(args,threshold,valid_scores,compressed_revision_candidate,math_prompt,train_questions,train_labels,task_program)
    return valid_scores,revision

def model_inference_sample(query):
    if args.few is True:
        prompt=few_shot_prompt_add_data(few_shot_prompt,query)
        answer_1 = query_azure_openai_chatgpt_chat(prompt)
    else:
        prompt=inference_prompt(query)
        answer_1 = query_azure_openai_chatgpt_chat(prompt)

    prompt2=extract_prompt(query[0],answer_1)
    answer_2 = query_azure_openai_chatgpt_chat(prompt2)
    answer_state, Answer_Texts = extract_answer(answer_2, query[2])
    return answer_state,answer_1,Answer_Texts

def model_inference_batch(questions,math_prompt,labels,task_program=''):
    triples = get_triples(questions, math_prompt+'\n'+task_program, labels)
    results = multi_threading_running(model_inference_sample, triples)
    wrong_triples=get_wrong_triples(results,triples)
    return  results, wrong_triples,calculation_performance(results)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='3', help='(default=%(default)s)')
    parser.add_argument('--model', type=str, default='gpt-35-turbo', help='(default=%(default)s)')
    parser.add_argument('--path', type=str, default='C:\\Users\\v-yiduoguo\\pycharmprojects\\pythonproject\\math_learning\\khan\\', help='(default=%(default)s)')
    parser.add_argument('--epoch', type=int,default=10,help='(default=%(default)s)')
    parser.add_argument('--batch_size', type=int, default=32, help='(default=%(default)s)')
    parser.add_argument('--valid_size', type=int, default=3, help='the number of validation samples is equal to valid_size*batch_szie')
    parser.add_argument('--T', type=int, default=5, help='Repeat T times to generate solution from errors')
    parser.add_argument('--threshold', type=float, default=1.0, help='Repeat T times to generate solution from errors')
    parser.add_argument('--few', type=bool,default=False, help='(default=%(default)s)')
    args = parser.parse_args()
   # args.few = True
    math_prompt = "\nLet's think step by step."
    task_program=''
    train_questions, train_labels,train_hints,test_questions, test_labels=data_process(args)
    #print(len(train_questions),len(test_questions))
    update_correct_time=[]
    update_failure_time = []
    valid_scores=[]
    recorder=Recorder(args)
    if args.few:
        few_shot_prompt=few_shot_prompt(train_questions,train_labels,train_hints)

    for epoch in range(args.epoch):
        train_questions,train_labels=shuffle_datapoints(train_questions,train_labels)
        update_program=False
        if epoch == 0:
            print('Start task',args.name)
            _,_,test_performance=model_inference_batch(test_questions,math_prompt,test_labels,task_program)

            for _ in range(3):
                valid_questions, valid_labels = get_validation_set(train_questions, train_labels,args)
                _, _,valid_performance = model_inference_batch(valid_questions, math_prompt, valid_labels)
                valid_scores.append(valid_performance)
            print('Original test performance',test_performance, 'Original valid performance', (valid_scores[-1]+valid_scores[-2]+valid_scores[-3])/3.0)
        for batch_id in range(len(train_questions)//args.batch_size):
            print('training')
            results, wrong_triples,train_performance = model_inference_batch(train_questions[args.batch_size*batch_id:args.batch_size*(batch_id+1)], math_prompt, train_labels[args.batch_size*batch_id:args.batch_size*(batch_id+1)],task_program)
            wrong_groups=get_wrong_group(wrong_triples,task_program)
            valid_scores,revison=Learning_to_program(args,args.threshold,valid_scores,wrong_groups,math_prompt,train_questions,train_labels,task_program)
            if revison is not None:
                task_program+='\n'+revison
                _, _, test_performance = model_inference_batch(test_questions, math_prompt, test_labels,
                                                               task_program)

                update_correct_time.append(1)
                update_program=True
                print('epoch :',epoch,'batch_id :',batch_id,'find new solution',revison,'test_performance',test_performance)
            else:
                update_failure_time.append(1)
            if len(update_correct_time)==3:
                    print("start compression!")
                    compressed_task_program = multi_threading_running(compression_process,[task_program for _ in range(5)])

                    valid_scores, effective_compressed_task_program = verification_process(args,-1.0, valid_scores, compressed_task_program,
                                                                    math_prompt, train_questions, train_labels)

                    if effective_compressed_task_program is not None:
                        print("success compression!",effective_compressed_task_program)
                        task_program=effective_compressed_task_program
                        update_correct_time=[]
                        update_program = True
        if update_program:
            _, _, test_performance = model_inference_batch(test_questions, math_prompt, test_labels,task_program)
            print('We update h in this epoch', epoch,task_program, 'the test performance now is',
                  test_performance)
        else:
            test_performance = None
            print('No found in this epoch', epoch)
        recorder.update(epoch,task_program,test_performance)

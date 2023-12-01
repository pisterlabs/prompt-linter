from django.shortcuts import render,redirect
import cohere
import os
from joblib import load
import numpy as np
import openai
COHERE_API_KEY = os.environ['COHERE_API_KEY']
co = cohere.Client(f"{COHERE_API_KEY}")
rfc_model_new = load("./savedModels/rfc_model_new_joblib.joblib")
wt_knn = load("./savedModels/wt_knn_joblib.joblib")
symptoms = load("./savedModels/symptoms_jobliib.joblib")
classes = load("./savedModels/classes_jobliib.joblib")

def diseasePrediction(request):
    if request.user.is_authenticated:
        if request.method == 'POST':
            symptoms_list = request.POST.getlist('options')
            def top_five(classes, model, input_list):
                mapping = {}
                pred_prob = model.predict_proba([input_list])

                for disease, prob in zip(classes, pred_prob[0]):
                    mapping[disease] = prob

                result = dict(sorted(mapping.items(), key=lambda x: x[1], reverse=True))

                # Get the keys of the dictionary as a list and slice the list
                sliced_keys = list(result.keys())[:3]

                # Create a new dictionary with the sliced keys
                sliced_dict = {k: result[k] for k in sliced_keys}

                # Print the sliced dictionary
                return sliced_dict
            def final_op(dict1, dict2):
                diseases = set(dict1.keys()) | set(dict2.keys())
                priors = {disease: 1.0/len(diseases) for disease in diseases}
                posterior = {}
                for disease in diseases:
                    if disease in dict1 and disease in dict2:
                        posterior[disease] = dict1[disease] * dict2[disease] * priors[disease]
                    elif disease in dict1:
                        posterior[disease] = dict1[disease] * priors[disease]
                    elif disease in dict2:
                        posterior[disease] = dict2[disease] * priors[disease]
                total_prob = sum(posterior.values())
                for disease in posterior:
                    posterior[disease] /= total_prob
                sorted_dict = dict(sorted(posterior.items(), key=lambda item: item[1], reverse=True))

                for key, val in sorted_dict.items():
                    sorted_dict[key] = round(val*100,3)
                return sorted_dict
            new_data = symptoms_list
            input_list = []
            for i in symptoms:
                if i in new_data:
                    input_list.append(1)
                else:
                    input_list.append(0)
            input_list = np.array(input_list)
            rfc_new_model_dict = top_five(classes, rfc_model_new, input_list)
            wt_knnl_dict = top_five(classes, wt_knn, input_list)
            final = final_op(wt_knnl_dict, rfc_new_model_dict)
            information = []
            for disease in final.keys():
                response = co.generate( model="command-nightly",
                    prompt=f"Give basic remedy for {disease}",
                    max_tokens=100,
                    temperature=0.2,
                    k=0,
                    p=0.75)
                # completions = openai.Completion.create(
                #     model="text-davinci-003",
                #     prompt=[{"role":"user", "content":f"Provide 3 simpe remedies {disease}"}]
                # )
                # completions = openai.Completion.create(
                # engine="gpt-3.5-turbo-instruct",
                # prompt= f"Provide 3 simpe remedies {disease}",
                # max_tokens=60,
                # n=1,
                # temperature=0.5,
                # )
                # response = completions.choices[0].text
                information.append(response.generations[0].text)
                
            
            return render(request, "predict.html", {"result": final , "symptoms":symptoms , 'my_info': information})
        return render(request, "predict.html", {"symptoms":symptoms})
    else:
        return redirect("landing")
        
def firstAid(request):
    if request.user.is_authenticated:
        return render(request, "firstAid.html")
    else:
        return redirect("landing")

def doctorRecommendation(request):
    if request.user.is_authenticated:
        return render(request, "doctor.html")
    else:
        return redirect("landing")

def docMaps(request):
    if request.user.is_authenticated:
        return render(request, "maps.html")
    else:
        return redirect("landing")

import csv
import cohere
import pickle
co = cohere.Client('rKNbSEzFUz1Naxs2ZwQQpZOL3IsPAY4pKLIpLDnG')
f_output = open('CohereFinal.txt', 'w')

arr_prediction = []
with open('Final_NasaFIRM_clean.csv', 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    if len(row):
        promptVal = "Write a description for the following fire incident with the following pieces of data in the form of 1-3 sentences. Do not make up any information, use only exactly what is given. Make sure to include all of the data, especially the intensity - The time in which the fire happened: " + row[4] + ", Location: " + row[1] + ", Date: " + row[2] + ", spotted at: " + row[3] + ", intensity of fire: "+ row[5]
        response = co.generate(
            model='command-xlarge-nightly',
            
            prompt=promptVal,
            max_tokens=314,
            temperature=0.3,
            k=0,
            p=0.75,
            frequency_penalty=0,
            presence_penalty=0,
            stop_sequences=[],
            return_likelihoods='NONE'
        )
        f_output.write("\n---PROMPT ---- \n" + promptVal+ "\n")
        f_output.write('\n---Prediction--- {}\n'.format(response.generations[0].text))

        arr_prediction.append(response.generations[0].text)
        pickle.dump(arr_prediction, open("cohere_prediction.p", "wb"))
        #print("\nPROMPT ---- \n" + promptVal)
        #print('\nPrediction--- \n {}'.format(response.generations[0].text))
        

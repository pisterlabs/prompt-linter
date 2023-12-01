import json
import openai
from time import sleep
import os

# openai.organization = os.environ.get("OPENAI_ORG_KEY")
# openai.api_key = os.environ.get("OPENAI_API_KEY")

openai.organization = "org-2wXrLf4fLEfdyawavmkAqi8z"
openai.api_key = "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"

file = open("splitting_examples.json", "r")
data = json.load(file)
file.close()

exemple= data["0"]
paragraphs = exemple["text"]
base = exemple["base"]

exemple_incomplet = "onze mai mil neuf cent quinze, vingt-quatre ans, domiciliée à Paris, 7, rue des Prêcheurs *\nfille de Charles Célestin Edouard LEPAGE, décédé, t de Pauline Désirée Germain VAPPEREAU * sa\nveuve * matelassière * domiciliée 7, rue des Prêcheurs, d'autre part Les futurs époux déclarent\nqu'il n'a pas été fait de contrat de mariage .- Marius Fernand VIDAL et Charlotte Lucie LEPAGE ont\ndécalé l'un après l'autre vouloir se prendre pour époux et Nous avons prononcé au nom de la\nloi qu'ils sont unis par le mariage En présence de: Charles GAUTHIER * employé, 4, Place du\nLouvre et de Pauline LEPAGE, matelassière * 7, rue des Prêcheurs * témoins majeurs, qui, lecture\nfaite, ont signé avec les époux et Nous, Charles Louis TOURY * Officier de l'état-civil du premier\narrondissement de Paris, Chevalier de la Légion d'Honneur ./."
paragraphes_incomplets = {
    "p1": "",
    "p2": "",
    "p3": "onze mai mil neuf cent quinze, vingt-quatre ans, domiciliée à Paris, 7, rue des Prêcheurs *\nfille de Charles Célestin Edouard LEPAGE, décédé, t de Pauline Désirée Germain VAPPEREAU * sa\nveuve * matelassière * domiciliée 7, rue des Prêcheurs, d'autre part",
    "p4": "Les futurs époux déclarent\nqu'il n'a pas été fait de contrat de mariage .- Marius Fernand VIDAL et Charlotte Lucie LEPAGE ont\ndécalé l'un après l'autre vouloir se prendre pour époux et Nous avons prononcé au nom de la\nloi qu'ils sont unis par le mariage",
    "p5": "En présence de: Charles GAUTHIER * employé, 4, Place du\nLouvre et de Pauline LEPAGE, matelassière * 7, rue des Prêcheurs * témoins majeurs, qui, lecture\nfaite, ont signé avec les époux et Nous, Charles Louis TOURY * Officier de l'état-civil du premier\narrondissement de Paris, Chevalier de la Légion d'Honneur"
}

exemple_incomplet_debut = "Le trente mars mil neuf cent quarante * onze heures cinq minutes ****\ndevant Nous ont comparu publiquement en la maison commune: Ermel Eug\u00e8ne LUCAS * imprimeur n\u00e9 \u00e0 Saint\nChristophe du Bois * Maine et Loire * le dix octobre mil neuf cent, trente-neuf ans, domicili\u00e9 \u00e0 Cholet\nMaine et Loir * 62, rue de Lorraine * actuellement aux arm\u00e9es; fils de Auguste Maximin LUCAS * d\u00e9c\u00e9d\u00e9, et\nde Marie Louise CHARBONNEAU, sa veuve, sans profession, domicili\u00e9e \u00e0 Cholet Divorc\u00e9 de Micheline\nHenriette SEROT, d'une part ,/- ET Am\u00e9lie Eug\u00e9nie LE GALLE, journali\u00e8re, n\u00e9e \u00e0 Le Palais * Morbihan *\nle quinze octobre mil neuf cent dix, vingt-neuf ans, domicili\u00e9e \u00e0 Paris, 13, rue de la Ferronnerie;\nfille de Joachim Marie LE GALLE, d\u00e9c\u00e9d\u00e9, et de Marie France"
paragraphes_incomplets_debut = {
    "p1" : "Le trente mars mil neuf cent quarante * onze heures cinq minutes ****\ndevant Nous ont comparu publiquement en la maison commune:",
    "p2" : "Ermel Eug\u00e8ne LUCAS * imprimeur n\u00e9 \u00e0 Saint\nChristophe du Bois * Maine et Loire * le dix octobre mil neuf cent, trente-neuf ans, domicili\u00e9 \u00e0 Cholet\nMaine et Loir * 62, rue de Lorraine * actuellement aux arm\u00e9es; fils de Auguste Maximin LUCAS * d\u00e9c\u00e9d\u00e9, et\nde Marie Louise CHARBONNEAU, sa veuve, sans profession, domicili\u00e9e \u00e0 Cholet Divorc\u00e9 de Micheline\nHenriette SEROT, d'une part ,/-",
    "p3" : "ET Am\u00e9lie Eug\u00e9nie LE GALLE, journali\u00e8re, n\u00e9e \u00e0 Le Palais * Morbihan *\nle quinze octobre mil neuf cent dix, vingt-neuf ans, domicili\u00e9e \u00e0 Paris, 13, rue de la Ferronnerie;\nfille de Joachim Marie LE GALLE, d\u00e9c\u00e9d\u00e9, et de Marie France",
    "p4" : "",
    "p5" : ""
}

def split_text(text_to_split):
    for i in range(3):
        for i in range(3):
            try:
                completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k",
                temperature=0.4,
                messages=[
                    {"role": "system", "content": "The user will provide you with French Mariage acts. You must always split these acts into 5 paragraphs. If a paragraphs seems to be missing, answer with an empty paragraph. The parapgrahs are : p1 = Date et heure maison commune\n p2 = Le mari et son entourage\n p3 = La mariée et son entourage\n p4 = Présence d'un acte de mariage, baiser et union des partenaires\n p5 = 'En présence de ...', Les informations concernant les témoins et l'adjoint au maire.\n\n"},
                    {"role": "user", "content" : data["0"]["base"]},
                    {"role": "assistant", "content": json.dumps(data["0"]["text"], indent=4)},
                    {"role": "user", "content" : data["1"]["base"]},
                    {"role": "assistant", "content": json.dumps(data["1"]["text"], indent=4)},
                    {"role": "user", "content": exemple_incomplet},
                    {"role": "assistant", "content": json.dumps(paragraphes_incomplets, indent=4)},
                    {"role": "user", "content": exemple_incomplet_debut},
                    {"role": "assistant", "content": json.dumps(paragraphes_incomplets_debut, indent=4)},
                    {"role": "user", "content" : text_to_split}
                ]
                )


                break
            except:
                completion = None
                print("Error while getting answer. Retry in 5 seconds")
                sleep(5)
                continue

        if completion is None:
            print("Error while getting answer. Returning...")

        #check if keys are p1, p2, p3, p4, p5 and if values are not empty or ''
        answer = completion.choices[0].message['content']
        
        answer = answer.replace('\n', '').replace('.','')


        answer = answer.replace('-\\n', '')
        answer = answer.replace('\\n', ' ')

        answer = answer.replace('"', '\'')

        answer = answer.replace("'p1'", '"p1"')
        answer = answer.replace("'p2'", '"p2"')
        answer = answer.replace("'p3'", '"p3"')
        answer = answer.replace("'p4'", '"p4"')
        answer = answer.replace("'p5'", '"p5"')

        answer = answer.replace('{\'', '{"')
        answer = answer.replace('{ \'', '{ "')

        answer = answer.replace('\'}', '"}')
        answer = answer.replace('\' }', '" }')

        answer = answer.replace('\':', '":')
        answer = answer.replace('\' :', '" :')

        answer = answer.replace(':\'', ':"')
        answer = answer.replace(': \'', ': "')

        answer = answer.replace('\',', '",')
        answer = answer.replace('\' ,', '" ,')

        answer = answer.replace(',\'', ',"')
        answer = answer.replace(', \'', ', "')

        #remplacer les apostrophes par des guillemets
        answer = answer.replace("\\'", "\'")
        #print(answer)
        answer = answer[answer.index('{'):]
        print(f'answer : {answer}')
        answer = json.loads(answer)
        #print(answer)

        #check if keys are p1, p2, p3, p4, p5 and if values are not empty or '' :
        if 'p1' not in answer.keys() or 'p2' not in answer.keys() or 'p3' not in answer.keys() or 'p4' not in answer.keys() or 'p5' not in answer.keys():
            print('At least one paragraph is missing... retrying')
            sleep(5)
            continue
            
        break

    return answer


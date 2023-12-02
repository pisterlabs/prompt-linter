import openai
import tiktoken
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from retrying import retry
import requests


from scripts.util import average_bert_score, average_rouge_scores, export_output, get_val_dataset, \
    truncate_text, number_of_tokens, tokenize, deepinfra_create, setup_logger, prepare_dataset

import os
from nltk.translate import meteor_score
import numpy as np
import argparse
import wandb

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu
from rouge import Rouge
from evaluate import load

import logging

# implement timer
import time
import datetime

### Initialization

import os
from dotenv import load_dotenv
load_dotenv()


def log_test_scores(meteor_score_avg, rouge_score_avg, bleu_score_avg, bert_score_avg):
    wandb.log({
        "METEOR_score/test": meteor_score_avg,
        "ROUGE_score/test/rouge-1/r": rouge_score_avg['rouge-1']['r'],
        "ROUGE_score/test/rouge-1/p": rouge_score_avg['rouge-1']['p'],
        "ROUGE_score/test/rouge-1/f": rouge_score_avg['rouge-1']['f'],
        "ROUGE_score/test/rouge-2/r": rouge_score_avg['rouge-2']['r'],
        "ROUGE_score/test/rouge-2/p": rouge_score_avg['rouge-2']['p'],
        "ROUGE_score/test/rouge-2/f": rouge_score_avg['rouge-2']['f'],
        "ROUGE_score/test/rouge-l/r": rouge_score_avg['rouge-l']['r'],
        "ROUGE_score/test/rouge-l/p": rouge_score_avg['rouge-l']['p'],
        "ROUGE_score/test/rouge-l/f": rouge_score_avg['rouge-l']['f'],
        "BLEU_score/test": bleu_score_avg,
        "BERT_score/test/precision": bert_score_avg['precision'],
        "BERT_score/test/recall": bert_score_avg['recall'],
        "BERT_score/test/f1": bert_score_avg['f1']
    })
    print()
    logger.info(f"Average METEOR score: {meteor_score_avg:.4f}")
    logger.info(f"Average ROUGE score: {rouge_score_avg}")
    logger.info(f"Average BLEU score: {bleu_score_avg:.4f}")
    logger.info(f"Average BERTScore: {bert_score_avg}")
    print()

def compute_scores(completion_dataset, num_examples=10):
    scores = {'meteor': [], 'rouge': [], 'bleu': [], 'bert': []}
    rouge = Rouge()
    bertscore = load("bertscore")

    for idx, entry in enumerate(completion_dataset):
        """
        target_text_tokens: list of tokens
        predicted_text_tokens: list of tokens

        predicted_text: string (plain text)
        target_text: string (plain text)

        tokenized_target_text: string (tokens)
        tokenized_predicted_text: string (tokens)
        """
        target_text_tokens = tokenize(entry['target'], model_name)
        predicted_text_tokens = tokenize(entry['predicted'], model_name)

        predicted_text = entry['predicted']
        target_text = entry['target']

        tokenized_target_text = ' '.join(target_text_tokens)
        tokenized_predicted_text = ' '.join(predicted_text_tokens)

        # Calculate Meteor scores
        meteor = meteor_score.meteor_score([target_text_tokens], predicted_text_tokens)
        scores['meteor'].append(meteor)

        # Calculate Rouge scores
        rouge_scores = rouge.get_scores(predicted_text, target_text)[0]
        scores['rouge'].append(rouge_scores)

        # Calculate Bleu scores
        bleu = sentence_bleu([tokenized_target_text], tokenized_predicted_text, weights=(.25, .25, .25, .25))
        scores['bleu'].append(bleu)

        # Calculate BERTScore
        bert = bertscore.compute(predictions=[predicted_text], references=[target_text],
                                 model_type="bert-base-multilingual-cased", lang=['de', 'fr', 'it'])
        scores['bert'].append(bert)

        output_examples.append({
            'language': entry['lang'],
            'input': entry['input'],
            'target': target_text,
            'predicted': predicted_text,
            'meteor': meteor,
            'bert-f1': bert['f1'][0],
            'bleu': bleu,
            'rouge-1_f1': rouge_scores['rouge-1']['f'],
            'rouge-2_f1': rouge_scores['rouge-2']['f'],
            'rouge-l_f1': rouge_scores['rouge-l']['f'],
            'bert_full': bert,
            'rouge_full': rouge_scores,
            })

        # Print examples
        if idx < num_examples:
            print("\n", flush=True)
            print("#" * 180, flush=True)
            logger.info(f"Example {idx + 1} of {len(completion_dataset)}")
            logger.info(f"Output: {predicted_text}")
            logger.info("-" * 100)
            logger.info(f"Label: {target_text}")
            logger.info("-" * 100)
            logger.info(f"METEOR Score: {meteor:.4f}")
            logger.info(f"ROUGE Score: {rouge_scores}")
            logger.info(f"BLEU Score: {bleu:.4f}")
            logger.info(f"BERTScore: {bert}")
            print("#" * 180, flush=True)
            print("\n", flush=True)

    return np.mean(scores['meteor']), average_rouge_scores(scores['rouge']), np.mean(
        scores['bleu']), average_bert_score(scores['bert'])


    # we want to return a list of dicts with the keys "input" and "target"
    return data_list

def get_one_shot_example(task, lang):
    if task == 'cvg':
        if lang == 'de':
            pre_text = "\n\nHier ein Beispiel für einen Sachverhalt und die dazugehörigen Erwägungen:"
            example_input = "Sachverhalt:\r\n1. B._, geboren 1969, war bei der A._ AG als Montagemitarbeiterin angestellt und dadurch bei der Schweizerischen Unfallversicherungsanstalt (SUVA) obligatorisch gegen die Folgen von Unfällen versichert. Am 2. Juni 2005 erlitt sie einen Arbeitsunfall und zog sich dabei Verletzungen am rechten Zeigefinger zu (Urk. 1 S. 3, Urk. 2 S. 2 lit. A). In der Folge erbrachte die SUVA die gesetzlichen Versicherungsleistungen, bis sie diese mit Verfügung vom 4. September 2006 per 1. Dezember 2006 einstellte (Urk. 3). Die dagegen am 25. September 2006 erhobene Einsprache wurde mit Einspracheentscheid vom 20. September 2007 abgewiesen (Urk. 2).\r\n2. Gegen den Einspracheentscheid vom 20. September 2007 erhob die Versicherte am 22. Oktober 2007 Beschwerde und beantragte die Zusprache der zustehenden gesetzlichen Leistungen (Urk. 1 S. 2). In formeller Hinsicht beantragte die Versicherte die Durchführung eines zweiten Schriftenwechsels, nachdem eine Stellungnahme von Dr. med. C._ zur ärztlichen Beurteilung durch die SUVA Versicherungsmedizin noch ausstehend sei (Urk. 1 S. 7).\r\nMit Eingabe vom 29. November 2007 beantragte die SUVA sodann die Sistierung des Verfahrens bis zum Vorliegen der Stellungnahme durch Dr. C._ (Urk. 7). "
            example_output = "Das Gericht zieht in Erwägung:\r\n1. Das Gericht kann die Angelegenheit zu neuer Entscheidung an die Vorinstanz zurückweisen, besonders wenn mit dem angefochtenen Entscheid nicht auf die Sache eingetreten oder der Sachverhalt ungenügend festgestellt wurde (§ 26 Abs. 1 des Gesetzes über das Sozialversicherungsgericht, GSVGer). In erster Linie kommt eine Rückweisung in Frage, wenn der Versicherungsträger auf ein Begehren überhaupt nicht eingetreten ist oder es ohne materielle Prüfung abgelehnt hat, wenn schwierige Ermessensentscheide zu treffen sind, oder wenn der entscheidrelevante Sachverhalt ungenügend abgeklärt ist (vgl. SVR 1995 ALV Nr. 27 S. 69).\r\n2.\r\n2.1 Gemäss den Ausführungen der Beschwerdeführerin in der Beschwerde vom 22. Oktober 2007 ist eine Stellungnahme von Dr. med. C._ zur ärztlichen Beurteilung durch die SUVA Versicherungsmedizin ausstehend (Urk. 1 S. 7). Dieser habe ein ganz erhebliches Schmerzsyndrom festgestellt, welches als CRPS Typ II definiert werde, und massgeblich für die Arbeitsunfähigkeit verantwortlich sei (Urk. 1 S. 6 f.). Es fehle bisher jede nachvollziehbare Erklärung dafür, weshalb Dr. C._ nicht fähig sein solle, ein CRPS Typ II als Folge eines Traumas zu diagnostizieren. Aus diesem Grund sei die von der Beschwerdegegnerin beigelegte Beurteilung vom 31. August 2007 Dr. C._ zur Stellungnahme unterbreitet worden (Urk. 1 S. 7).\r\n2.2 Die Beschwerdegegnerin sodann ging in ihrer Eingabe vom 29. November 2007 davon aus, dass sich aus der von der Beschwerdeführerin in Aussicht gestellten Stellungnahme neue Erkenntnisse bezüglich des medizinischen Sachverhaltes ergeben könnten und beantragte eine Sistierung des Verfahrens bis zum Vorliegen der Stellungnahme von Dr. C._. Es sei davon auszugehen, dass diese Stellungnahme auch der SUVA Versicherungsmedizin noch vorgelegt werden müsse (Urk. 7).\r\n2.3 Insgesamt ist gestützt auf die Ausführungen der Parteien davon auszugehen, dass ärztliche Berichte ausstehend sind, welche für die Feststellung des medizinischen Sachverhalts erheblich sind. Der dem vorliegenden Fall zugrunde liegende Sachverhalt erscheint somit ungenügend abgeklärt und die entscheidrelevanten Unterlagen nicht vollständig. Die Beschwerde ist daher in dem Sinne gutzuheissen, dass der angefochtene Einspracheentscheid aufgehoben und die Sache an die Beschwerdegegnerin zurückgewiesen wird, damit diese nach der abschliessenden Abklärung des medizinischen Sachverhaltes sowie der Gewährung des rechtlichen Gehörs gestützt auf die vollständigen Unterlagen über den Leistungsanspruch der Beschwerdeführerin neu entscheide.\r\n3. Nach § 34 Abs. 1 GSVGer hat die obsiegende Beschwerde führende Person Anspruch auf Ersatz der Parteikosten. Diese werden ohne Rücksicht auf den Streitwert nach der Bedeutung der Streitsache, der Schwierigkeit des Prozesses und dem Masse des Obsiegens bemessen (§ 34 Abs. 3 GSVGer). Vorliegend erscheint eine Prozessentschädigung von Fr. 1\'600.-- (inkl. Mehrwertsteuer und Barauslagen) als angemessen."
        elif lang == 'fr':
            pre_text = "\n\nVoici un exemple de faits et de considérations:"
            example_input = "EN FAIT\r\n1. Monsieur S_, né le _1984, domicilié à Louga au Sénégal, est titulaire d’un baccalauréat obtenu dans son pays en été 2005 avec la mention \"passable\" et la moyenne de 10 sur 20. De plus, il a effectué une année d’études littéraires à l’Université Cheikh Anta Diop de Dakar.\r\n2. Le 8 mars 2006, M. S_ a rempli une demande d’immatriculation auprès de l’Université de Genève en postulant pour le diplôme de l’école de langue et de civilisation françaises.\r\n3. Le 11 août 2006, la division administrative et sociale des étudiants (ci-après : Dase) a écrit à M. S_ que sa demande d’immatriculation était refusée puisque d’une part, le baccalauréat qu’il produisait ne comportait aucune mention, la moyenne de 14 sur 20 étant requise et que d’autre part, il avait effectué une année universitaire seulement, au lieu des deux requises.\r\n4. Par lettre du 24 août, réceptionnée par l’Université de Genève le 30 août 2006, M. S_ a réitéré sa demande d’immatriculation à laquelle il tenait beaucoup. Ce courrier a été considéré comme une opposition.\r\n5. Par décision du 11 septembre 2006, la DASE a rejeté l’opposition pour les motifs déjà indiqués dans la décision initiale.\r\n6. Par acte posté le 20 septembre 2006 et reçu le 26 septembre 2006, complété par un courrier réceptionné le 17 octobre 2006, M. S_ a prié la commission de recours de l’Université (ci-après : CRUNI) de faire preuve d’indulgence à son égard et d’accepter son immatriculation car il voulait poursuivre ses études à l’Université de Genève compte tenu de la réputation et de la qualité d’enseignement de celle-ci.\r\n7. Le 16 octobre 2006, l’Université de Genève a conclu au rejet du recours, l’intéressé ne satisfaisant pas aux exigences légales pour être admis comme étudiant.\r\n8. Le 3 novembre 2006, la vice-présidente de la CRUNI a prié l’Université de lui faire parvenir les directives élaborées par la conférence des recteurs des Universités suisses (CRUS) à laquelle la réponse faisait référence.\r\nL’Université a fait parvenir, le 14 novembre 2006, les textes en question, qui ont été transmis pour information au recourant.\r\n9. Sur quoi, la cause a été gardée à juger. "
            example_output = " EN DROIT\r\n1. Dirigé contre la décision sur opposition du 11 septembre 2006 et interjeté dans le délai légal et la forme prescrite auprès de l’autorité compétente, le recours est recevable (art. 62 de la loi sur l’Université du 26 mai 1973 - LU –\r\nC 1 30\r\n; art. 87 du règlement de l’Université du 7 septembre 1988 - RU –\r\nC 1 30.06\r\n; art. 26 et 27 du règlement interne relatif aux procédures d’opposition et de recours du 25 février 1977 - RIOR).\r\n2. a. A teneur de l’article 63D alinéa 1 LU, les personnes qui possèdent une maturité gymnasiale, un diplôme de fin d’études délivré par une haute école spécialisée (HES) ou un titre jugé équivalent sont admises à l’immatriculation. Pour le surplus, les conditions d’immatriculation sont fixées par le RU (art. 63D al. 3 LU).\r\nb. Selon l’article 15 RU, les candidats qui possèdent une maturité fédérale, une maturité cantonale reconnue ou un titre équivalent sont admis à l’immatriculation ; le rectorat détermine l’équivalence des titres (art. 15 al. 1 et 2 RU). La CRUNI a déjà jugé que cette délégation n’était pas contestable (\r\nACOM/64/2005\r\ndu 27 septembre 2005).Le tableau des équivalences est publié par le rectorat dans une brochure intitulée \"Conditions d’immatriculation\", distribuée à tous les candidats à l’immatriculation.\r\nc. S’agissant des titulaires d’un baccalauréat général sénégalais, la moyenne minimale exigée est de 14 sur 20. Le recourant ayant obtenu une moyenne de 10 sur 20 ne satisfait pas à cette exigence.\r\nd. De plus, il résulte également du RU et de la brochure précitée que lorsque la moyenne minimale exigée par l’Université de Genève pour le baccalauréat n’a pas été atteinte, elle peut éventuellement être compensée par la réussite préalable de deux années au moins (120 crédits ECTS) d’études universitaires dans la même orientation que celle choisie à l’Université de Genève. Or, M. Sylla, comme indiqué ci-dessus, n’a effectué qu’une année académique au Sénégal ce qui ne correspond pas aux 120 crédits ECTS requis.\r\nEn conséquence, M. S_ ne peut être immatriculé à l’Université de Genève, raison pour laquelle son recours sera rejeté selon une jurisprudence constante de la CRUNI (ACOM/82/2006 du 20 septembre 2006 ;ACOM/39/2005 du 1er juin 2005). La décision sur opposition ne peut qu’être confirmée.\r\n4. Vu la nature du litige aucun émolument ne sera perçu (art. 33 RIOR)."
        elif lang == 'it':
            pre_text = "\n\nEcco un esempio di fatti e considerazioni:"
            example_input = "in fatto: \r\nA. \r\nIl 29 settembre 2005 RI 1 ha sottoscritto con la CO 1 un contratto di lavoro con il quale è stato assunto in qualità di addetto al trattamento termico per uno stipendio mensile di fr. 3550.– per tredici mensilità. Il rapporto di lavoro, che ha avuto inizio il 3 ottobre 2005, si è concluso il successivo 31 dicembre.\r\nB.\r\nIl 15 febbraio 2006 RI 1 ha convenuto CO 1 davanti al Giudice di pace del circolo della Riviera per ottenere il pagamento di fr. 1597.90 rivendicati quale differenza tra il salario effettivamente percepito e quello di sua spettanza sulla base del CCL al quale la convenuta è assoggetta, essendo egli in possesso di un certificato federale di capacità quale polimeccanico, e dovendo quindi essere remunerato quale operaio qualificato. All\'udienza del 28 febbraio 2006, indetta per la discussione, la convenuta si è opposta all\'istanza rilevando di aver assunto l\'istante non come operaio qualificato ma solo per permettergli di completare la sua formazione professionale, tant\'è che essa ha chiesto all\'Ufficio cantonale delle misure attive un bonus di inserimento, ragione per cui il salario corrisponde a quello dovuto a una persona in formazione.\r\nC.\r\nCon sentenza 4 aprile 2006 il Giudice di pace, basandosi sulle prove documentali che hanno confermato la tesi della convenuta, ha respinto le pretese salariali del lavoratore avendo questi percepito uno stipendio conforme all\'attività svolta in seno alla convenuta.\r\nD.\r\nCon il presente tempestivo gravame RI 1 è insorto contro il predetto giudizio chiedendone l\'annullamento. Il ricorrente rimprovera al primo giudice di aver arbitrariamente valutato le prove documentali ed erroneamente applicato il diritto sostanziale non riconoscendo il carattere vincolante dei salari minimi previsti dal CCL al quale era assoggettata la convenuta e non considerando la documentazione dallo stesso prodotta a comprova del fatto che le mansioni a lui affidate rientravano tra quelle oggetto della sua formazione professionale di operaio qualificato e non semplicemente in formazione come preteso dalla convenuta. Nelle sue osservazioni del 2 maggio 2006 la convenuta postula la reiezione del ricorso. "
            example_output = "Considerando\r\nin diritto: 1.\r\nGiusta l\'art. 327 lett. g CPC una sentenza del Pretore o del Giudice di pace può essere annullata quando è stata manifestamente violata una norma di diritto materiale o formale oppure in caso di valutazione manifestamente errata di atti di causa o di prove. Per costante giurisprudenza del Tribunale federale una decisione è arbitraria quando viola gravemente una norma o un principio giuridico chiaro ed indiscusso o quando contrasta in modo intollerabile con il sentimento della giustizia e dell\'equità. Arbitrio e violazione della legge non vanno confusi; per essere definita come arbitraria tale violazione dev\'essere manifesta e riconosciuta (o riconoscibile) a prima vista; l\'arbitrio non può essere ravvisato già nella circostanza che un\'altra soluzione sarebbe immaginabile o persino preferibile; è doveroso scostarsi da questa scelta solamente se simile soluzione appare come insostenibile, in contraddizione palese con la situazione reale, non sorretta da ragione oggettiva o lesiva di un diritto certo (DTF 132 I 17 consid. 5.1).\r\n2.\r\nIl ricorrente contesta l\'accertamento del Giudice di pace secondo cui il salario pattuito corrispondeva a quello per un periodo di formazione. Egli ribadisce che si trattava di lavoro per il quale egli possedeva il necessario certificato di capacità e quindi da remunerare quale operaio qualificato ai sensi dell\'art. 14 del CCL. Ora, di fronte alle due tesi contrapposte non può dirsi che quella condivisa dal primo giudice sia arbitraria, ovvero insostenibile, già per il fatto che non risulta essere in contraddizione con le risultanze istruttorie. Al riguardo, è vero che l\'art. 14 CCL definisce operaio qualificato colui che è in possesso di un attestato di capacità federale di fine tirocinio, quale è indubbiamente l\'attestato prodotto dall\'istante, tuttavia, in concreto, dal rapporto finale d\'attività, sottoscritto dall\'istante medesimo, si evince che questi era stato assunto per un periodo di inserimento, quindi per un periodo di formazione (doc. 1).\r\nAlla luce di quanto sopra esposto il ricorso, che non ha evidenziato nessun titolo di cassazione, tantomeno quello di cui all\'art. 327 lett. g CPC, deve essere respinto.\r\n3.\r\nL\'art. 417 cpv. 1 lett. e CPC prevede la gratuità della procedura nelle azioni derivanti da contratto di lavoro. Non si prelevano quindi tasse né spese, mentre alla convenuta, che ha presentato osservazioni per il tramite di un avvocato, deve essere riconosciuta un\'indennità per ripetibili."
    elif task == 'summ':
        pre_text = "\n\nHier ein Beispiel für ein Gerichtsurteil mit Regeste:"
        if lang == 'de':
            example_input = "Sachverhalt\r\nBGE 134 IV 57 S. 57\r\nA.\r\nDie Präsidentin des Bezirksgerichts Bremgarten sprach X. am 22. Januar 2007 des mehrfachen fahrlässigen Beschäftigens von Ausländern ohne Arbeitsbewilligung im Sinne von Art. 23 Abs. 4 Satz 2 i.V.m.\r\nArt. 3 Abs. 3 ANAG\r\nschuldig und verurteilte ihn zu einer Busse von Fr. 2\'000.- respektive bei schuldhafter Nichtbezahlung zu einer Ersatzfreiheitsstrafe von 40 Tagen.\r\nAuf Berufung hin bestätigte das Obergericht des Kantons Aargau am 25. Juli 2007 das erstinstanzliche Urteil im Schuldpunkt, reduzierte\r\nBGE 134 IV 57 S. 58\r\ndie ausgefällte Busse aber auf Fr. 1\'000.- bzw. die Ersatzfreiheitsstrafe auf 10 Tage.\r\nX. wird zur Last gelegt, vom 27. Januar 2005 bis 3. Juli 2006 bzw. vom 12. Juni 2006 bis 31. August 2006 bei der A. AG zwei Handwerker, einen Plattenleger und einen Steinmetz, deutscher Nationalität beschäftigt zu haben, die nicht über die erforderlichen Aufenthalts- bzw. Arbeitsbewilligungen verfügten.\r\nB.\r\nX. führt Beschwerde in Strafsachen an das Bundesgericht mit dem Antrag auf Freisprechung von Schuld und Strafe.\r\nDas Bundesgericht weist die Beschwerde ab.\r\nErwägungen\r\nAus den Erwägungen:\r\n4.\r\nSeit dem 1. Juni 2002 gilt das Abkommen vom 21. Juni 1999 zwischen der Schweizerischen Eidgenossenschaft einerseits und der Europäischen Gemeinschaft und ihren Mitgliedstaaten andererseits über die Freizügigkeit (Freizügigkeitsabkommen, FZA; SR 0.142.112.681). Bürgerinnen und Bürger der EU- und EFTA-Staaten haben danach das Recht, sich zur Aufnahme oder Ausübung einer Erwerbstätigkeit im gesamten Hoheitsgebiet der Schweiz frei zu bewegen und aufzuhalten. Gemäss Art. 2 Anhang I des Freizügigkeitsabkommens wird zum Nachweis des Rechts, sich im Hoheitsgebiet einer Vertragspartei aufzuhalten, eine Aufenthaltsbewilligung ausgestellt. Das Freizügigkeitsabkommen kennt dabei zwei Arten von Aufenthaltsbewilligungen: Bei Arbeitsverhältnissen mit einer Dauer von mehr als drei Monaten, aber weniger als einem Jahr werden Kurzaufenthaltsbewilligungen EG erteilt, bei unbefristeten Arbeitsverträgen oder solchen mit einer Dauer von mindestens einem Jahr Daueraufenthaltsbewilligungen EG mit einer Gültigkeit von fünf Jahren (Art. 6 Anhang I des Freizügigkeitsabkommens,\r\nArt. 4 der Verordnung vom 22. Mai 2002 über die Einführung des freien Personenverkehrs [VEP; SR 142.203]\r\n).\r\nDiese in Anwendung des Freizügigkeitsabkommens ausgestellten Bewilligungen haben nach der Rechtsprechung des Gerichtshofs der Europäischen Gemeinschaften (EuGH) nicht rechtsbegründenden Charakter, sondern bloss deklarative Bedeutung (Urteile vom 5. Februar 1991 in der Rechtssache C-363/89,\r\nRoux, Slg. 1991, I-273, Randnr. 12 sowie vom 25. Juli 2002 in der Rechtssache C-459/99, Mouvement contre le racisme, l\'antisémitisme et la xénophobie [MRAX], Slg. 2002, I-6591, Randnr. 74). Das bedeutet, dass der\r\nBGE 134 IV 57 S. 59 Aufenthalt bzw. die Ausübung einer Erwerbstätigkeit auch bei fehlender Bewilligung nicht rechtswidrig ist mit der Folge, dass der Arbeitgeber, welcher EU- oder EFTA-Staatsangehörige in der Schweiz ohne Aufenthalts- bzw. Arbeitserlaubnis beschäftigte, nicht nach Art. 23 Abs. 4 ANAG strafbar wäre. Allerdings ist vor dem Hintergrund der etappenweisen Einführung der vollen Personenfreizügigkeit zu beachten, dass die Erteilung von Aufenthaltsbewilligungen - mit Ausnahme solcher für Arbeitseinsätze von weniger als vier Monaten - für Erwerbstätige aus den alten EU-Mitgliedstaaten sowie Zypern und Malta während der ersten fünf Jahre, also bis Ende Mai 2007, kontingentiert war (Art. 10 des Freizügigkeitsabkommens; vgl. Art. 2 des Protokolls vom 26. Oktober 2004 über die Ausdehnung des Freizügigkeitsabkommens auf die neuen EU-Mitgliedstaaten [AS 2006 S. 995]). Soweit und solange die Zulassung zur Ausübung einer Erwerbstätigkeit der Kontingentierung - einer arbeitsmarktlichen Beschränkung im Sinne von Art. 10 des Freizügigkeitsabkommens - untersteht, ist für den Stellenantritt übergangsrechtlich doch noch eine Aufenthaltsbewilligung erforderlich (Art. 26 Abs. 2 Anhang I des Freizügigkeitsabkommens). Die Arbeitsstelle darf und kann somit während des Übergangsregimes legal erst angetreten werden, wenn die entsprechende Bewilligung, welche gemäss Art. 6 Abs. 7 Anhang I des Freizügigkeitsabkommens allerdings ohne Aufschub zu erteilen ist, vorliegt. Wird sie nicht eingeholt, kann deshalb der Straftatbestand der Beschäftigung ohne Bewilligung nach Art. 23 Abs. 4 ANAG erfüllt sein.\r\nIm vorliegenden Fall geht es um unbefristete Arbeitsverhältnisse, die der Kontingentierung unterstanden. Für den rechtmässigen Stellenantritt wären daher nach dem Gesagten Aufenthaltsbewilligungen erforderlich gewesen. Dass im Zeitpunkt des Vertragsabschlusses mit der ausländischen Arbeitskraft eine gültige Zusicherung der Aufenthaltsbewilligung EG/EFTA vorgelegen hat, ändert daran nichts. Denn eine solche Zusicherung stellt die Erteilung der Aufenthaltsbewilligung zur Ausübung der Erwerbstätigkeit lediglich in Aussicht (Art. 8 VEP), berechtigt die ausländische Person aber nicht per se zum Stellen- bzw. Arbeitsantritt. Der diesbezügliche Einwand des Beschwerdeführers erweist sich deshalb als unbehelflich. "
            example_output = "Regeste\r\nRechtswidriges Beschäftigen von ausländischen Arbeitskräften (Art. 23 Abs. 4 und Art. 3 Abs. 3 ANAG).\r\nAufenthaltsbewilligungen, die in Anwendung des Freizügigkeitsabkommens ausgestellt werden, haben grundsätzlich deklaratorischen Charakter. Solange allerdings die Zulassung zur Ausübung einer Erwerbstätigkeit der Kontingentierung untersteht, bleibt der Stellenantritt ohne Aufenthaltsbewilligung unrechtmässig (E. 4)."
        elif lang == 'fr':
            example_input = "Sachverhalt\r\nab Seite 88\r\nBGE 135 III 88 S. 88\r\nA.\r\nLe 10 août 1994, le Tribunal de première instance de Munich (Allemagne) a condamné X. à verser à Y., son ex-épouse, la somme de 1\'645 euros à titre de pension alimentaire.\r\nLe 1er octobre 2007, Y. a requis la poursuite de son ex-époux pour un montant de 35\'523 fr. 20, plus intérêts à 5 % dès le 15 mars 2007, terme moyen. Selon le taux de change retenu par la créancière (à savoir 1 euro = 1,6611 fr.), la pension mensuelle, d\'un montant de 1\'645 euros, correspond à la somme de 2\'732 fr. 50. X. a formé opposition au commandement de payer qui lui était notifié.\r\nB.\r\nLe 20 décembre 2007, Y. a requis du Tribunal de première instance du canton de Genève la reconnaissance et l\'exécution du jugement du Tribunal de première instance de Munich, ainsi que la mainlevée définitive de l\'opposition formée par son ex-mari au commandement de payer.\r\nBGE 135 III 88 S. 89 Par jugement du 4 avril 2008, le Tribunal de première instance a notamment reconnu et déclaré exécutoire en Suisse le jugement allemand (ch. 1) et prononcé la mainlevée définitive de l\'opposition faite au commandement de payer - sans toutefois préciser à concurrence de quel montant - (ch. 2). Statuant sur appel de X. le 19 juin 2008, la Cour de justice a, entre autres, réformé le ch. 2 en prononçant la mainlevée à concurrence de 35\'285 fr. 25 avec intérêt à 5 % l\'an dès le 15 mars 2007.\r\nC.\r\nX. dépose un recours en matière civile contre cette dernière décision, concluant au rejet de la requête de mainlevée définitive. Le recours a été rejeté par arrêt du 21 novembre 2008.\r\nErwägungen\r\nExtrait des considérants:\r\n4.\r\nLe recourant soutient qu\'en jugeant que l\'intimée ne devait pas prouver par pièce le taux de change entre l\'euro et le franc suisse, la cour cantonale aurait violé l\'\r\nart. 80 al. 1 LP.\r\n4.1\r\nA teneur de l\'art. 67 al. 1 ch. 3 LP, la réquisition de poursuite adressée à l\'Office énonce le montant de la créance en valeur légale suisse. La conversion en valeur légale suisse d\'une créance stipulée en monnaie étrangère est une règle d\'ordre public et une exigence de la pratique. En imposant cette conversion, le législateur n\'a cependant pas entendu modifier le rapport de droit liant les parties et nover en une dette de francs suisses celle que les intéressés ont librement fixée en devises étrangères (ATF 134 III 151 consid. 2.3 et les références citées; ROLAND RUEDIN, in Commentaire romand, Poursuite et faillite, 2005, no s 27 s. Ad art. 67 LP). La conversion se fait au cours de l\'offre des devises du jour de la réquisition de poursuite (ATF 51 III 180 consid. 4; BlSchK 1997 p. 62 consid. 5e; RUEDIN, op. cit., n os 29 s. Ad art. 67 LP).\r\nSelon la jurisprudence, les faits notoires, qu\'il n\'est pas nécessaire d\'alléguer ni de prouver (ATF 130 III 113\r\nconsid. 3.4 et les arrêts cités), sont ceux dont l\'existence est certaine au point d\'emporter la conviction du juge, qu\'il s\'agisse de faits connus de manière générale du public (allgemeine notorische Tatsachen) ou seulement du juge (amtskundige oder gerichtskundige Tatsachen; VOGEL/SPÜHLER, Grundriss des Zivilprozessrechts, 8 e éd. 2006, p. 255 n. 17; FABIENNE HOHL, Procédure civile, tome I, 2001, n. 945). La jurisprudence précise que, pour être notoire, un renseignement ne doit pas être constamment BGE 135 III 88 S. 90 présent à l\'esprit, il suffit qu\'il puisse être contrôlé par des publications accessibles à chacun (arrêt du Tribunal fédéral 4P.277/1998 du 22 février 1999 consid. 3d, in RSDIE 2000 p. 575).\r\nDe nos jours, le taux de conversion des monnaies est un fait notoire, qui ne doit être ni allégué ni prouvé. Il peut en effet être contrôlé sur internet, par des publications officielles et dans la presse écrite; il est donc accessible à chacun (cf. arrêt du Tribunal fédéral 5P.236/1988 du 8 novembre 1988 consid. 1b, in SJ 1989 p. 205; arrêt du Tribunal fédéral 4P.277/1998 du 22 février 1999 consid. 3d, in RSDIE 2000 p. 575; également PIERRE-ROBERT GILLIÉRON, Commentaire de la loi fédérale sur la poursuite pour dettes et la faillite, art. 1-88 LP, 1999, no 63 ad art. 80 LP). L\'internet permet en outre d\'accéder rapidement au taux de conversion en vigueur à une date donnée - par exemple la date de la réquisition de poursuite -; il n\'est donc pas nécessaire d\'obtenir une confirmation bancaire ou une copie de la presse parue à la date recherchée. Il suffit ainsi de quelques minutes pour déterminer qu\'au 1er octobre 2007, le cours de l\'euro en francs suisses était de 1,6603 et effectuer ensuite la conversion des 1\'645 euros en francs suisses (http://www.fxtop.com donne les taux officiels diffusés par la Banque centrale européenne).\r\n4.2\r\nC\'est par conséquent à tort que le recourant soutient que le taux de conversion doit être prouvé par pièces et qu\'il y aurait donc violation de l\'art. 80 al. 1 LP pour ce motif.\r\nLa cour cantonale a fixé le taux de conversion à 1,65 fr., soit à un taux inférieur au taux réel notoire de 1,6603 fr. La poursuivante n\'ayant cependant pas recouru contre l\'arrêt cantonal, il n\'y a pas lieu de réformer cette décision en sa faveur.\r\nIl est superflu d\'examiner les griefs formulés par le recourant à l\'encontre de la \"valeur approximative\" retenue par la Cour de justice. "
            example_output = "Regeste\r\nArt. 67 Abs. 1 Ziff. 3 SchKG\r\n; Umrechnungskurs in gesetzliche Schweizerwährung für eine in Euro festgelegte Forderung.\r\nDer Umrechnungskurs des Euro ist eine notorische Tatsache, die vom Betreibungsgläubiger weder behauptet noch bewiesen werden muss (E. 4)."
        elif lang == 'it':
            example_input = "Erwägungen\r\nab Seite 201\r\nBGE 141 IV 201 S. 201\r\nDai considerandi:\r\n8.\r\n8.2.1\r\nÈ stato accertato, senza arbitrio, che la ricorrente ha più volte chiesto a F. di trovare, nel senso di contattare e ingaggiare (avendo precisato che aveva i soldi per pagare), qualcuno che potesse uccidere il marito e che egli rifiutò di fare quello che gli si domandava.\r\n8.2.2\r\nContrariamente a quanto sostenuto nel gravame, la contestata richiesta risulta tutt\'altro che generica: permetteva di ben comprendere sia il genere di infrazione finale prospettata (reato contro la vita) sia la vittima designata sia il comportamento da assumere, ossia reperire e ingaggiare qualcuno allo scopo, atteso che vi era a disposizione denaro. F. non si è risolto a commettere alcunché, motivo per cui si è di fronte solo a un tentativo di istigazione e la questione del nesso causale tra l\'atto di persuasione e la decisione dell\'istigato di commettere il reato neppure si pone. Infatti, il nesso di causalità è necessario esclusivamente in presenza di un\'istigazione consumata (v. BERNHARD STRÄULI, in Commentaire romand, Code pénal, vol. I,BGE 141 IV 201 S. 202 2009, n. 20 ad art. 24 CP). Non è peraltro contestato che l\'istigazione aveva quale scopo ultimo la commissione di un assassinio, ossia di un crimine.\r\nIn quanto la ricorrente non ha chiesto a F. di provvedere a uccidere il marito, ma \"solo\" di trovare qualcuno che lo facesse, si è in presenza di un tentativo di istigazione indiretta. In passato il Tribunale federale ha lasciato irrisolta la problematica della punibilità del tentativo d\'istigazione di un altro all\'istigazione di una terza persona (sentenza 6S.448/2004 del 3 ottobre 2005 consid. 4.3). Sull\'argomento la dottrina è divisa. Per una parte, il tentativo di istigazione di secondo grado non sarebbe punibile. Poiché, rispetto all\'istigatore diretto e al potenziale autore, l\'istigatore indiretto è più distante dal risultato del reato prospettato (e quindi da una lesione o da un\'esposizione a pericolo del bene giuridico tutelato), non dovrebbe essere trattato più severamente degli altri e pertanto non dovrebbe essere punito se l\'istigatore diretto nulla ha intrapreso per istigare il potenziale autore del crimine (MICHA NYDEGGER, Zurechnungsfragen der Anstiftung im System strafbarer Beteiligung, 2012, pag. 169; GÜNTER STRATENWERTH, Allgemeiner Teil I: Die Straftat, 4a ed. 2011, § 13 n. 132; DONATSCH/TAG, Verbrechenslehre, 9a ed. 2013, pag. 162). Per un\'altra corrente della dottrina invece, atteso che l\'art. 24 cpv. 2 CP deroga al principio dell\'accessorietà reale, il tentativo di istigazione di secondo grado sarebbe punibile anche se l\'istigatore diretto neppure ha cominciato a persuadere il potenziale autore del crimine (STRÄULI, op. cit., n. 54 ad art. 24 CP; PHILIPPE GRAVEN, L\'infraction pénale punissable, 2\r\nA ed. 1995, pag. 304). Quest\'ultima opinione merita assenso. Entrambi i capoversi dell\'art. 24 CP sono strutturati in modo identico, per cui, come l\'art. 24 cpv. 1 CP non esclude l\'istigazione (consumata) di secondo grado, non esigendo di determinare direttamente altri a commettere un crimine o un delitto (v. DTF 73 IV 216\r\nconsid. 2a), così anche l\' art. 24 cpv. 2 CP non estromette dal suo campo di applicazione la tentata istigazione di secondo grado, non essendo limitato al tentativo di determinare direttamente altri a commettere un crimine. L\'autore di un tentativo di istigazione, anche se indiretta, ha la volontà che il crimine sia commesso: la sua intenzionalità non si riferisce a una semplice minaccia verso un bene giuridico, ma alla sua lesione (v. sentenza Str.84/1983 del 7 settembre 1983 consid. 2a, in SJ 1984 pag. 160). Certo l\'esposizione a pericolo del bene giuridico tutelato dal diritto penale è minore in caso di tentativo di istigazione indiretta rispetto al tentativo di istigazione diretta; l\'art. 24 cpv. 2 CP non fissa tuttavia una soglia di pericolo a BGE 141 IV 201 S. 203\r\npartire dalla quale sanzionare la tentata istigazione. È piuttosto nell\'ambito della commisurazione della pena che occorre considerare la gravità reale del tentativo di istigazione, le conseguenze concrete dell\'atto commesso e la prossimità del risultato (v. sentenza 6S.44/2007 del 6 giugno 2007 consid. 4.5.5). Nella fattispecie la Corte cantonale ha effettivamente considerato tali aspetti al momento di commisurare la pena.\r\nSicché su questo punto la condanna della ricorrente non viola l\' art. 24 cpv. 2 CP ed è conforme al diritto federale. "
            example_output = "Regeste\r\nArt. 24 Abs. 2 StGB; indirekte Anstiftung (Kettenanstiftung), Versuch.\r\nAuch die versuchte indirekte Anstiftung (Kettenanstiftung) zu einem Verbrechen ist strafbar (E. 8.2.2)."
    else:
        raise ValueError(f"Task {task} not supported")

    return f"{pre_text}{get_short_prompt(example_input, task, lang)}{example_output}"

def get_short_prompt(input, task, lang):
    if task == 'cvg':
        if lang == 'de':
            return f'\n\n[Sachverhalt des Schweizer Gerichtsurteils]:\n{input}\n\n[Erwägungen]:\n'
        elif lang == 'fr':
            return f"\n\n[Faits du jugement suisse]:\n{input}\n\n[Considérations]:\n"
        elif lang == 'it':
            return f"\n\n[Fatti della sentenza svizzera]:\n{input}\n\n[Considerazioni]:\n"
    elif task == 'summ':
        return f"\n\n[Gegebener Sachverhalt, Erwägungen und Dispositiv]:\n{input}\n\n[Regeste auf Deutsch]:\n"


def get_instruction(input, lang, task, shot):
    """
    Creates the instruction for the API completion
    """
    situation_instruction = {
        "cvg": {
            # 457 tokens
            "de": 'Ziel: Generiere Erwägungen basierend auf dem gegebenen Sachverhalt eines Schweizer Gerichtsurteils.\nHintergrund: Ein Schweizer Gerichtsurteil besteht aus Rubrum, Sachverhalt, Erwägungen, Dispositiv (Urteilsformel) und Unterschrift. Die Erwägungen sind die rechtliche Würdigung des Geschehens durch das Gericht.\nAnweisung:\n-Sachverhalt Verstehen: Der gegebene Sachverhalt enthält bestrittene und unbestrittene Fakten, die Begehren der Parteien, das Beweisverfahren und die Prozessgeschichte.\n-Beginne mit Prozessvoraussetzungen: Prüfe zunächst, ob die Prozessvoraussetzungen (z.B. Zuständigkeit des Gerichts) erfüllt sind. Wenn nicht strittig, reicht es aus zu bestätigen, dass die Voraussetzungen erfüllt sind.\n-Rechtliche Würdigung:\nEruiere relevante Rechtssätze basierend auf den behaupteten und rechtlich relevanten Tatsachen.\n-Setze dich mit den rechtlichen Standpunkten der Parteien auseinander.\n-Beachte die Beweislastverteilung und würdige die Beweise frei, aber berücksichtige relevante gesetzliche Beweisregeln.\n-Iura novit curia: Deine rechtliche Würdigung muss nicht zwangsläufig dem rechtlichen Vorbringen der Parteien entsprechen. Berücksichtige andere mögliche Argumentationslinien.\n-Zusammenfassung: Fasse am Ende deine Erwägungen, das Ergebnis Ihrer rechtlichen Würdigung, zusammen.\n-Output: Der generierte Text sollte strukturiert, klar und in der Form von typischen Erwägungen eines Schweizer Gerichtsurteils sein.',
            # 415 tokens
            "fr": "But: Génère des considérations basées sur les faits donnés d'un jugement suisse.\nContexte: Un jugement suisse est composé du rubrum, des faits, des considérations, du dispositif (formule du jugement) et de la signature. Les considérations sont l'appréciation juridique des événements par le tribunal.\nInstructions:\n- Comprends les faits: Les faits donnés contiennent des faits contestés et non contestés, les demandes des parties, la procédure de preuve et l'historique du procès.\n- Commence par les conditions de procédure: Vérifie d'abord si les conditions de procédure (par exemple, la compétence du tribunal) sont remplies. Si cela n'est pas contesté, il suffit de confirmer que les conditions sont remplies.\n- Appréciation juridique:\nÉvalue les dispositions juridiques pertinentes basées sur les faits allégués et juridiquement pertinents.\n- Confronte-toi aux points de vue juridiques des parties.\n- Tiens compte de la répartition de la charge de la preuve et évalue les preuves librement, mais tiens compte des règles légales de preuve pertinentes.\n- Iura novit curia: Ton appréciation juridique ne doit pas nécessairement correspondre aux arguments juridiques présentés par les parties. Considère d'autres lignes d'argumentation possibles.\n- Résumé: Résume à la fin de tes considérations le résultat de ton appréciation juridique.\n- Résultat: Le texte généré devrait être structuré, clair et sous la forme de considérations typiques d'un jugement suisse.",
            # 406 tokens
            "it": "Obiettivo: Genera considerazioni basate sui fatti presentati in una sentenza svizzera.\nContesto: Una sentenza svizzera si compone di rubrum, fatti, considerazioni, dispositivo (formula della sentenza) e firma. Le considerazioni rappresentano la valutazione giuridica degli eventi da parte del tribunale.\nIstruzioni:\n- Comprendi i fatti: I fatti presentati includono fatti contestati e non contestati, le richieste delle parti, la procedura probatoria e la storia del processo.\n- Inizia con le condizioni processuali: Verifica prima di tutto se le condizioni processuali (ad es. la competenza del tribunale) sono soddisfatte. Se non contestate, basta confermare che le condizioni sono state soddisfatte.\n- Valutazione giuridica:\nValuta le norme giuridiche rilevanti in base ai fatti affermati e giuridicamente rilevanti.\n- Confrontati con i punti di vista giuridici delle parti.\n- Tieni conto della distribuzione dell'onere della prova e valuta le prove liberamente, ma considera le regole di prova legalmente rilevanti.\n- Iura novit curia: La tua valutazione giuridica non deve necessariamente corrispondere alle argomentazioni giuridiche delle parti. Considera altre possibili linee di argomentazione.\n- Riassunto: Riassumi alla fine delle tue considerazioni il risultato della tua valutazione giuridica.\n- Risultato: Il testo generato dovrebbe essere strutturato, chiaro e nella forma di considerazioni tipiche di una sentenza svizzera."
            },
        "summ": {
            # 342 tokens
            "de": "Ziel: Generiere eine Regeste basierend auf einem Schweizer Gerichtsurteils.\nHintergrund: Ein Schweizer Gerichtsurteil setzt sich aus Sachverhalt, Erwägungen und Dispositiv zusammen. Die Regeste dient als Kurzzusammenfassung und beinhaltet Leitsätze des Urteils. Nur Leitentscheide haben eine Regeste.\nAnweisung:\n1. Sachverhalt: Lies und verstehe den gegebenen Sachverhalt.\n2. Erwägungen: Analysiere die Erwägungen, um die Hauptargumente und Gründe zu identifizieren.\n3. Dispositiv: Beachte das Dispositiv, da es das endgültige Urteil enthält.\n4. Erstelle die Regeste: Die Regeste sollte aus drei sehr kurzen Teilen bestehen: a. Zitiere die wichtigsten relevanten Artikelziffern (ohne den Artikeltitel). b. Nenne kurze, relevante, deskriptive Keywords, über die Thematik des Falls. c. Formuliere einen sehr kurzen Fliesstext, der die wichtigsten Erwägungen zitiert und kurz zusammenfasst.\nOutput: Die Regeste sollte eine klare und strukturierte Kurzzusammenfassung des Urteils bieten, die aus zitierten Artikeln, Keywords und einem sehr kurzen Fliesstext besteht.",
            "fr": "Ziel: Generiere eine Regeste basierend auf einem Schweizer Gerichtsurteils.\nHintergrund: Ein Schweizer Gerichtsurteil setzt sich aus Sachverhalt, Erwägungen und Dispositiv zusammen. Die Regeste dient als Kurzzusammenfassung und beinhaltet Leitsätze des Urteils. Nur Leitentscheide haben eine Regeste.\nAnweisung:\n1. Sachverhalt: Lies und verstehe den gegebenen Sachverhalt.\n2. Erwägungen: Analysiere die Erwägungen, um die Hauptargumente und Gründe zu identifizieren.\n3. Dispositiv: Beachte das Dispositiv, da es das endgültige Urteil enthält.\n4. Erstelle die Regeste: Die Regeste sollte aus drei sehr kurzen Teilen bestehen: a. Zitiere die wichtigsten relevanten Artikelziffern (ohne den Artikeltitel). b. Nenne kurze, relevante, deskriptive Keywords, über die Thematik des Falls. c. Formuliere einen sehr kurzen Fliesstext, der die wichtigsten Erwägungen zitiert und kurz zusammenfasst.\nOutput: Die Regeste sollte eine klare und strukturierte Kurzzusammenfassung des Urteils bieten, die aus zitierten Artikeln, Keywords und einem sehr kurzen Fliesstext besteht.",
            "it": "Ziel: Generiere eine Regeste basierend auf einem Schweizer Gerichtsurteils.\nHintergrund: Ein Schweizer Gerichtsurteil setzt sich aus Sachverhalt, Erwägungen und Dispositiv zusammen. Die Regeste dient als Kurzzusammenfassung und beinhaltet Leitsätze des Urteils. Nur Leitentscheide haben eine Regeste.\nAnweisung:\n1. Sachverhalt: Lies und verstehe den gegebenen Sachverhalt.\n2. Erwägungen: Analysiere die Erwägungen, um die Hauptargumente und Gründe zu identifizieren.\n3. Dispositiv: Beachte das Dispositiv, da es das endgültige Urteil enthält.\n4. Erstelle die Regeste: Die Regeste sollte aus drei sehr kurzen Teilen bestehen: a. Zitiere die wichtigsten relevanten Artikelziffern (ohne den Artikeltitel). b. Nenne kurze, relevante, deskriptive Keywords, über die Thematik des Falls. c. Formuliere einen sehr kurzen Fliesstext, der die wichtigsten Erwägungen zitiert und kurz zusammenfasst.\nOutput: Die Regeste sollte eine klare und strukturierte Kurzzusammenfassung des Urteils bieten, die aus zitierten Artikeln, Keywords und einem sehr kurzen Fliesstext besteht.",
            }
        }
    if shot == 0:
        instruction = f"{situation_instruction[task][lang]}{get_short_prompt(input, task, lang)}"
    elif shot == 1:
        instruction = f"{situation_instruction[task][lang]}{get_one_shot_example(task, lang)}{get_short_prompt(input, task, lang)}"
    else:
        raise ValueError("shot must be 0 or 1")
    print(f"<Instruction>{instruction}<Instruction/>")
    return instruction

def generate_completions(dataset, max_input_length, max_output_length, model, api, shot):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    dataset_with_predicted = []
    logger.info("Using model: " + model)
    for entry in dataset:
        entry_with_predicted = generate(api, entry, max_output_length, model, shot)
        dataset_with_predicted.append(entry_with_predicted)
    return dataset_with_predicted

@retry(stop_max_attempt_number=10, wait_fixed=30000)  # Retries 20 times with a 10-second delay between retries
def generate(api, entry, max_output_length, model, shot):
    instruct = get_instruction(entry["input"], entry["lang"], task_name, shot)
    # measure time to generate completion
    start = time.time()
    if api == "openai":
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "user", "content": instruct}
                ],
            max_tokens=max_output_length,
            )
        predicted = completion.choices[0].message["content"]

    elif api == "claude":
        anthropic = Anthropic()
        completion = anthropic.completions.create(
            model=model_name,
            prompt=f"{HUMAN_PROMPT} {instruct}{AI_PROMPT}",
            max_tokens_to_sample=max_output_length,
            )
        predicted = completion.completion
    elif api == "deepinfra":
        completion = deepinfra_create(
            model_name=model_name,
            prompt=f"<<SYS>>\n{instruct}\n<</SYS>>\n\n",
            max_tokens=max_output_length,
            )
        predicted = completion
    else:
        raise NotImplementedError("API not implemented yet")
    end = time.time()
    completion_time = end - start
    # if model is gpt-4: wait, so that we don't exceed 10k tokens per minute: calculate using input length, output length and copmletion time
    if model == "gpt-4":
        actual_token_length = number_of_tokens(entry["input"], model_name) + number_of_tokens(predicted, model_name)
        # to not exceed rate limit of 10k tokens per minute
        time_to_wait = actual_token_length / 10000 * 60 - completion_time
        logger.info("Time to wait: " + str(time_to_wait))
        # wait if time to wait is positive
        if time_to_wait > 0:
            time.sleep(time_to_wait)
    logger.info("Time to generate completion: " + str(completion_time))
    print("predicted: ", predicted)
    entry_with_predicted = {
        "input": entry["input"], "target": entry["target"], "predicted": predicted, "lang": entry["lang"]}
    return entry_with_predicted


########################################################################################################################

# measure time for whole script
start = time.time()

logger = setup_logger()

parser = argparse.ArgumentParser()
parser.add_argument("--finetune", help="Want to finetune model?")
parser.add_argument("--model", help="Model name for finetune / evaluation (depends on finetune flag")
parser.add_argument("--api", help="Which api to generate completions: openai, claude, deepinfra, vertex")
parser.add_argument("--train_size", help="Size of training set", type=int)
parser.add_argument("--eval_size", help="Size of evaluation set", type=int)
parser.add_argument("--test_size", help="Size of test set", type=int)
parser.add_argument("--input_length", help="Input sequence length for training, evaluation and generation", type=int)
parser.add_argument("--output_length", help="Output sequence length for training, evaluation and generation", type=int)
parser.add_argument("--epochs", help="Number of training epochs", type=int)
parser.add_argument("--total_batch_size", help="The total batch size to use", type=int)
parser.add_argument("--gm", help="GPU memory size for batch size", type=int)
parser.add_argument("--origin", help="Use dataset with origin cases")
parser.add_argument("--sum", help="Loads summarization dataset if True")
parser.add_argument("--shot", help="n-shot task (either 0 or 1)", type=int)
args = parser.parse_args()

model_name = args.model
api_name = args.api
shot = args.shot

print(f"This is a {shot}-shot task")

if args.origin == "True" and args.sum == "True":
    raise ValueError("Cannot use both origin and sum flags (as True)")

# print all args
logger.info(args)

eval_dataset = get_val_dataset(logger, args.sum, args.origin)

# Update args values with the full lengths of the dataset splits if the args values are -1
if args.eval_size == -1:
    args.eval_size = len(eval_dataset)

# Select subsets of the dataset based on the updated args values
seed = 42
eval_dataset = eval_dataset.shuffle(seed).select(range(args.eval_size))

project_name = "summarization" if args.sum == "True" else "court view generation"

if args.sum == "True":
    task_name = "summ"
elif args.origin == "True":
    task_name = "cvg-origin"
else:
    task_name = "cvg"


logger.info("Project name: " + project_name)
os.environ["WANDB_PROJECT"] = project_name
os.environ["WANDB_RUN_GROUP"] = f"{model_name}, {len(eval_dataset)}"

# add train size, seq length to output dir
output_dir = f"output/{task_name}/{args.model.split('/')[-1]}_evalsize={args.eval_size}_inlen={args.input_length}_outlen={args.output_length}_origin={args.origin}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}"
# set wandb run name
wandb.init(name=output_dir.split('/')[-1]) # means
# log output dir to wandb
wandb.log({"output_dir": output_dir})
# log task name to wandb
wandb.log({"task_name": task_name})

logger.info("Model name:" + model_name + " finetune: " + " output_dir: " + output_dir)
logger.info("Eval dataset size: " + str(len(eval_dataset)) )

# log all args to wandb
wandb.config.update(args)

output_examples = []

# prepare dataset for generation
eval_data = prepare_dataset(eval_dataset, args.input_length, args.output_length, args.sum, args.origin, model_name)

# generate output examples using API completions
completion_dataset = generate_completions(eval_data, args.input_length, args.output_length, model_name, api_name, shot)

# Evaluate model on test dataset
meteor_score_avg, rouge_score_avg, bleu_score_avg, bert_score_avg = compute_scores(completion_dataset)

# Print and log scores to wandb
log_test_scores(meteor_score_avg, rouge_score_avg, bleu_score_avg, bert_score_avg)

try:
    # save output examples to file
    export_output(output_examples, output_dir, task_name)
except Exception as e:
    logger.info("Error exporting output examples: " + str(e))

# measure time for whole script
end = time.time()

# wandb log time per item
wandb.log({"time_per_item": (end - start) / len(eval_dataset)})
# wandb log shot
wandb.log({"shot": shot})

# in readable format
logger.info("Time for whole script: " + str(datetime.timedelta(seconds=end - start)))
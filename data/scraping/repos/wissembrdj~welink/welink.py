from main.input_preprocessing.input_treatement import Input_treatement
from main.candidate_generation.string_match import StringMatch
from main.models.similarities import Similarities 
from main.candidate_ranking.candidate_similarities import Candidate_similarities
from main.candidate_ranking.coherence import Coherence
from main.candidate_ranking.edit_distance import Edit_distance
import json
import time


class Welink():
    
    def __init__(self, query):
        self.query=query   
    
    def userquerymodel(self):
        start_time = time.time()
        input_query=Input_treatement(self.query)
        treated_query_p=input_query.input_treatement()
        treated_query=treated_query_p.mentions_list
        deteted_entities=treated_query_p.detected_ne
        print(deteted_entities)
        
        nbr_ne=len(deteted_entities)

        print('pre traitement de la requete', time.time() - start_time)
        entities_sim=[]
        named_e=[]
        ne_nb=len(treated_query)
        results_disambiguation=[]
        entities_mentions=[]
        r=0
        if treated_query:
            for ne in treated_query: 
                named_entity=ne.name 
                mention=len(named_entity)/10
                named_entity_type=""
                best_candidate=[]
                entities_sim=[]
                candidates=StringMatch(ne)
                candidates_ec=candidates.string_match()
                print('string match:', time.time() - start_time)  
                if candidates_ec:                  
                    sim_c=Candidate_similarities(ne, candidates_ec)
                    cosine_sim_context=sim_c.cosine()
                    a=1
                    total_sim=0
                    for i in candidates_ec:
                        candidate=i.ec_name
                        candidates_relations= i.properties + i.objects
                        sim_proprieties= Coherence(ne.sentence_words, candidates_relations)
                        jaccard_sim_properties= sim_proprieties.jaccard_variation()
                        edit_distance_name=Edit_distance(named_entity,candidate)
                        edit_distance=edit_distance_name.minimumEditDistance()   
                        max_name_len=edit_distance_name.compare_strings_len()
                        if edit_distance!=0:  
                            edit_distance=(edit_distance/max_name_len)
                        
    
                        if ne_nb>1 :
                            sim_entities= Coherence(ne.sentence_words,i.objects)
                            jaccard_sim_entities= sim_entities.jaccard_variation()
                        else:
                            jaccard_sim_entities=0 

                        total_sim= cosine_sim_context[a] + jaccard_sim_properties + jaccard_sim_entities - edit_distance
                        if named_entity in deteted_entities:
                            total_sim=total_sim+0.5

                        total_sim=total_sim+mention
                        
                        similarities_e=Similarities(ne, i, cosine_sim_context[a], jaccard_sim_properties, jaccard_sim_entities,total_sim)
                        entities_sim.append(similarities_e)
                        a=a+1  
                        
           
                    print('candidate ranking:', time.time() - start_time)       
                    best_candidate = sorted(entities_sim, key=lambda x: x.total_sim, reverse=True)
                    print('tri', time.time()- start_time)
                    results_disambiguation.append(best_candidate)
                    print("--- %s seconds ---" % (time.time() - start_time))

        
        final = sorted(results_disambiguation, key=lambda x: x[0].total_sim, reverse=True)            
      
        threshold=nbr_ne
        n=0
        filtred_entities_sim=[]
        if final:
            if len(final)==1:
                if final[0][0].total_sim>0.6:    
                    filtred_entities_sim=final
            else:
                if threshold>0:
                    if len(final)<threshold:
                        for rd in final:
                            if rd[0].total_sim>0.6:
                                filtred_entities_sim.append(rd)
                    else:    
                        while n<threshold:    
                            if final[n][0].total_sim>0.6:
                                filtred_entities_sim.append(final[n])    
                            n=n+1 
                else:
                    if final[0][0].total_sim>0.6:
                        filtred_entities_sim.append(final[0])          
                        
                          
                             
        return filtred_entities_sim
        
        
    

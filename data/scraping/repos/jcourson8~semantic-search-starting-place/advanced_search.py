import sys
import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from supabase.client import Client
from langchain.vectorstores import SupabaseVectorStore
from langchain.docstore.document import Document

def similarity_search(
        query,
        client,
        embedding,
        table= "match_vectors",
        k = 6,
        threshold = 0.5,
    ):
        vectors = embedding.embed_documents([query])
        query_embedding = vectors[0]
        res = client.rpc(
            table,
            {
                "query_embedding": query_embedding,
                "match_count": k,
            },
        ).execute()

        match_result = [
            (
                Document(
                    metadata=search.get("metadata", {}),  # type: ignore
                    page_content=search.get("content", ""),
                ),
                search.get("similarity", 0.0),
            )
            for search in res.data
            if search.get("content")
        ]

        documents = [doc for doc, _ in match_result]

        return documents

def main():
    query = sys.argv[1]
    # Use the argument...

    load_dotenv()
    file_path = os.getenv("FILE_PATH")
    openai_key = os.getenv("OPENAI_KEY")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    supabase_client= create_client(supabase_url, supabase_key)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    
    matched_docs = similarity_search(query, supabase_client, embeddings)
    uuids = []
    # from matched_docs, get the uuid from metadata and add it to list of uuids
    for doc in matched_docs:
        uuids.append(doc.metadata["uuid"])

    print("UUIDs:" + str(uuids))

if __name__ == "__main__":
    main()


# Example document:
# [Document(page_content="title: Veteran–child communication about parental PTSD: A mixed methods pilot study , rs: ShowReport?id=f26d7e02-132c-4094-9a36-69a653a95c1a&rt=rs, apaCitation: Sherman, M. D., Larsen, J., Straits-Troster, K., Erbes, C., & Tassey, J. (2015). Veteran-child communication about parental PTSD: A mixed methods pilot study. Journal of Family Psychology, 29(4), 595-603. doi:10.1037/fam0000124, abstractReach: Many adults with posttraumatic stress disorder (PTSD) are parents who must navigate relationships with their children in the face of this diagnosis. This study investigated communication with children regarding parent's PTSD as well as parental experiences receiving treatment for PTSD in a Veteran sample. Findings indicated that Veterans desired to share information about their PTSD with their children, but that they experienced several barriers to doing so., doi: http://dx.doi.org/10.1037/fam0000124, focus: Children\nMental health\nParents\nVeterans\nYouth, subjectAffiliation: Veteran, population: Adolescence (13 - 17 yrs)\nAdulthood (18 yrs & older)\nAged (65 yrs & older)\nChildhood (birth - 12 yrs)\nMiddle age (40 - 64 yrs)\nPreschool age (2 -5 yrs)\nSchool age (6 - 12 yrs)\nThirties (30 - 39 yrs)\nYoung adulthood (18 - 29 yrs), methodology: Empirical Study\nInterview\nFocus Group\nQualitative Study\nQuantitative Study, authors: Sherman, Michelle D., Larsen, Jessica, Straits-Troster, Kristy, Erbes, Christopher, Tassey, John, paperAbstract: The majority of adults with posttraumatic stress disorder (PTSD) are parents. Parents with PTSD report lower levels of parenting satisfaction, poorer parent–child relationships, and elevated incidence of child distress and behavioral problems in comparison with parents without PTSD. Although literature exists regarding parent–child communication about serious mental illness and physical health problems, research has yet to examine this communication regarding parental PTSD. This 3-site, mixed methods study involved 19 veteran parents who had a diagnosis of PTSD; participants were recruited from VA medical centers. Veterans participated in focus groups or individual interviews and completed questionnaires, responding to questions about motivations and barriers for disclosure of their PTSD to their children, the content of such disclosure, experiences at the VA as a parent, and desired VA family resources. Although many veterans described a desire to talk with their children about PTSD, they experience many barriers to doing so, including both personal reservations and feelings (e.g., avoidance of discussing PTSD, shame) and concerns about the consequences of disclosure on their children (e.g., child distress, loss of child’s respect for veteran). Regarding veterans’ experience at the VA, 21% reported that none of their providers had assessed if they have children, and 21% experienced the VA system as not welcoming to them as parents, citing both logistical issues (e.g., lack of childcare) and provider neglect of parenting concerns. Veterans indicated they would like the VA to offer parenting classes, workshops for families, child care, and family therapy. (PsycINFO Database Record (c) 2016 APA, all rights reserved), publisher: American Psychological Association, publicationType: Article\nREACH Publication, authorAffiliation: Department of Family Social Science, University of Minnesota, MDS\nPsychiatry and Behavioral Sciences, University of Oklahoma Health Sciences Center, JL\nDepartment of Veterans Affairs, Phoenix VAMC, KST\nCenter for Chronic Disease Outcomes Research, Minneapolis VAMC, CE\nPsychiatry and Behavioral Sciences, University of Oklahoma Health Sciences Center, JT, keywords: mental disorders, military veterans, parent child communication, parental characteristics, parenting, posttraumatic stress disorder, reachPublicationType: Research Summary, sponsors: VA South Central (VISN 16), Mental Illness Research, Education and Clinical Center", metadata={'uuid': 'f26d7e02-132c-4094-9a36-69a653a95c1a'}),
import streamlit as st
import os
import pymilvus
import openai


with st.form("my_form"):
    st.write("Submit a support case")
    text_val = st.text_area("Describe your problem?")

    submitted = st.form_submit_button("Submit")
    if submitted:
        import os
        import pymilvus
        import openai

        org_id = 360033549136 # TODO Load from customer login data

        pymilvus.connections.connect(uri=os.environ["MILVUS_URL"], token=os.environ["MILVUS_TOKEN"])
        collection = pymilvus.Collection("zendesk")

        embedding = openai.Embedding.create(input=text_val, model="text-embedding-ada-002")['data'][0]['embedding']

        results = collection.search(data=[embedding], anns_field="vector", param={}, limit=1, output_fields=["_id", "subject"], expr=f'status == "new" and organization_id == {org_id}')

        if results[0].distances[0] < 0.35:
            matching_ticket = results[0][0].entity
            st.write(f"This case seems very similar to {matching_ticket.get('subject')} (id #{matching_ticket.get('_id')}). Make sure it has not been submitted before")
        else:
            # TODO Actually send out the ticket
            st.write("Submitted!")
            article_results = collection.search(data=[embedding], anns_field="vector", param={}, limit=5, output_fields=["title", "html_url"], expr=f'_ab_stream == "articles"')
            st.write(article_results[0])
            if len(article_results[0]) > 0:
                st.write("We also found some articles that might help you:")
                for hit in article_results[0]:
                    if hit.distance < 0.362:
                        st.write(f"* [{hit.entity.get('title')}]({hit.entity.get('html_url')})")


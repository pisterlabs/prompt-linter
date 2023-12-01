import streamlit as st
state = st.session_state
# from streamlit import components
import pandas as pd
from datetime import datetime
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
# import colorlover as cl
# colors = cl.to_rgb(cl.scales['7']['qual']['Set2'])
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

from scripts import tools, getdata, topicproc
tools.page_config()
tools.css()

# load data
if 'init' not in st.session_state:
    ready = getdata.init_data()
else:
    ready= True

if ready:

    df_filtered = state.df_filtered

    # header
    st.subheader('Topic modeling')
    getdata.df_summary_header()

    # placeholder for status updates
    placeholder = st.empty()

    placeholder.markdown('*. . . Initializing . . .*\n\n')
    # get source data
    data = [str(t).split() for t in df_filtered.clean_text.values.tolist() if str(t) not in ['None','nan']]

    placeholder.markdown('*. . . Compiling data . . .*\n')
    data_bigrams, id2word, corpus = topicproc.get_ta_models(data)

    placeholder.empty()

    # evaluate topics
    with st.form(key='topic_selection'):

        st.markdown('## Set topic parameters')

        nt_cols = st.columns([1,2,2])

        st.markdown('Select a number of topics to generate')

        with nt_cols[0]:
            nt_range = list(range(5,16))
            if 'num_topics' not in state:
                state.num_topics = 8

            nt_selected = state.num_topics
            # num_topics_selected = int(st.selectbox('Number of topics to generate', range(5,16)))
            # state.num_topics = int(st.selectbox('Number of topics to generate', range(5,16)))
            state.num_topics = int(st.selectbox('Number of topics to generate', nt_range, index=nt_range.index(nt_selected)))

        with nt_cols[1]:

            st.markdown('Select data display')
            ta_abs_btn = st.radio('Data display method', ['Absolute','Normalized'], label_visibility="hidden")
            st.caption("'Absolute' will show the raw count of articles on that topic. 'Normalized' will show the relative proportion of that topic for a given time (scale is 0 to 1.0)")

        with nt_cols[2]:

            st.markdown('Words to omit from the results')
            omit_tm = st.text_input('Results a little messy? Enter specific terms to omit from the results. Separate multiple terms with a comma.')

        topic_btn = st.form_submit_button('Generate topic model')

    topic_placeholder = st.empty()

    if topic_btn:

        if omit_tm != '':
            data = [[ct for ct in t.split() if ct not in [o.strip() for o in omit_tm.split(',')]] for t in df_filtered.clean_text.values.tolist()]
            # data = [t.split() for t in df_filtered.clean_text.values.tolist()]

            topic_placeholder.info('*Please wait . . . Compiling data . . .*\n')
            data_bigrams, id2word, corpus = topicproc.get_ta_models(data)

        topic_placeholder.info('*Please wait . . . Building topic models for dataset  (step 2 of 4) . . .*\n')

        lda_model = topicproc.get_lda_model(id2word, corpus, state.num_topics)
        topic_placeholder.info(f'*Please wait . . . Assigning {len(df_filtered):,} items to {state.num_topics} topics  (step 3 of 4) . . .*\n')
        topic_df = topicproc.get_topic_df(df_filtered, lda_model, corpus)

        # plot topic frequency over time
        topic_placeholder.info('*Please wait . . . Visualizing data (step 4 of 4) . . .*\n')

        topic_placeholder.empty()

        st.caption('Click an item in the legend to exclude from the results. Double click to isolate that item.')

        st.plotly_chart(topicproc.topics_by_month(lda_model, topic_df, ta_abs_btn),use_container_width=True, height=400)

        # perform topic modeling and convert to df
        with st.expander("Review topics"):

            for i in range(0,lda_model.num_topics):
                num_top = len(topic_df[topic_df.top_topic==str(i)])

                try:
                    num_all = topic_df[i].count()
                except:
                    num_all = 0

                sa_cols = st.columns([2,2,3])

                with sa_cols[0]:

                    st.markdown(f'## Topic {i+1}')
                    st.markdown(f'**Statistically present in** {num_all:,} items')
                    st.markdown(f'**Primary topic in** {num_top:,} items')

                with sa_cols[1]:
                    st.markdown(f'**Top keywords**')
                    st.markdown(f"{', '.join([lda_model.id2word[t[0]] for t in lda_model.get_topic_terms(i)])}")

                try:
                    wc = topicproc.get_wc(lda_model, i)
                except:
                    raise
                    wc = None
                    st.markdown(f'Topic {i+1} has no statistically significant results to display')

                if wc:
                    sa_cols[2].pyplot(wc)

            # with st.form(key='topic_form'):
            #     st.write('### Filter the dataset?')
            #     st.write("""Select one or more topics to limit the active dataset for review on other pages. This filter can be reset on the "Filter this dataset" page using the "Reset all filters" button.""")
            #     top_filter = st.multiselect("Topic select", [f'Topic {i}' for i in range(1,lda_model.num_topics + 1)],
            #                     label_visibility="hidden", key="topfilter")
            #     top_filter_button = st.form_submit_button(label='Apply filter')
            #
            # if top_filter_button:
            #     st.write(top_filter)


    with st.expander('What is the ideal number of topics to generate?'):

        st.write('The ideal number of topics for a given set of documents will vary, depending on factors such as ***content, thematic cohesiveness, and others...*** By comparing certain evaluation metrics, the user can refine the number of topics best suited for a given set of documents.')

        st.markdown("""In short, the **Perplexity** is an intrinsic evaluation measure of the predictive quality \
        of the language model, with a lower number representing a better model. While this is useful for machine \
        learning tasks, it may not be the best measure for creating a human interpretable model.""")
        st.markdown("""**Coherence** is a measure of the internal semantic similiarity within a topic, with a higher \
        number representing a more semantically clear topic. This number represents the overall topics’ interpretability\
        and is likely the better measure to use when creating a model to be visually reviewed.""")

        explore_topic_btn = st.button(label='Explore topic coherence for this dataset')

        topiceval_placeholder = st.empty()
        if explore_topic_btn:

            coherence_df = pd.DataFrame()
            ct = 1

            eval_range = range(5,16)
            for num_topics in eval_range:

                topiceval_placeholder.info(f'*Please wait . . . Evaluating metrics using {num_topics} topics ({ct} of {len(eval_range)}) . . .*')
                ct += 1
                lda_model_results = topicproc.get_lda_model(id2word, corpus, num_topics)
                # Compute Perplexity - a measure of how good the model is. lower the better.
                perplexity = lda_model_results.log_perplexity(corpus)
                # Compute Coherence Score - Higher the topic coherence, the topic is more human interpretable
                # using 'u_mass' which is known to be faster
                # c_v may be better but is unsustainably slow in this environment
                coherence_model_lda = CoherenceModel(model=lda_model_results, texts=data_bigrams, dictionary=id2word, coherence='u_mass')
                coherence_lda = coherence_model_lda.get_coherence()
                coherence_df = pd.concat([coherence_df, pd.DataFrame([{'Number of topics':num_topics,
                                    'Perplexity':perplexity, 'Coherence':coherence_lda}])])
                # coherence_df.drop_duplicates(inplace=True)
            coherence_df.reset_index(inplace=True, drop=True)

            topiceval_placeholder.empty()
            st.markdown("""This plot shows the evaluations of a variety of topic models created using the Gensim \
            implementation of Latent Dirichlet Allocation (LDA) method.""")

            coh_tops = coherence_df.sort_values('Coherence', ascending=False)[:3]['Number of topics'].tolist()
            st.write(f"**Based on these results, the ideal number of topics for this dataset would likely be *{coh_tops[0]}, {coh_tops[1]}, or {coh_tops[2]}*.**")

            coh_cols = st.columns([3,1])
            with coh_cols[0]:

                st.plotly_chart(topicproc.plot_coherence(coherence_df))
            with coh_cols[1]:
                st.dataframe(coherence_df)


    placeholder.empty()

    with st.expander('About this page'):

        st.write("""**Topic modeling** is an algorithmic process that evaluates a set of documents to \
        computationally identify topics contained within. A set of keywords is generated for each topic, \
        along with a probabilistic rate of occurrence for each keyword within that topic. Individual \
        documents are assigned a relative proportion of each topic (to total 100%) based on the content. \
        It is based on the assumptions that (a) a text (document) is composed of several topics, and \
        (b) a topic is composed of a collection of words.

        A topic modeling algorithm is a mathematical/statistical model used to infer what are the \
        topics that better represent the data. This site uses the Gensim library to perform a Latent \
        Dirichlet Allocation (LDA) analysis, a \
        method introduced in 2003 that is commonly used for NLP tasks.(Blei et al, 2003) LDA takes \
        an unsupervised approach to generate a pre-selected number of topics and assign each document \
        to one or more relevant topics.""")

        st.write("More information can be found in Gensim's documentation: https://radimrehurek.com/gensim/models/coherencemodel.html")
        st.markdown("### References")
        st.caption('[1] Blei, D. M., Ng, A. Y., Jordan, M. I. "Latent dirichlet allocation", The Journal of Machine Learning Research, 3 (2003):993–1022.')
        st.caption('[2] Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. In Proceedings of the eighth ACM international conference on Web search and data mining (pp. 399–408). Retrieved from https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf')

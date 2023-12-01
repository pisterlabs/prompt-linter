import pinecone_helpers
import langchain_helpers
import pinecone_helpers
import openai_helpers
import youtube_helpers


def summarize_top_k_comments(query, video_id, title, k):
    comments = youtube_helpers.get_k_comments(video_id,k)
    tokens = openai_helpers.num_tokens_from_string(''.join(comments))
    print('context tokens are ', tokens)
    prompt = f'''
    Please summarize the following comments for a youtube video. Try and focus on answering the 
    query in your summary.  If the query cannot be answered with a summary, return 'none'
    title: {title}
    comments: {comments}
    query: {query}
    summary:'''
    response = openai_helpers.generate_chat_completion(prompt)
    return response


def get_relevant_comments(query, title, video_id):
    #embedding = langchain_helpers.use_HyDE(title).embed_query(query)
    hypothetical_comments = openai_helpers.get_HyDE(query, title, 3)
    print(hypothetical_comments)
    embeded_hypothetical_query = openai_helpers.get_query_embedding(hypothetical_comments)
    context = pinecone_helpers.get_context(embeded_hypothetical_query, video_id, 9)
    comments = [vector['metadata']['text'] for vector in context]
    return comments

def want_summary_or_comments(query, title):
    prompt = f'''
    For a given query respond with the most appropriate way to format answers to question about a youtube video. 
    The 2 ways to format answers are 'summary' and 'comment list'. This is a binary decision and you must provide
    one of these answers. For context the title of the video is provided
    title: {title}
    Query: What do the top comments say?
    Answer format: summary
    Query: Are people happy with this video?
    Answer format: summary
    Query: what do people think of this video?
    Answer format: summary
    Query: Who else is excited about the future?
    Answer format: comment list
    Query: Sam is off his rocker
    Answer format: comment list
    Query: Summarize the negative comments
    Answer format: summary
    Query: {query}
    Answer format:'''
    response = openai_helpers.generate_chat_completion(prompt)
    return response
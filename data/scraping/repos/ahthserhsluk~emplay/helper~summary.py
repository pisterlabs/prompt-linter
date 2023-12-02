import cohere
co = cohere.Client('yNDbN0b9zyS85S8ny9ibVFLD9M2kqA9yiT64Vxnd')




def get_summary(text):
    summary = co.summarize(text)
    return summary


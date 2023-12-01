from transformers import T5Tokenizer, T5Model
import torch
from openai.embeddings_utils import cosine_similarity

# load the T5 tokenizer and model. try 't5-base' instead t5-small,  T5-Small 这个同样架构下的小模型，参数数量只有 6000 万个, t5-base🈶️2.2亿个。
# 这段代码执行的过程可能会有点慢。因为第一次加载模型的时候，Transformer 库会把模型下载到本地并缓存起来，整个下载过程会花一些时间。
tokenizer = T5Tokenizer.from_pretrained('t5-small', model_max_length=512)
model = T5Model.from_pretrained('t5-small')
# set the model to evaluation mode
model.eval()

# encode the input sentence
def get_t5_vector(line):
    input_ids = tokenizer.encode(line, return_tensors='pt', max_length=512, truncation=True)
    # generate the vector representation
    with torch.no_grad():
        outputs = model.encoder(input_ids=input_ids)
        vector = outputs.last_hidden_state.mean(dim=1)
    return vector[0]


positive_review_in_t5 = get_t5_vector("An Amazon review with a positive sentiment.")
negative_review_in_t5 = get_t5_vector('An Amazon review with a negative sentiment.')

positive_text = """Wanted to save some to bring to my Chicago family but my North Carolina family ate all 4 boxes before I could pack. These are excellent...could serve to anyone"""
negative_text = """First, these should be called Mac - Coconut bars, as Coconut is the #2 ingredient and Mango is #3.  Second, lots of people don't like coconut.  I happen to be allergic to it.  Word to Amazon that if you want happy customers to make things like this more prominent.  Thanks."""

def test_t5():
  positive_example_in_t5 = get_t5_vector(positive_text)
  negative_example_in_t5 = get_t5_vector(negative_text)

  def get_t5_score(sample_embedding):
    return cosine_similarity(sample_embedding, positive_review_in_t5) - cosine_similarity(sample_embedding, negative_review_in_t5)

  positive_score = get_t5_score(positive_example_in_t5)
  negative_score = get_t5_score(negative_example_in_t5)

  print("T5好评例子的评分 : %f" % (positive_score))
  print("T5差评例子的评分 : %f" % (negative_score))

test_t5()
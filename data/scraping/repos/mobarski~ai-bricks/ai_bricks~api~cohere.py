import cohere
import time
import os

api_key = None


def use_key(key):
	global api_key
	api_key = key
if not api_key:
	use_key(os.getenv('COHERE_KEY'))


# models: medium|xlarge
def model(name, **kwargs):
	return TextModel(name, **kwargs)


class TextModel:
	PARAMS = ['model','temperature','stop_sequences'] # TODO
	MAPPED = {'stop':'stop_sequences'}
	
	def __init__(self, name, **kwargs):
		self.name = name
		self.config = kwargs
		self.config['model'] = name
		self.client = cohere.Client(api_key)

	def complete(self, prompt, **kw):
		out = {}
		#
		kwargs = dict(
			prompt = prompt,
			max_tokens = 100, # TODO
		)
		config = self.config.copy()
		config.update(kw) # NEW
		for k,v in config.items():
			k = self.MAPPED.get(k,k) # NEW
			if k in self.PARAMS:
				kwargs[k] = v
		t0 = time.time()
		#
		resp = self.client.generate(**kwargs)
		#
		out['rtt'] = time.time() - t0
		out['text'] = resp.generations[0].text
		out['raw'] = resp # XXX
		return out
	
	def complete_many(self, prompts):
		out = {}
		out['texts'] = []
		t0 = time.time()
		for p in prompts:
			resp = self.complete(p)
			out['texts'] += [resp['text']]
		out['rtt'] = time.time() - t0
		return out
	
	def embed(self, text):
		out = {}
		resp = self.embed_many([text])
		out['vector'] = resp['vectors'][0]
		return out
	
	def embed_many(self, texts):
		out = {}
		t0 = time.time()
		resp = self.client.embed(texts)
		out['rtt'] = time.time() - t0
		out['vectors'] = resp.embeddings
		return out


# REF: https://cohere.ai/pricing
# REF: https://dashboard.cohere.ai/api-keys
# REF: https://docs.cohere.ai/reference/generate
# REF: https://docs.cohere.ai/reference/embed
# REF: https://docs.cohere.ai/reference/tokenize

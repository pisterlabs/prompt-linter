# USAGE:
# ut = DBUsageTracker('usage',100)
# for _ in range(10):
#   resp = openai.ChatCompletion.create(**kwargs)
#   ut.add_resp(resp)
# print(ut.get_usage())

from random import randint

class UsageTracker:

	def __init__(self):
		self.by_model = {}

	def add_resp(self, resp: dict):
		"add usage from OpenAI API response"
		model = resp['model']
		part_id = self._get_part_id()
		usage = self._get_usage_part(part_id)
		self._update(usage, resp, model)
		self._sync(usage, part_id)

	def get_usage(self) -> dict:
		"return usage by model"
		return self.by_model

	def get_cost(self) -> dict:
		"return cost by model (USD)"
		pass # TODO

	# internal methods

	def _update(self, usage, resp, model):
		"update usage by model object"
		if model not in usage:
			usage[model] = {}
		#
		resp['usage']['calls'] = 1
		for k,v in resp['usage'].items():
			if k not in usage[model]:
				usage[model][k] = 0
			usage[model][k] += v

	def _get_part_id(self) -> str:
		"return partition id for usage object"
		return 'default'

	def _get_usage_part(self, part_id):
		return self.by_model

	def _sync(self, usage, part_id):
		pass


class PartitionedUsageTracker(UsageTracker):
	"partitioning the tracker facilitates handling race conditions in non-atomic storage"
	def __init__(self, name='usage', n_partitons=10):
		self.db = self._get_db()
		self.n_partitons = n_partitons
		self.name = name

	def _get_db(self):
		return {}

	def _get_part_id(self):
		p = randint(0, self.n_partitons-1)
		return str(p)

	def _part_key(self, part_id):
		return f"{self.name}-{part_id}"

	def _get_usage_part(self, part_id):
		return self.db.get(self._part_key(part_id), {})

	def _sync(self, usage, part_id):
		self.db[self._part_key(part_id)] = usage

	def get_usage(self):
		"return aggregated usage from all partitions"
		total = {}
		for i in range(0, self.n_partitons):
			usage = self.db.get(self._part_key(i), {})
			for m in usage:
				if m not in total:
					total[m] = {}
				for k,v in usage[m].items():
					if k not in total[m]:
						total[m][k] = 0
					total[m][k] += v
		return total


class DBUsageTracker(PartitionedUsageTracker):
	def _get_db(self):
		import databutton as db
		return db.storage.json

	def _get_usage_part(self, part_id):
		return self.db.get(self._part_key(part_id), default={})

	def _sync(self, usage, part_id):
		self.db.put(self._part_key(part_id), usage)


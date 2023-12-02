import guidance
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM

@pytest.mark.parametrize("model_name,valid_options",[("gpt2", ["HTR2B","HTR2A"]), ("gpt2", ["GPEE", "ABCD", "ABCE", "ABCF"]), 
("gpt2", ["HTR2C","NR3C1","HTR2A"])])
class TestGuidance():
	def test_select_options(self,model_name, valid_options):
		tokenizer = AutoTokenizer.from_pretrained(model_name)
		model = AutoModelForCausalLM.from_pretrained(model_name)

		gpt2 = guidance.llms.Transformers(model=model, tokenizer=tokenizer)
		guidance.llm = gpt2
		program = guidance("""The best option is: {{select 'acronyms' options=valid_options}}""")
		executed_program = program(valid_options=valid_options)
		assert False
		assert executed_program["acronyms"] in valid_options


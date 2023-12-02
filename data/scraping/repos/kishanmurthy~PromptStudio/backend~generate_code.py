from open_api_calls import generate_openai_output


def generate_python_code(flow_name, version, node_input, node_dict, tp_sort):
	lines = []

	# import statements 
	lines.append("import sys\n")
	lines.append("from langchain.llms import OpenAI\n")
	lines.append("from langchain.prompts import PromptTemplate\n\n")

	lines.append("apikey = sys.argv[1]\n\n")

	# openai api call with prompts as parameter and response as output
	openai_api_call_function = "def generate_openai_output(prompt, input_dict, input_tags, apikey):\n" +  "\tprompt_tempt = PromptTemplate.from_template(prompt)\n"  + "\tfiltered_dict = {key: input_dict[key] for key in input_tags}\n" + "\tfinal_prompt = prompt_tempt.format(**filtered_dict)\n" + "\tllm = OpenAI(openai_api_key=apikey)\n" + "\toutput = llm.predict(final_prompt)\n" + "\treturn output\n\n"
	lines.append(openai_api_call_function)

	# add line to call openai api for each prompt
	for input in node_input:
		data = node_input[input]
		data = data.replace("\n", "\\n")
		data = data.replace("\"", "\\\"")
		input = input.replace("{{", "")
		input = input.replace("}}", "")
		lines.append(input + "= \"" + str(data) + "\"\n")

	lines.append("\n")
	lines.append("input_dict = {}\n")
	node_output_tag = ""
	for node_id in tp_sort:
		node = node_dict[node_id]
		if node.node_type == "prompt":
			input_tags = []
			for node_incoming in node.node_incomings:
				tag = node_dict[node_incoming].node_output_tag
				tag = tag.replace("{{", "")
				tag = tag.replace("}}", "")
				input_tags.append(tag)
			node_prompt = node.node_prompt
			node_prompt = node_prompt.replace("\n", "\\n")
			node_prompt = node_prompt.replace("\"", "\\\"")
			node_prompt = node_prompt.replace("{{", "{")
			node_prompt = node_prompt.replace("}}", "}")

			
			node_output_tag = node.node_output_tag
			node_output_tag = node_output_tag.replace("{{", "")
			node_output_tag = node_output_tag.replace("}}", "") 
			lines.append("prompt = \"" + node_prompt + "\"\n")

			
			for tag in input_tags:
				lines.append("input_dict[\"" + tag + "\"] = " + tag + "\n")

			input_tag_string = ""
			for tag in input_tags:
				input_tag_string = input_tag_string + "\"" + tag + "\"" + ","
			print(input_tag_string)
			input_tag_string = input_tag_string[:len(input_tag_string)-1]
			lines.append("input_tags = [" + input_tag_string + "]\n")
			lines.append(node_output_tag + "= generate_openai_output(prompt, input_dict, input_tags, apikey)\n\n")

	lines.append("print(" + node_output_tag + ")\n")

	file_path = "generated_codes/" + str(flow_name) + "_" + str(version) + ".py"
	file = open(file_path,'w')
	for line in lines:
		file.write(line)
	return file_path


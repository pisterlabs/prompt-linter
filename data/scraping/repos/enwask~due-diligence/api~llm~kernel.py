from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

from api.llm.skills import *
from api.secrets import OPENAI_KEY

# Construct the kernels and add AI services
kernel = Kernel()
kernel_finetuned = Kernel()
kernel.add_chat_service("gpt", OpenAIChatCompletion("gpt-3.5-turbo", OPENAI_KEY))
kernel_finetuned.add_chat_service("gpt", OpenAIChatCompletion("ft:gpt-3.5-turbo-0613:enwask::8Gu2ySO6", OPENAI_KEY))

# Set up the product comparison skill
product_comparison: Skill = load_skill(kernel, "ProductComparison")
product_comparison_finetuned: Skill = load_skill(kernel_finetuned, "ProductComparison")

# Skill functions for extracting features
extract_features_fun: SkillFunction = get_skill_function(product_comparison_finetuned, "ExtractFeatures")
extract_features_async_fun: AsyncSkillFunction = get_async_skill_function(product_comparison_finetuned, "ExtractFeatures")

# Skill functions for shortening product names
shorten_name_fun: SkillFunction = get_skill_function(product_comparison, "ShortenName")
shorten_name_async_fun: AsyncSkillFunction = get_async_skill_function(product_comparison, "ShortenName")

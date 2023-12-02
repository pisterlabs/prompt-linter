import os
import openai

openai.api_key = os.getenv("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")

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Product description: A home milkshake maker\nSeed words: fast, healthy, compact.\nProduct names: HomeShaker, Fit Shaker, QuickShake, Shake Maker\n\nProduct description: A pair of shoes that can fit any foot size.\nSeed words: adaptable, fit, omni-fit.",
  temperature=0.8,
  max_tokens=60,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)

# Prompt
# Product description: A home milkshake maker
# Seed words: fast, healthy, compact.
# Product names: HomeShaker, Fit Shaker, QuickShake, Shake Maker

# Product description: A pair of shoes that can fit any foot size.
# Seed words: adaptable, fit, omni-fit.
# Sample response
# Product names: AdaptFit, OmniSecure, Fit-All, AdaptShoes.
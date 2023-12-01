import os
from nightfall import Confidence, DetectionRule, Detector, RedactionConfig, MaskConfig, Nightfall
import os
import openai, json, secrets, my_secrets

with open('api_key.txt', 'r') as f:
    api_key = f.read().strip()

nightfall = Nightfall(api_key) # By default Nightfall will read the NIGHTFALL_API_KEY environment variable
# openai.api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = my_secrets.OPENAI_API_KEY


# The message you intend to send
prompt = "what number is this 4916-6734-7572-5015?"
payload = [ prompt ]

# Define an inline detection rule that looks for Likely Credit Card Numbers and redacts them
detection_rule = DetectionRule([
                            Detector(
                                min_confidence=Confidence.LIKELY,
                                nightfall_detector="CREDIT_CARD_NUMBER",
                                display_name="Credit Card Number",
                                redaction_config=RedactionConfig(
                                    remove_finding=False, 
                                    substitution_phrase="[REDACTED]")
                            )])

# Send the message to Nightfall to scan it for sensitive data
# Nightfall returns the sensitive findings, and a copy of your input payload with sensitive data redacted
findings, redacted_payload = nightfall.scan_text(
                        payload,
                        detection_rules=[detection_rule])

# If the message has sensitive data, use the redacted version, otherwise use the original message
if redacted_payload[0]:
    message_body = redacted_payload[0]
else:
    message_body = payload[0]

print("After content filtering - this is what will be sent to ChatGPT:\n\n", message_body, "\n\n----\n\n")

# Send prompt to OpenAI model for AI-generated response
completion = openai.ChatCompletion.create(
  model='gpt-3.5-turbo',
  messages=[
    {"role": "user", "content": message_body}
  ]
)

print("Here's a generated response you can send the customer:\n\n", completion['choices'][0].message.content)
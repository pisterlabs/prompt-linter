import langchain
import openai
import streamlit
import hubspot

# Retrieve customer preferences and previous interactions from Hubspot
customer_preferences = hubspot.get_customer_preferences()
previous_interactions = hubspot.get_previous_interactions()

# Generate personalized reminders using Langchain analysis and OpenAI
latest_interaction = previous_interactions[-1]
analysis_result = langchain.analyze(latest_interaction)
reminder_message = openai.generate_reminder_message(analysis_result, customer_preferences)

# Send reminder message to customer via email or text
reminder_type = customer_preferences.get("reminder_type")
if reminder_type == "email":
    hubspot.send_email_reminder(reminder_message)
elif reminder_type == "text":
    hubspot.send_text_reminder(reminder_message)

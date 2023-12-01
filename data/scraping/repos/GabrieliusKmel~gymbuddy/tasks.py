from celery import shared_task
from openai import OpenAI
from django.utils import timezone
from datetime import timedelta
from django.contrib import messages

@shared_task
def generate_chat_advice_task(profile_id):
    from .models import Profile
    profile = Profile.objects.get(pk=profile_id)
    if not profile.get_chat_advice():
        if profile.is_complete():
                client = OpenAI(
                    api_key="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",
                )
                conversation = f"User Profile:\nHeight: {profile.height} cm.\nWeight: {profile.weight} kg.\nAge: {profile.age} years old.\nGender: {profile.get_gender_display()}\nActivity Level: {profile.get_activity_level_display()}\nWeight Goal: {profile.get_weight_goal_display()}"
                messages = [
                    {
                        "role": "system",
                        "content": "Be super straightforward and simple. You are the good trainer and nutritionist. Give a detailed meal plan with calories count and a detailed workout plan in the gym with reps count for the given user profile data and rest plan. Don't write that you were given this data. Just give the answer. Also, if you can add headings you write, like: 'Meal Plan:', 'Workout Plan:', 'Rest Plan:', 'Note:'. Thank you.",
                    },
                    {
                        "role": "user",
                        "content": conversation,
                    }
                ]
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages
                )
                chat_response = response.choices[0].message.content.strip()
                profile.set_chat_advice(chat_response)
                time_now = timezone.now()
                profile.set_chat_advice_time(time_now)
                time_left = time_now + timedelta(days=1)
                profile.set_time_left(time_left)
    else:
        time_now = timezone.now()
        print(profile.time_left, time_now)
        if profile.time_left < time_now:
            client = OpenAI(
                api_key="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",
            )
            conversation = f"User Profile:\nHeight: {profile.height} cm.\nWeight: {profile.weight} kg.\nAge: {profile.age} years old.\nGender: {profile.get_gender_display()}\nActivity Level: {profile.get_activity_level_display()}\nWeight Goal: {profile.get_weight_goal_display()}"
            messages = [
                {
                    "role": "system",
                    "content": "Be super straightforward and simple. You are the good trainer and nutritionist. Give a detailed meal plan with calories count and a detailed workout plan in the gym with reps count for the given user profile data and rest plan. Don't write that you were given this data. Just give the answer. Also, if you can add headings you write, like: 'Meal Plan:', 'Workout Plan:', 'Rest Plan:', 'Note:'. Thank you.",
                },
                {
                    "role": "user",
                    "content": conversation,
                }
            ]
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages
            )
            chat_response = response.choices[0].message.content.strip()
            profile.set_chat_advice(chat_response)
            time_now = timezone.now()
            profile.set_chat_advice_time(time_now)
            time_left = time_now + timedelta(days=1)
            profile.set_time_left(time_left)

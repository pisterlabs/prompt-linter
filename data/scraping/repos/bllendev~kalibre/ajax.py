# django
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import get_user_model


CustomUser = get_user_model()

# tools
import os
import json
import openai

# local
from ai.models import TokenUsage, Conversation, Message
from ai.constants import TOKEN_USAGE_DAILY_LIMIT, AI_PROMPT
import tiktoken


# update your `update_token_usage` function to use `tiktoken`
def update_token_usage(request, message):
    date_today = timezone.now().date()
    token_usage, created = TokenUsage.objects.get_or_create(user=request.user, date=date_today)

    # use tiktoken to count the tokens in the new messages only
    tokenizer = tiktoken.get_encoding("cl100k_base")  # moved outside of the loop
    token_ids = list(tokenizer.encode(message))

    # update the user's total tokens used
    token_usage.tokens_used += len(token_ids)
    token_usage.save()


def query_ai(request, user_message, summary=False):
    """
    - sends a message to the AI Librarian (GPT-3.5 Turbo)
    - accounts for token usage (TokenUsage model)
    returns:
       the AI's response (str)
    """
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # extract user info
    username = request.user.username
    user = CustomUser.objects.get(username=username)

    # create a conversation if not exists
    conversation_id = request.POST.get('conversation_id')
    conversation = Conversation.objects.get(id=conversation_id, user=user)

    # create a message from user
    Message.objects.create(
        conversation=conversation,
        sender=Message.SENDER_USER,
        text=user_message,
        sent_at=timezone.now()
    )
    update_token_usage(request, user_message)

    # get ai response using current conversation
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation.get_messages(),
        max_tokens=1000,
        temperature=0.7,
    )
    ai_message = response['choices'][0]['message']['content'].strip()

    # create a message from AI
    Message.objects.create(
        conversation=conversation,
        sender=Message.SENDER_AI,
        text=ai_message,
        sent_at=timezone.now()
    )

    # summarize if the user has exceeded their daily token limit
    if summary:
        ai_message = set_latest_messages(ai_message)

    # update the token usage after a successful request
    update_token_usage(request, ai_message)  # updating with the correct argument
    return ai_message


def set_latest_messages(messages):
    """
    - sets the latest messages to the last two messages in the conversation (to lower TokenUsage)
    - the last message is the system's role
    - the second to last message is the summary of the conversation
    """
    if len(messages) > 2:
        # Take only the last two messages
        messages = messages[-2:]
    return messages


def ai_librarian(request):
    """
    - ajax view that sends a message to the AI Librarian (GPT-3.5 Turbo)
    """
    if request.method == 'POST':
        # extract messages from request
        user_message = ""
        ai_message = None
        try:
            user_message_json = request.POST.get('user_message', "")
            user_message = json.loads(user_message_json)
        except Exception as e:
            print(f"ERROR: {e}")

        # extract user info
        username = request.user.username
        user = CustomUser.objects.get(username=username)

        date_today = timezone.now().date()
        token_usage, created = TokenUsage.objects.get_or_create(user=user, date=date_today)
        try:
            # check if the user has exceeded their daily token limit
            if token_usage.tokens_used > TOKEN_USAGE_DAILY_LIMIT:
                return JsonResponse({'error': 'You have exceeded your daily token limit.'}, status=400)  # 400 -> bad request
            else:
                ai_message = query_ai(request, user_message)
        except Exception as e:
            print(f"ai.ajax.ai_librarian ERROR: {e}")
            return JsonResponse({'error': f'An error occurred while processing your request...'}, status=500)  # 500 -> internal server error

        return JsonResponse({'message': ai_message})

    return JsonResponse({'error': 'Invalid request method'}, status=400)  # 400 -> bad request


@csrf_exempt
def create_conversation(request):
    try:
        # extract user info
        username = request.user.username
        user = CustomUser.objects.get(username=username)

        # check if conversation_id exists in the session
        conversation = Conversation.objects.create(user=user)
        system_query = [{"role": "system", "content": AI_PROMPT}]

        # create a message for the system prompt (ai)
        Message.objects.create(
            conversation=conversation,
            sender=Message.SENDER_AI,
            text=AI_PROMPT,
        )

        # return the conversation_id in the response
        return JsonResponse({'conversation_id': conversation.id})

    except Exception as e:
        print(f"create_conversation - ERROR: {e}")
        return JsonResponse({'error': 'Internal Server Error'}, status=500)

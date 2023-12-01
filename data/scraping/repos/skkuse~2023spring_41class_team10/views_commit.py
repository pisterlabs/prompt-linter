from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response

import os
import json
import openai
import requests

from backend.settings import OPENAI_API_KEY, DATA_DIR
from users.views import get_access_token

chat_role = "assistant"


class GitHubCommitView(APIView):

    def get(self, request, *args, **kwargs):
        code = request.GET.get('code')
        access_token = get_access_token(code)
        headers = {'Authorization': f'token {access_token}'}
        res = requests.get('https://api.github.com/user/repos', headers=headers)
        print("res", res)

        return Response({"status": "success", "message": "Commit and push successful"})


class GitHubCommitMessageView(APIView):
    suffix = "\nPlease look at the code and suggest a Github commit message. No additional explanation other than the commit message is required. Please answer in English"
    meta_messages = []

    def post(self, request, *args, **kwargs):
        print("GitHubCommitMessageView")
        chat_messages = self.meta_messages
        openai.api_key = OPENAI_API_KEY
        openai.Model.list()
        body = json.loads(request.body.decode('utf-8'))
        
        role = body.get("role", "user")
        old_code = body.get("old_code", "")
        new_code = body.get("new_code", "")
        lang = body.get("lang", "")
        submission_id = body.get("submission_id", 0)
        
        # submission_id 있으면 과거와 비교해서 commit message 생성
        # 없으면 첫 커밋으로 생각하고 commit message 생성
        if old_code == "" :
            content = f"""The {lang} code has been written by the user.\nCODE\n{new_code}\nThis code is user's first commit."""
        else:
            content = f"""The {lang} code has been written by the user.\nOLD CODE\n{old_code}\nNEW CODE\n{new_code}\ncommit Notice the difference between the two codes when proposing a commit message."""

        chat_messages.append({"role": role, "content": content+self.suffix})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=chat_messages,
            max_tokens=1024
        )

        answer = completion.choices[0].message['content']
        print("answer", answer)
        chat_messages.append({"role": chat_role, "content": answer})

        return JsonResponse({"status": "success", "message": answer})

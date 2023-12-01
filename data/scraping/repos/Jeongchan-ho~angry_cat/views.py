from rest_framework import status, permissions
from rest_framework.decorators import APIView
from rest_framework.response import Response
import openai
import os
from ai_process.serializers import PictureSerializer
from ai_process.cat import picture_generator
from .models import Picture


class MentgenView(APIView):
    """MentgenView

    chat gpt로 고양이 멘트를 생성합니다.

    Attributes:
        permission (permissions): IsAuthenticated 로그인한 사용자만 접속을 허용합니다.
    """

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        """Mentgen.post

        post요청 시 입력받은 description으로 cat_says를 생성하여 반환합니다.

        정상 시 200 / "unlike했습니다." || "like했습니다." 메시지 반환
        오류 시 401 / 권한없음(비로그인)
        오류 시 404 / 존재하지않는 게시글
        """
        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "입력값에 주어진 상황(참고로 이것은 사진의 내용과 연관이 있다옹!)을 지켜보던 고양이가 있다고 가정한다옹. 문장의 화자와 너(고양이)는 다른 객체임을 명확히 인지하라냥! 그 고양이가 화난 이유를 한문장으로 만들어라냥. 웃기고 고양이다운 이유여야한다옹! 부적절한 문장이거나 이해할 수 없다면, 고양이답게 화내고 공격하겠다고 협박해라냥!!",
                },
                {"role": "user", "content": request.data.get("description", "")},
            ],
        )
        message = result.choices[0].message.content
        return Response({"message": message}, status=status.HTTP_200_OK)


class PicgenView(APIView):
    """PicgenView

    post 요청시 입력된 사진으로 변환된 사진을 생성하여 반환합니다.
    매 요청마다 두 사진을 Picture모델에 저장합니다.

    Attributes:
        permission (permissions): IsAuthenticated 로그인한 사용자만 접속을 허용합니다.
    """

    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        """PicgenView.post

        post요청 시 입력받은 사진으로 변환된 사진을 생성하여 반환합니다.
        """
        Picture.objects.filter(article=None, author=request.user).delete()
        serializer = PictureSerializer(data=request.data)
        if serializer.is_valid():
            orm = serializer.save(author=request.user)
            change_pic = picture_generator("media/" + orm.__dict__["input_pic"])
            orm.change_pic = change_pic.replace("media/", "")
            orm.save()
            new_serializer = PictureSerializer(instance=orm)
            return Response(new_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

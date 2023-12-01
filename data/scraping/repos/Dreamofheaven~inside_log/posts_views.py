from rest_framework.decorators import api_view,permission_classes
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.response import Response
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from base.models import Post, Review
from base.serializers import PostSerializer, ReviewSerializer
from rest_framework import status
from datetime import datetime
import openai
import os 

# set api key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Call the chat GPT API
def completion(word):
    response = openai.ChatCompletion.create(
        model = 'gpt-3.5-turbo',
        messages = [
            {'role': 'system', 'content': '너는 부정적인 언어를 긍정적으로 순화시켜주는 위로와 조언의 동반자이다.'},
            {'role': 'user','content': word}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

@api_view(['GET'])
def getPosts(request):
    query = request.query_params.get('keyword')
    if query == None:
        query = ''
    posts = Post.objects.filter(user_id=request.user, title__icontains=query)
    serializer = PostSerializer(posts, many=True) 
    return Response({'posts': serializer.data})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def createPosts(request):
    data = request.data
    user = request.user
    now = datetime.now()
    posts=Post.objects.create(
        title= data['title'],
        body= data['body'],
        user_id=user,
        status=True,
        created_at=now,
    )
    serializer=PostSerializer(posts, many=False)
    return Response(serializer.data)

@api_view(['GET'])
def getPostsReview(request,pk):
    reviews = Review.objects.filter(post_id=pk)    
    serializer = ReviewSerializer(reviews, many=True) 
    return Response(serializer.data)

@api_view(['POST'])
def createPostsReview(request,pk):
    post = Post.objects.get(id=pk)
    comment = completion(f"{post.body}. 답변은 200자 이내로 한글로 말해줘.") 
    print(comment)
    data = request.data
    now = datetime.now()
    if comment:
        content = {'detail':'아직 안 풀렸구나. 새로운 문장을 만들어줄게.'}
        review = Review.objects.filter(post_id=pk).delete() 
        review = Review.objects.create(
            post = post, 
            name = 'chatgpt',
            comment = comment,
            createdAt = now,
        )
        serializer = ReviewSerializer(review, many=False)
        return Response(serializer.data)
    else: 
        review = Review.objects.create(
            post = post,
            name = 'chatgpt',
            comment = comment,
            createdAt = now,
        )
        serializer = ReviewSerializer(review, many=False)
        return Response(serializer.data)

@api_view(['GET'])
def getPost(request, pk):
    try:
        post = Post.objects.get(id=pk)
    except Post.DoesNotExist:
        return Response({'detail': 'Post not found.'}, status=status.HTTP_404_NOT_FOUND)
    serializer = PostSerializer(post)
    return Response(serializer.data)

@api_view(['PUT'])
@permission_classes([IsAuthenticated])
def updatePosts(request, pk):
    data = request.data
    post = Post.objects.get(id=pk) 
    post.title = data['title']
    post.body = data['body']
    post.save()
    serializer = PostSerializer(post, many=False)
    return Response(serializer.data)

@api_view(['DELETE'])
def deletePosts(request, pk):
    post = Post.objects.get(id=pk)
    post.delete()
    return Response('Post Deleted')

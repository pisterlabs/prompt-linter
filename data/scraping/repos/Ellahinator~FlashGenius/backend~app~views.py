import json
import os
import datetime
import uuid
from dotenv import load_dotenv
import openai
from django.shortcuts import render, get_object_or_404
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework_simplejwt.tokens import RefreshToken
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.permissions import IsAuthenticated
from .forms import UserCreationForm, FlashcardForm
from .models import Deck, Flashcard, DeckFlashcard
from .serializers import DeckSerializer, FlashcardSerializer, DeckFlashcardSerializer
from django.contrib.auth import login, logout, authenticate,update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.models import User
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_exempt
# Create your views here.

# Initialize OpenAI API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class IndexView(APIView):
    def get(self, request):
        return Response({'message': 'Hello, world!'})

# Protected route example
class ProtectedView(APIView):
    authentication_classes = (JWTAuthentication,)
    permission_classes = (IsAuthenticated,)

    def get(self, request):
        return Response({'message': 'You are authenticated.'})

class AuthView(APIView):
    @csrf_exempt
    def post(self, request, action):
        data = parse_request_body(request)
        if data is None:
            return Response({"status": "error", "message": "Invalid JSON."}, status=status.HTTP_400_BAD_REQUEST)
        
        if action == 'signup':
            return self.signup(request, data)
        elif action == 'login':
            return self.login(request, data)
        elif action == 'logout':
            return self.logout(request)
        elif action == 'change_password':
            return self.change_password(request,data)
        
        return Response({"status": "error", "message": "Invalid action."}, status=status.HTTP_400_BAD_REQUEST)

    def signup(self, request, data):
        form = UserCreationForm(data)
        if form.is_valid():
            user = form.save()
            access_token = create_jwt_with_user_info(user)
            login(request, user)
            return Response({"status": "success", "access_token": access_token, "message": "Registration successful."}, status=status.HTTP_201_CREATED)
        else:
            return Response({"status": "error", "errors": form.errors}, status=status.HTTP_400_BAD_REQUEST)

    def login(self, request, data):
        username = data.get('username')
        password = data.get('password')
        user = authenticate(username=username, password=password)
        if user is not None:
            access_token = create_jwt_with_user_info(user)
            return Response({"status": "success", "access_token": access_token, "message": f"You are now logged in as {user}."}, status=status.HTTP_200_OK)
        else:
            if username_exists(username):
                return Response({"status": "error", "message": "Incorrect password."}, status=status.HTTP_400_BAD_REQUEST)
            return Response({"status": "error", "message": "Incorrect username."}, status=status.HTTP_400_BAD_REQUEST)

    def logout(self, request):
        logout(request)
        return Response({"status": "success", "message": "Successfully logged out."}, status=status.HTTP_200_OK)

    def change_password(self,request,data):
        
        user = request.user
        form = PasswordChangeForm(user,data)
        if form.is_valid():
            user = form.save()
            # Update the session to prevent the user from being logged out after changing the password
            update_session_auth_hash(request,user)
            return Response({"status":"success","message":"Password changed successfully"},status=status.HTTP_200_OK)
        else:
            return Response({"status":"error","errors":form.errors},status=status.HTTP_400_BAD_REQUEST) 


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_info(request):
    user = request.user
    return Response({
        'user_id': user.id,
        'username': user.username,
        'email': user.email
    })

@ensure_csrf_cookie
def get_csrf_token(request):
    return Response({'status': "success", "message": "CSRF cookie set"})

class DeckView(APIView):
    authentication_classes = (JWTAuthentication,)
    permission_classes = (IsAuthenticated,)

    @csrf_exempt
    def post(self, request, action):
        if action == 'create':
            return self.create_deck(request)
        elif action == 'delete':
            deck_id = request.data.get('deck_id')
            return self.delete_deck(request, deck_id)
        elif action == 'update':
            deck_id = request.data.get('deck_id')
            deck_name = request.data.get('deck_name')
            return self.update_deck(request, deck_id, deck_name)

        elif action == 'get':
            deck_id = request.data.get('deck_id')
            return self.get_deck(request, deck_id)
        else:
            return Response({"message": "Invalid action."}, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request):
        try:
            user_decks = Deck.objects.filter(user=request.user)
            serializer = DeckSerializer(user_decks, many=True)
            return Response({
                "status": "success",
                "decks": serializer.data
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                "status": "error",
                "message": str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                                
    def create_deck(self, request):
        # Assume you are receiving a block of text and deck_id as POST data
        flashcard_content = request.data.get('content', '')
        # Retrieve the deck_name or generate a unique one if not provided
        deck_name = request.data.get('deck_name', f"Deck-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}")

        try:
            print("Calling OpenAI API...")
            # Call OpenAI API to generate flashcards
            response = openai.ChatCompletion.create(
                model="ft:gpt-3.5-turbo-0613:personal::8HjCOVU3",
                messages=[
                    {
                        "role": "system",
                        "content": "Given a block of text, extract key terms and their explanations, and format them into a set of flashcards in JSON format. Each flashcard contains a term and its corresponding definition."},
                    {
                        "role": "user",
                        "content": flashcard_content
                    }
                ]
            )
            # Parse the JSON string from the response
            assistant_content = json.loads(response['choices'][0]['message']['content'])
            flashcards_data = assistant_content['flashcards']
            # Create a new deck for this block of text
            deck = Deck.objects.create(user=request.user, deck_name=deck_name)
            # Create each generated flashcard and link it to the deck
            for item in flashcards_data:
                term = item['term']
                definition = item['definition']
                flashcard = Flashcard.objects.create(term=term, definition=definition, user=request.user)
                DeckFlashcard.objects.create(deck=deck, flashcard=flashcard)

            return Response({"status": "success", "deck_id": deck.deck_id,"message": "Flashcards and Deck created successfully." }, status=status.HTTP_201_CREATED)
        except json.JSONDecodeError:
            return Response({"status": "error", "message": "Could not decode flashcards data from the API."}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"status": "error", "message": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

    def delete_deck(self, request, deck_id):
        deck = get_object_or_404(Deck, pk=deck_id)
        if request.user == deck.user:
            deck.delete()
            return Response({"status": "success", "message": "Deck deleted successfully."}, status=status.HTTP_200_OK)
        else:
            return Response({"status": "error", "message": "You do not have permission to delete this deck."}, status=status.HTTP_403_FORBIDDEN)
    
    def update_deck(self, request, deck_id, deck_name):
        try:
            deck = get_object_or_404(Deck, pk=deck_id)
            if request.user == deck.user:
                deck.deck_name = deck_name
                deck.save()
                return Response({"status": "success", "message": "Deck name updated successfully."}, status=status.HTTP_200_OK)
            else:
                return Response({"status": "error", "message": "You do not have permission to update this deck."}, status=status.HTTP_403_FORBIDDEN)
        except Exception as e:
            return Response({"status": "error", "message": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
    @csrf_exempt
    def get_deck(self, request, deck_id):
        try:
            deck = get_object_or_404(Deck, deck_id=deck_id)
            
            if request.user != deck.user:
                return Response({"status": "error", "message": "You do not have permission to view this deck."}, status=status.HTTP_403_FORBIDDEN)
            
            deck_flashcards = DeckFlashcard.objects.filter(deck=deck).order_by('position')
            serialized_deck = DeckSerializer(deck).data
            serialized_flashcards = DeckFlashcardSerializer(deck_flashcards, many=True).data
            
            return Response({
                "status": "success",
                "deck": serialized_deck,
                "flashcards": [fc['flashcard'] for fc in serialized_flashcards]
            }, status=status.HTTP_200_OK)
        
        except Exception as e:
            return Response({"status": "error", "message": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class FlashcardView(APIView):
    authentication_classes = (JWTAuthentication,)
    permission_classes = (IsAuthenticated,)

    @csrf_exempt
    def post(self, request, action):
        if action == 'create':
            return self.create_flashcard(request)
        elif action == 'delete':
            flashcard_id = request.data.get('flashcard_id')
            if flashcard_id:
                return self.delete_flashcard(request, flashcard_id)
        elif action == 'edit':
            flashcard_id = request.data.get('flashcard_id')
            if flashcard_id:
                return self.edit_flashcard(request, flashcard_id)
        
        return Response({"status": "error", "message": "Invalid action."}, status=status.HTTP_400_BAD_REQUEST)

    def create_flashcard(self, request):
        form = FlashcardForm(request.POST)
        if form.is_valid():
            flashcard = form.save(commit=False)
            flashcard.user = request.user  # Authenticated User
            deck_id = request.POST.get('deck_id')  # Get the deck associated with the flashcard

            try:
                deck = Deck.objects.get(pk=deck_id, user=request.user)  # Make sure the deck belongs to the user
            except Deck.DoesNotExist:
                return Response({"status": "error", "message": "Deck not found or you don't have permission to add to this deck."}, status=status.HTTP_400_BAD_REQUEST)

            flashcard.save()
            DeckFlashcard.objects.create(deck=deck, flashcard=flashcard)

            return Response({"status": "success", "message": "Flashcard added to the deck successfully."}, status=status.HTTP_201_CREATED)
        else:
            return Response({"status": "error", "message": form.errors}, status=status.HTTP_400_BAD_REQUEST)

    def delete_flashcard(self, request, flashcard_id):
        flashcard = get_object_or_404(Flashcard, pk=flashcard_id)
        if request.user == flashcard.user:
            flashcard.delete()
            return Response({"status": "success", "message": "Flashcard deleted successfully."}, status=status.HTTP_200_OK)
        else:
            return Response({"status": "error", "message": "You do not have permission to delete this flashcard."}, status=status.HTTP_403_FORBIDDEN)

    def edit_flashcard(self, request, flashcard_id):
        try:
            flashcard = Flashcard.objects.get(pk=flashcard_id, user=request.user)  # Assuming 'user' is the user field in your Flashcard model
        except Flashcard.DoesNotExist:
            return Response({"status": "error", "message": "Flashcard not found or you don't have permission to edit it."}, status=status.HTTP_400_BAD_REQUEST)

        form = FlashcardForm(request.POST, instance=flashcard)  # Populate the form with the flashcard data
        if form.is_valid():
            updated_flashcard = form.save()
            return Response({"status": "success", "message": "Flashcard updated successfully."}, status=status.HTTP_200_OK)
        else:
            return Response({"status": "error", "message": form.errors}, status=status.HTTP_400_BAD_REQUEST)


# Helper functions

def username_exists(username):
    return User.objects.filter(username=username).exists()


def parse_request_body(request):
    try:
        return json.loads(request.body.decode('utf-8'))
    except json.JSONDecodeError:
        return None
    
def create_jwt_with_user_info(user):
    refresh = RefreshToken.for_user(user)

    refresh['user_id'] = user.id

    return str(refresh.access_token)


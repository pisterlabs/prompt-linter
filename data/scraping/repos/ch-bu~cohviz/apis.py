# encoding: utf-8

import random
import string
import json

from Levenshtein import distance

from langdetect import detect

from django.contrib.auth.models import User
from django.http import HttpResponse
from django.http import JsonResponse


from rest_framework.renderers import JSONRenderer
from rest_framework.parsers import JSONParser
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAdminUser
from rest_framework.permissions import AllowAny
from rest_framework.authentication import SessionAuthentication
from rest_framework.authentication import BasicAuthentication

from coherenceanalyzer.coherenceanalyzer import analyzeTextCohesion

from cohapp import constants
from cohapp.models import Experiment, Measurement, Group, Subject
from cohapp.serializers import ExperimentSerializer
from cohapp.serializers import MeasurementSerializer
from cohapp.serializers import SubjectSerializer
from cohapp.serializers import GroupSerializer
from cohapp.serializers import TextDataSerializer

# Load language models
from languagemodels import analyzer_english


# ======================= Helper Classes =================================
class JSONResponse(HttpResponse):

    """
    An HttpResponse that renders its content into JSON
    """

    def __init__(self, data, **kwargs):
        content = JSONRenderer().render(data)
        kwargs['content_type'] = 'application/json'
        super(JSONResponse, self).__init__(content, **kwargs)


class CsrfExemptSessionAuthentication(SessionAuthentication):

    """
    Overrides SessionAuthentication enforce_csrf
    method. With this Class no csrf token
    is required
    """

    def enforce_csrf(self, request):
        return  # To not perform the csrf check previously happening


class MeasurementView(APIView):

    """
    API for measurements of a given experiment.
            GET: Returns all measurements for a given experiment.
            POST: Saves measurements for a given experiment.
    """

    authentication_classes = (CsrfExemptSessionAuthentication,
                              BasicAuthentication)
    permission_classes = (AllowAny,)

    def get(self, request, experiment_password):

        # Filter measurements for a given experiment
        experiment = Experiment.objects.get(master_pw=experiment_password).id
        measurements = Measurement.objects.filter(
            experiment_id=experiment)

        # Translate models into python native datatypes
        serializer = MeasurementSerializer(measurements, many=True)

        # Convert serializer to JSON and return JSONResponse
        return Response(serializer.data, status=200)

    def post(self, request, experiment_password):

        # Get current experiment
        experiment = Experiment.objects.get(master_pw=experiment_password).id

        # Stream request into python native datatype
        data = JSONParser().parse(request)

        # Get group for measurement
        treatment = data['group']
        group = Group.objects.get(abbreviation=treatment).id

        # Overwrite keys
        data['experiment'] = experiment
        data['group'] = group

        # Translate models into python native datatypes
        serializer = MeasurementSerializer(data=data)

        # Check if serializer is valid
        if serializer.is_valid():
            serializer.save()

            return Response(serializer.data, status=201)

        # Serializer is not valid
        return Response(serializer.errors, status=400)


class CognitiveLoadRevisionView(APIView):
    """
    API to store the cognitive load measure after
    the revison of a text.

        POST: Stores cognitive load items from revision
    """

    authentication_classes = (CsrfExemptSessionAuthentication,
        BasicAuthentication)
    permission_classes = (AllowAny, )

    def post(self, request, experiment_password):
        # Try to get the experiment
        try:
            experiment = Experiment.objects.get(master_pw=experiment_password)
        except Experiment.DoesNotExist:
            return Response({'description': 'Experiment could not be found.'}, status=404)

        # Try to get the user from the request
        try:
            user = User.objects.get(username=request.user)
            subject = Subject.objects.get(experiment=experiment, user=user.id)
        # User does not exist
        except Subject.DoesNotExist:
            return Response({'description': 'User could not be found.'}, status=404)

        # Get data from request
        data = request.data
        data['subject'] = subject.id
        data['experiment'] = experiment.id
        data['measurement'] = subject.nr_measurements

        # Serialize data
        serializer = CognitiveLoadRevisionSerializer(data=data)


        if serializer.is_valid():
            serializer.save()

            return Response(serializer.data, status=201)

        return Response(serializer.errors, status=400)


class CognitiveLoadDraftView(APIView):
    """
    API to store the cognitive load measure after
    the revison of a text.

        POST: Stores cognitive load items from draft
    """

    authentication_classes = (CsrfExemptSessionAuthentication,
        BasicAuthentication)
    permission_classes = (AllowAny, )

    def post(self, request, experiment_password):
        # Try to get the experiment
        try:
            experiment = Experiment.objects.get(master_pw=experiment_password)
        except Experiment.DoesNotExist:
            return Response({'description': 'Experiment could not be found.'}, status=404)

        # Try to get the user from the request
        try:
            user = User.objects.get(username=request.user)
            subject = Subject.objects.get(experiment=experiment, user=user.id)
        # User does not exist
        except Subject.DoesNotExist:
            return Response({'description': 'User could not be found.'}, status=404)

        # Get data from request
        data = request.data
        data['subject'] = subject.id
        data['experiment'] = experiment.id
        data['measurement'] = subject.nr_measurements

        # Serialize data
        serializer = CognitiveLoadDraftSerializer(data=data)


        if serializer.is_valid():
            serializer.save()

            return Response(serializer.data, status=201)

        return Response(serializer.errors, status=400)


class SingleExperimentView(APIView):

    """
    API for a single experiment.
            GET: Returns data for a single experiment.
    """

    authentication_classes = (CsrfExemptSessionAuthentication,
                              BasicAuthentication)
    permission_classes = (IsAdminUser,)

    def get(self, request, experiment_password):

        try:
            experiment = Experiment.objects.get(master_pw=experiment_password,
                                                experimentator=request.user)
        except Experiment.DoesNotExist:
            return Response({}, status=400)

        # Translate model into python native datatype
        serializer = ExperimentSerializer(experiment)

        # Convert serializer to JSON and return JSONResponse
        return Response(serializer.data, status=200)


class UserExperimentView(APIView):

    """
    API for all users.
            GET:  Returns all subjects for a given experiment.
            POST: Generates a specific number of users for a given experiment.
    """

    authentication_classes = (CsrfExemptSessionAuthentication,
                              BasicAuthentication)
    permission_classes = (IsAdminUser,)

    def get(self, request, experiment_password):

        # Check if experiment exists
        try:
            experiment = Experiment.objects.get(master_pw=experiment_password,
                                                experimentator=request.user)
        except Experiment.DoesNotExist:
            return Response({}, status=404)

        # Get users for given experiment
        subjects = Subject.objects.filter(experiment=experiment)

        serializer = SubjectSerializer(subjects, many=True)

        return Response(serializer.data, status=200)

    def post(self, request, experiment_password):

        # Stream request into python native datatype
        nr_users = JSONParser().parse(request)['nr_users']

        # Loop over number of users to be generated
        for user in range(int(nr_users)):

            # Generate random username
            username = ''.join(random.choice(string.lowercase) for x in
                               range(6))
            # Generate random password
            password = User.objects.make_random_password()[1:7]
            subject = Subject()

            # Create new subject
            subject.generate_user(username, password, experiment_password)

        return Response({}, status=201)


class UserSpecificView(APIView):

    """
    API for a specific user
            GET: Returns information of a specific user.
    """

    authentication_classes = (CsrfExemptSessionAuthentication,
                              BasicAuthentication)
    permission_classes = (AllowAny,)

    # def get_subject(self, user_name, experiment_id):

    #     # Check if user exists
    #     try:
    #         user = User.objects.get(username=user_name)
    #         subject = Subject.objects.get(experiment=experiment_id,
    #                                       user=user)
    #         return subject
    #     # User does not exist
    #     except Subject.DoesNotExist:
    #         return Response({}, status=404)

    def get(self, request, experiment_password):

        # Get experiment
        try:
            experiment = Experiment.objects.get(master_pw=experiment_password)
        except Experiment.DoesNotExist:
            return Response({}, status=404)

        # Get user from request
        try:
            subject = Subject.objects.get(experiment=experiment,
                                          user=request.user)
        # User does not exist
        except Subject.DoesNotExist:
            return Response({}, status=404)

        serializer = SubjectSerializer(subject)
        return Response(serializer.data)


class UserSpecificNameView(APIView):

    """
    API for a specific user
            GET: Returns information of a specific user with a name.
    """

    authentication_classes = (CsrfExemptSessionAuthentication,
                              BasicAuthentication)
    permission_classes = (AllowAny,)

    def get(self, request, user_name, experiment_id):

        # Get user from request
        try:
            user = User.objects.get(username=user_name)
            subject = Subject.objects.get(experiment=experiment_id,
                                          user=user)
        # User does not exist
        except Subject.DoesNotExist:
            return Response({}, status=404)

        serializer = SubjectSerializer(subject, many=True)
        return Response(serializer.data)


class ExperimentView(APIView):

    """
    API for all experiments
            GET: Returns all experiments from a given experimentator.
            POST: Writes a new experiment into the database.
    """

    authentication_classes = (CsrfExemptSessionAuthentication,
                              BasicAuthentication)
    permission_classes = (IsAdminUser,)

    def get(self, request):

        # Filter all experiments for logged in user
        experiments = Experiment.objects.filter(experimentator=request.user).\
            order_by('-date')

        # Translate models into python native datatypes
        serializer = ExperimentSerializer(experiments, many=True)

        # Convert serializer to JSON and return JSONResponse
        return Response(serializer.data, status=200)

    def post(self, request):

        data = request.data

        # Add experimentator to data -> field is required
        data['experimentator'] = request.user.id

        # Generate passwort for experiment
        data['master_pw'] = User.objects.make_random_password(40,
                'abcdefghjkmnpqrstuvwxyz23456789')

        serializer = ExperimentSerializer(data=data)

        if serializer.is_valid():

            serializer.save()

            return Response(serializer.data, status=201)

        return Response(serializer.errors, status=400)


class TextDataView(APIView):

    """
    API for text data
        POST: Writes text data to database and increment count on
              measurement number for subject.
    """

    authentication_classes = (CsrfExemptSessionAuthentication,
                              BasicAuthentication)
    permission_classes = (AllowAny,)

    def post(self, request, experiment_password):

        # Get data from request
        data = request.data

        # Subject exists
        try:
            subject = Subject.objects.get(user=request.user)
        # Subject does not exist
        except Subject.DoesNotExist:
            # TODO Error message
            return Response({'error': 'An error'},
                            status=400)

        # Add data for serializer
        try:
            experiment = Experiment.objects.get(master_pw=experiment_password)
        except Experiment.DoesNotExist:
            return Response({'error': 'Experiment does not exist'}, status=404)

        # Get current measure
        try:
            measure = Measurement.objects.get(
                experiment=experiment, nr_group=subject.group,
                measure=subject.nr_measurements + 1)
        # Measure does not exist
        except Measurement.DoesNotExist:
            # Render error message to user
            # TODO: error message
            return Response({'error': 'An error'},
                            status=400)

        # Add serializer data
        data['subject'] = subject.id
        data['group'] = measure.group.id
        data['measurement'] = measure.id
        data['experiment'] = experiment.id
        data['levenshtein_distance'] = distance(
            data['pre_text'], data['post_text'])

        # Serialize data
        serializer = TextDataSerializer(data=data)

        # Check if serializer is valid
        if serializer.is_valid():
            serializer.save()

            # Increment measurement number for subject
            subject.nr_measurements += 1
            subject.save()

            # Return success message
            return Response(serializer.data, status=201)

        # Serializer is not valid
        return Response(serializer.data, status=400)


class RegistrationView(APIView):

    """
    API for user registration
            POST: Checks if user data is valid, if yes changes
                      password and returns success JsonResponse
    """

    authentication_classes = (CsrfExemptSessionAuthentication,
                              BasicAuthentication)
    permission_classes = (AllowAny,)

    def post(self, request, user_name, experiment_id):

        data = request.data

        # Check if experiment exists
        try:
            experiment = Experiment.objects.get(id=int(experiment_id),
                                                master_pw=data['master_pw'])
        except Experiment.DoesNotExist:
            return Response({'error': constants.errors['wrong_master']},
                            status=400)

        # Check if user exists
        try:
            user = User.objects.get(username=data['username'])
        # User does not exist
        except User.DoesNotExist:
            return Response({'error': constants.errors['user_does_not_exist']},
                            status=400)

        # User has already logged in
        if user.last_login is not None:
            return Response({'error': constants.errors['already_logged_in']},
                            status=400)

        # Check if user is registered for experiment
        try:
            subject = Subject.objects.get(user=user, experiment=experiment)
        # User is not registered for this experiment
        except Subject.DoesNotExist:
            return Response({'error':
                             constants.errors['user_wrong_experiment']},
                            status=400)

        # Neues Passwort speichern
        user.set_password(data['pwd'])
        user.save()

        return Response({'success': constants.success['password_saved']},
                        status=201)


class GroupView(APIView):

    """
    API for all groups
            GET: Returns all groups with name, description, and abbreviation.
    """

    authentication_classes = (CsrfExemptSessionAuthentication,
                              BasicAuthentication)
    permission_classes = (IsAdminUser,)

    def get(self, request):

        groups = Group.objects.all()

        # Translate models into python native datatypes
        serializer = GroupSerializer(groups, many=True)

        # Convert serializer to JSON and return JSONResponse
        return Response(serializer.data, status=200)


class TextAnalyzer(APIView):

    """
    API for text analysis
        POST: Returns a json object with data from analyzed text.
    """

    authentication_classes = (CsrfExemptSessionAuthentication,
                              BasicAuthentication)
    permission_classes = (AllowAny, )

    def post(self, request):

        # Get text from post data
        text = request.data['text']

        # Text is empty
        if not text.strip():
            return JsonResponse({}, status=500)
        # Text is not empty
        else:
            text_language = detect(text)
            # Detect language
            if text_language == 'en':
                print '**** Englisch *****'
                # Analyze english text
                results = analyzer_english.get_data_for_visualization(text)
            elif text_language == 'de':
                print '**** German *****'
                # Analyze german text
                results = analyzer = analyzeTextCohesion(text)
            else:
                return JsonResponse({}, status=500)

        return JsonResponse(results, status=200)

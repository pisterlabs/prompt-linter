from django import forms

from guidances.models import Guidance


class GuidanceRegisterForm(forms.Form):
    class Meta:
        model = Guidance
        fields = ['project_title', 'student', 'advisor', 'coordination']
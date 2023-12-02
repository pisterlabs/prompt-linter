import openai
import pandas as pd
import io

from django import forms
from django.conf import settings
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.core.validators import RegexValidator
from django.core.exceptions import ValidationError

from .models import UserProfile

validate_openai_api_key_format = RegexValidator(
    regex=r"^sk-[a-zA-Z0-9]{32,}$",
    message="The provided OpenAI API key does not have the correct format. Please enter a key with the correct format.",
    code="invalid_api_key_format",
)


def validate_openai_api_key_usage(value):
    """Validate the OpenAI API key by making a test request to the API."""
    try:
        openai.api_key = value
        openai.Completion.create(
            engine="davinci",
            prompt="This is a test",
            temperature=0,
            max_tokens=5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"],
        )
    except openai.error.RateLimitError as e:
        raise ValidationError(
            message="The provided OpenAI API key is not valid because of the rate limit. Please enter a valid key.",
            code="invalid_api_key_usage",
        )
    except openai.error.AuthenticationError as e:
        raise ValidationError(
            message="The provided OpenAI API key is not valid. Please enter a valid key.",
            code="invalid_api_key_usage",
        )


class NewUserForm(UserCreationForm):
    openai_api_key = forms.CharField(
        min_length=32,
        max_length=100,
        required=True,
        label="OpenAI API Key",
        validators=[validate_openai_api_key_format, validate_openai_api_key_usage],
        help_text="Please enter your OpenAI API key. If you do not have one, please visit <a href='https://beta.openai.com/'>https://beta.openai.com/</a> to get one.",
        widget=forms.PasswordInput(),
    )

    class Meta:
        model = User
        fields = ["email", "username", "password1", "password2", "openai_api_key"]

    def save(self, commit=True):
        user = super(NewUserForm, self).save(commit=False)
        if commit:
            user.save()
            # User profile is automatically created
            userprofile = UserProfile.objects.get(user=user)
            # Save OpenAI API key
            userprofile.openai_api_key = self.cleaned_data["openai_api_key"]
            userprofile.save()
        return user


def validate_csv_format(value):
    """Validate the CSV file by checking if it can be opened with pandas.read_csv (Only first 5 rows)."""

    if not value.name.endswith(".csv"):
        raise ValidationError(
            message="The provided file is not a CSV file.",
            code="invalid_csv_format",
        )

    try:
        if value.multiple_chunks():
            pd.read_csv(value.temporary_file_path(), nrows=5)
        else:
            pd.read_csv(io.BytesIO(next(value.chunks())), nrows=5)
    except Exception as e:
        raise ValidationError(
            message="The provided CSV file is too large.",
            code="invalid_csv_size",
        )


def validate_csv_size(value):
    if value.size > settings.CSV_UPLOAD_MAX_SIZE:
        raise ValidationError(
            message="The provided CSV file is too large.",
            code="invalid_csv_size",
        )


class UploadFileForm(forms.Form):
    """
    Form to upload a CSV, Open API key, and a text prompt
    """

    file = forms.FileField(validators=[validate_csv_size, validate_csv_format])

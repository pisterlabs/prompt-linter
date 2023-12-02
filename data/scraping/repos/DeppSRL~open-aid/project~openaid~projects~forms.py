from django import forms
from django.utils.text import slugify
from django.utils.translation import ugettext_lazy as _
from haystack.forms import FacetedSearchForm
from openaid.projects.models import Project, Markers, Activity, ChannelReported


def text_cleaner(text):
    # http://stackoverflow.com/questions/2077897/substitute-multiple-whitespace-with-single-whitespace-in-python
    return u' '.join(text.split()).strip(u' ')


class ProjectForm(forms.ModelForm):
    class Meta:
        model = Project


class ActivityForm(forms.ModelForm):
    def clean_title(self):
        title = self.cleaned_data.get('title')
        return text_cleaner(title)

    def clean_description(self):
        description = self.cleaned_data.get('description')
        return text_cleaner(description)

    def clean_long_description(self):
        long_description = self.cleaned_data.get('long_description')
        return text_cleaner(long_description)

    def clean(self):
        data = super(ActivityForm, self).clean()

        if data['description']:
            if data['long_description'] and slugify(data['description']) == slugify(data['long_description']):
                data['long_description'] = ''

        return data

    class Meta:
        model = Activity


class CodeListForm(forms.Form):
    id = forms.IntegerField(min_value=0)
    name = forms.CharField(max_length=500)


class MarkersForm(forms.ModelForm):
    class Meta:
        model = Markers


class ChannelReportedForm(forms.ModelForm):
    class Meta:
        model = ChannelReported


class FacetedProjectSearchForm(FacetedSearchForm):
    default_order = 'start_year'
    default_desc = True
    order_by = forms.ChoiceField(initial=default_order, required=False, choices=(
        ('start_year', _("Start year")),
        ('end_year', _("End year")),
    ))
    desc = forms.BooleanField(initial=default_desc, required=False)

    def __init__(self, *args, **kwargs):
        super(FacetedProjectSearchForm, self).__init__(*args, **kwargs)

    def search(self):
        sqs = super(FacetedProjectSearchForm, self).search()

        data = {}

        if self.is_valid():
            data = self.cleaned_data

        order_field = data.get('order_by', None) or self.default_order
        is_desc = data.get('desc', self.default_desc)
        if is_desc:
            order_field = '-{0}'.format(order_field)

        return sqs.order_by(order_field)

    def no_query_found(self):
        """
        Retrieve all search results for empty query string
        """
        return self.searchqueryset.all()


class FacetedInitiativeSearchForm(FacetedProjectSearchForm):
    pass

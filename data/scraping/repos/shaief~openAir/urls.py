from django.conf.urls import patterns, include, url
from django.views.generic import TemplateView
from openair.records import views
from tastypie.api import Api
from openair.records.api import StationResource

v0_api = Api(api_name='v0')
v0_api.register(StationResource())

urlpatterns = patterns('',
    url(r'^$', TemplateView.as_view(template_name="records/index.html"),
        name='home'),
    url(r'^parameters/$', views.parameters, name='parameters'),

    url(r'^parameter/(?P<abbr>[-_()a-zA-Z0-9. ]+)/$', views.parameter,
        name='parameter'),
    url(r'^parameter/(?P<abbr>[-_()a-zA-Z0-9. ]+)/json/$', views.parameter_json,
        name='parameter_json'),
    #url(r'^station-(?P<station_id>\d+)-(?P<start>\d+)-to-(?P<end>\d+)$',
    #views.demo_linechart),
    # =====================================================================
    # urls for station view, including jsons:
    url(r'^station/(?P<url_id>[0-9]+)/$',
        views.station, name='station'),
    url(r'^station/(?P<url_id>[0-9]+)/json/$',
        views.station_json, name='station_json'),
    url(r'^station_parameters/(?P<url_id>[0-9]+)/(?P<abbr>[-_ ()a-zA-Z0-9.]+)/$',
        views.station_parameters, name='station_parameters'),
    url(r'^station_parameters/(?P<url_id>[0-9]+)/(?P<abbr>[-_ ()a-zA-Z0-9.]+)/json/$',
        views.station_parameters_json, name='station_parameters_json'),
    url(r'^wind/(?P<zone_url_id>[0-9]+)/(?P<station_url_id>[0-9]+)/$',
        views.wind, name='wind'),
    url(r'^wind/(?P<url_id>[0-9]+)/json/$',
        views.wind_json, name='wind_json'),
    url(r'^dailyparam/json/(?P<url_id>[0-9]+)/(?P<abbr>[-_ ()a-zA-Z0-9.]+)/$',
        views.dailyparam_json, name='dailyparam_json'),
    # =====================================================================
    url(r'^map/$', views.map, name='map'),
    url(r'^indexmap/$',
        TemplateView.as_view(template_name='records/indexmap.html'),
        name='indexmap'),
    url(r'^zones/$', views.zones, name='zones'),
    url(r'^api/', include(v0_api.urls)),
    # =====================================================================
    url(r'^stations/json/$', views.stations_json, name='stations_json'),
)

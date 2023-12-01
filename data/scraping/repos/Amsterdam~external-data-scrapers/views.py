from datapunt_api.rest import DatapuntViewSet
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import OrderingFilter

from ..filters import StadsdeelFilter
from .models import GuidanceSign, ParkingGuidanceDisplay, ParkingLocation
from .serializers import (GuidanceSignSerializer,
                          ParkingGuidanceDisplaySerializer,
                          ParkingLocationSerializer)


class ParkingLocationFilter(StadsdeelFilter):
    class Meta:
        model = ParkingLocation
        fields = [
            'stadsdeel',
            'buurt_code',
            'state',
            'type'
        ]


class ParkingLocationView(DatapuntViewSet):
    queryset = ParkingLocation.objects.all()

    serializer_detail_class = ParkingLocationSerializer
    serializer_class = ParkingLocationSerializer

    filter_backends = (
        DjangoFilterBackend,
        OrderingFilter
    )

    filterset_class = ParkingLocationFilter
    ordering_fields = ('pub_date', 'scraped_at')


class GuidanceSignFilter(StadsdeelFilter):
    class Meta:
        model = GuidanceSign
        fields = [
            'stadsdeel',
            'buurt_code',
            'state',
            'type',
            'removed'
        ]


class GuidanceSignView(DatapuntViewSet):
    queryset = GuidanceSign.objects.all()

    serializer_detail_class = GuidanceSignSerializer
    serializer_class = GuidanceSignSerializer

    filter_backends = (
        DjangoFilterBackend,
        OrderingFilter
    )

    filterset_class = GuidanceSignFilter
    ordering_fields = ('pub_date', 'scraped_at')


class ParkingGuidanceDisplayView(DatapuntViewSet):
    queryset = ParkingGuidanceDisplay.objects.all()

    serializer_detail_class = ParkingGuidanceDisplaySerializer
    serializer_class = ParkingGuidanceDisplaySerializer

    filter_backends = (
        DjangoFilterBackend,
        OrderingFilter
    )

    filter_fields = ['guidance_sign', 'type', ]
    ordering_fields = ('pub_date', 'scraped_at')

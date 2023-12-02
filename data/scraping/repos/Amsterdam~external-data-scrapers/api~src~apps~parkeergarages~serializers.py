from datapunt_api.rest import HALSerializer, RelatedSummaryField
from rest_framework import serializers

from .models import GuidanceSign, ParkingGuidanceDisplay, ParkingLocation


class ParkingLocationSerializer(HALSerializer):
    class Meta:
        model = ParkingLocation
        fields = '__all__'


class ParkingGuidanceDisplaySerializer(serializers.ModelSerializer):
    class Meta:
        model = ParkingGuidanceDisplay
        fields = '__all__'


class GuidanceSignRelatedSummaryField(RelatedSummaryField):
    """
    Since ParkingGuidanceDisplay contains a non primarykey as foreignkey
    of GuidanceSign, we overwrite here the href to add the correct id filter.
    """
    def to_representation(self, value):
        result = super().to_representation(value)
        result['href'] = '{}={}'.format(
            result['href'].split('=')[0], value.instance.api_id
        )
        return result


class GuidanceSignSerializer(HALSerializer):
    displays = GuidanceSignRelatedSummaryField()

    class Meta:
        model = GuidanceSign
        fields = (
            '_links',
            'id',
            'api_id',
            'geometrie',
            'name',
            'type',
            'state',
            'pub_date',
            'removed',
            'stadsdeel',
            'buurt_code',
            'scraped_at',
            'displays',
        )

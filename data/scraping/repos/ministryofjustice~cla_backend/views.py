import datetime
from uuid import UUID

from dateutil import parser
from dateutil.relativedelta import relativedelta
from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.db.models import Q
from django.db import transaction
from django.utils import six, timezone
from django.shortcuts import get_object_or_404

from cla_eventlog import event_registry
from historic.models import CaseArchived
from legalaid.permissions import IsManagerOrMePermission

from rest_framework import viewsets, mixins, status
from rest_framework.decorators import detail_route, list_route
from rest_framework.response import Response as DRFResponse
from rest_framework.filters import OrderingFilter, DjangoFilterBackend, SearchFilter, BaseFilterBackend
import operator
from django.db import models


from cla_provider.models import Provider, OutOfHoursRota, Feedback, ProviderPreAllocation
from cla_eventlog.views import BaseEventViewSet, BaseLogViewSet
from cla_provider.helpers import ProviderAllocationHelper, notify_case_assigned

from core.drf.mixins import FormActionMixin, ClaCreateModelMixin, ClaUpdateModelMixin
from core.drf.viewsets import CompatGenericViewSet
from core.drf.paginator import StandardResultsSetPagination
from notifications.views import BaseNotificationViewSet

from complaints.views import (
    BaseComplaintViewSet,
    BaseComplaintConstantsView,
    BaseComplaintCategoryViewSet,
    BaseComplaintLogViewset,
)

from timer.views import BaseTimerViewSet

from legalaid.models import PersonalDetails, Case
from legalaid.views import (
    BaseUserViewSet,
    BaseCategoryViewSet,
    BaseNestedEligibilityCheckViewSet,
    BaseMatterTypeViewSet,
    BaseMediaCodeViewSet,
    FullPersonalDetailsViewSet,
    BaseThirdPartyDetailsViewSet,
    BaseAdaptationDetailsViewSet,
    BaseAdaptationDetailsMetadataViewSet,
    FullCaseViewSet,
    BaseCaseNotesHistoryViewSet,
    AscCaseOrderingFilter,
    BaseCSVUploadReadOnlyViewSet,
    BaseCaseLogMixin,
    BaseEODDetailsViewSet,
    BaseContactResearchMethodViewSet,
)

from cla_common.constants import REQUIRES_ACTION_BY, CASE_SOURCE
from knowledgebase.views import BaseArticleViewSet, BaseArticleCategoryViewSet
from diagnosis.views import BaseDiagnosisViewSet
from guidance.views import BaseGuidanceNoteViewSet

from .permissions import CallCentreClientIDPermission, OperatorManagerPermission
from .serializers import (
    EligibilityCheckSerializer,
    CaseSerializer,
    ProviderSerializer,
    OutOfHoursRotaSerializer,
    OperatorSerializer,
    AdaptationDetailsSerializer,
    PersonalDetailsSerializer,
    BarePersonalDetailsSerializer,
    ThirdPartyDetailsSerializer,
    LogSerializer,
    FeedbackSerializer,
    CreateCaseSerializer,
    CaseListSerializer,
    CaseArchivedSerializer,
    CaseNotesHistorySerializer,
    CSVUploadSerializer,
    CSVUploadDetailSerializer,
    EODDetailsSerializer,
)

from .forms import (
    ProviderAllocationForm,
    DeclineHelpCaseForm,
    DeferAssignmentCaseForm,
    SuspendCaseForm,
    AlternativeHelpForm,
    CallMeBackForm,
    StopCallMeBackForm,
    DiversityForm,
)

from .models import Operator
from .utils.organisation import CaseOrganisationAssignCurrentOrganisationMixin

from cla_auditlog.models import AuditLog


class CallCentrePermissionsViewSetMixin(object):
    permission_classes = (CallCentreClientIDPermission,)


class CallCentreManagerPermissionsViewSetMixin(object):
    permission_classes = (CallCentreClientIDPermission, OperatorManagerPermission)


class CategoryViewSet(CallCentrePermissionsViewSetMixin, BaseCategoryViewSet):
    pass


class EligibilityCheckViewSet(
    CallCentrePermissionsViewSetMixin,
    ClaCreateModelMixin,
    ClaUpdateModelMixin,
    mixins.RetrieveModelMixin,
    BaseNestedEligibilityCheckViewSet,
):
    serializer_class = EligibilityCheckSerializer

    # this is to fix a stupid thing in DRF where pre_save doesn't call super
    def perform_create(self, serializer):
        original_obj = self.get_object()
        self.__pre_save__ = self.get_serializer_class()(original_obj).data
        super(EligibilityCheckViewSet, self).perform_create(serializer)

    def perform_update(self, serializer):
        original_obj = self.get_object()
        self.__pre_save__ = self.get_serializer_class()(original_obj).data
        super(EligibilityCheckViewSet, self).perform_update(serializer)


class MatterTypeViewSet(CallCentrePermissionsViewSetMixin, BaseMatterTypeViewSet):
    pass


class MediaCodeViewSet(CallCentrePermissionsViewSetMixin, BaseMediaCodeViewSet):
    pass


class ContactResearchMethodViewSet(CallCentrePermissionsViewSetMixin, BaseContactResearchMethodViewSet):
    pass


class DateRangeFilter(BaseFilterBackend):
    def filter_queryset(self, request, qs, view):

        filter = {}
        start_date = request.query_params.get("start", None)
        end_date = request.query_params.get("end", None)

        if start_date is not None:
            filter["{field}__gte".format(field=view.date_range_field)] = parser.parse(start_date).replace(
                tzinfo=timezone.get_current_timezone()
            )
        if end_date is not None:
            filter["{field}__lte".format(field=view.date_range_field)] = parser.parse(end_date).replace(
                tzinfo=timezone.get_current_timezone()
            )

        qs = qs.filter(**filter)
        return qs


class CaseViewSet(
    CallCentrePermissionsViewSetMixin,
    ClaCreateModelMixin,
    BaseCaseLogMixin,
    CaseOrganisationAssignCurrentOrganisationMixin,
    FullCaseViewSet,
):
    serializer_class = CaseListSerializer
    # using CreateCaseSerializer during creation
    serializer_detail_class = CaseSerializer

    queryset = Case.objects.all().select_related("eligibility_check", "personal_details")
    queryset_detail = Case.objects.all().select_related(
        "eligibility_check",
        "personal_details",
        "adaptation_details",
        "matter_type1",
        "matter_type2",
        "eod_details",
        "diagnosis",
        "media_code",
        "eligibility_check__category",
        "created_by",
    )

    filter_backends = (AscCaseOrderingFilter,)

    def get_queryset(self, **kwargs):
        """
        Returns the following:
            all:
                no querystring
            operator:
                only == 'operator'
            eod:
                only == 'eod'
            web:
                only == 'web'
            phone:
                only == 'phone'
        """
        this_operator = get_object_or_404(Operator, user=self.request.user)
        qs = super(CaseViewSet, self).get_queryset(**kwargs)

        only_param = self.request.query_params.get("only")
        if only_param == "my":
            qs = qs.filter(created_by=this_operator.user)
        elif only_param == "eod":
            qs = qs.extra(where=[self.FLAGGED_WITH_EOD_SQL])
        elif only_param == "web":
            qs = qs.filter(source=CASE_SOURCE.WEB)
        elif only_param == "phone":
            qs = qs.filter(source=CASE_SOURCE.PHONE)

        if this_operator.is_cla_superuser_or_manager:
            qs = qs.extra(
                select={
                    "complaint_count": """
                    SELECT COUNT(complaints_complaint.id)
                    FROM complaints_complaint
                    JOIN legalaid_eoddetails
                        ON complaints_complaint.eod_id = legalaid_eoddetails.id
                    WHERE legalaid_case.id = legalaid_eoddetails.case_id
                        AND complaints_complaint.resolved IS NULL
                """
                }
            )
        else:
            qs = qs.extra(select={"complaint_count": "SELECT NULL"})

        return qs

    def get_serializer_class(self):
        # if POST create request => use special Serializer
        #   otherwise use standard one
        if self.request.method == "POST" and not self.kwargs.get("reference"):
            return CreateCaseSerializer
        return super(CaseViewSet, self).get_serializer_class()

    def get_dashboard_qs(self, qs):
        if self.request.user.operator.is_manager:
            action_by_operator = Q(requires_action_by=REQUIRES_ACTION_BY.OPERATOR)
            action_by_operator_manager = Q(requires_action_by=REQUIRES_ACTION_BY.OPERATOR_MANAGER)
            qs = qs.filter(action_by_operator | action_by_operator_manager)
        else:
            qs = qs.filter(requires_action_by=REQUIRES_ACTION_BY.OPERATOR)

        qs = qs.filter(Q(requires_action_at__isnull=True) | Q(requires_action_at__lte=timezone.now()))

        return qs

    def perform_create(self, serializer):
        user = self.request.user
        if not isinstance(user, AnonymousUser):
            serializer.validated_data["created_by"] = user
        return super(CaseViewSet, self).perform_create(serializer)

    def retrieve(self, request, *args, **kwargs):
        response = super(CaseViewSet, self).retrieve(request, *args, **kwargs)
        self.instance.audit_log.add(AuditLog.objects.create(user=request.user, action=AuditLog.ACTIONS.VIEWED))
        return response

    @list_route()
    def future_callbacks(self, request, **kwargs):
        """
        Returns a list of callback cases between start_of_day and
            start_of_day + 7 days (excluded)
        """
        now = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        in_7_days = now + datetime.timedelta(days=7)
        qs = (
            self.get_queryset()
            .filter(requires_action_at__gte=now, requires_action_at__lt=in_7_days)
            .order_by("requires_action_at")
        )
        self.object_list = self.filter_queryset(qs)

        serializer = self.get_serializer(self.object_list, many=True)

        return DRFResponse(serializer.data)

    @detail_route()
    def assign_suggest(self, request, reference=None, **kwargs):
        """
        @return: dict - 'suggested_provider' (single item) ;
                        'suitable_providers' all possible providers for this category.
        """

        as_of = None
        if "as_of" in request.GET and (settings.DEBUG or settings.TEST_MODE):
            as_of = parser.parse(request.GET.get("as_of"))
            as_of = as_of.replace(tzinfo=timezone.get_current_timezone())

        obj = self.get_object()
        helper = ProviderAllocationHelper(as_of=as_of)

        if hasattr(obj, "eligibility_check") and obj.eligibility_check is not None and obj.eligibility_check.category:
            category = obj.eligibility_check.category
            ProviderPreAllocation.objects.clear(case=obj)
            suggested = helper.get_suggested_provider(category)

            if suggested:
                suggested_provider = ProviderSerializer(suggested).data
                ProviderPreAllocation.objects.pre_allocate(category, suggested, obj)
            else:
                suggested_provider = None
        else:
            category = None
            suggested_provider = None

        suitable_providers = [ProviderSerializer(p).data for p in helper.get_qualifying_providers(category)]
        suggestions = {
            "suggested_provider": suggested_provider,
            "suitable_providers": suitable_providers,
            "as_of": helper.as_of,
        }

        return DRFResponse(suggestions)

    @detail_route(methods=["post"])
    def assign(self, request, reference=None, **kwargs):
        """
        Assigns the case to a provider
        """
        obj = self.get_object()
        helper = ProviderAllocationHelper()

        category = obj.eligibility_check.category if obj.eligibility_check else None
        suitable_providers = helper.get_qualifying_providers(category)

        # find given provider in suitable - avoid extra lookup and ensures
        # valid provider
        for sp in suitable_providers:
            if sp.id == int(request.data["provider_id"]):
                p = sp
                break
        else:
            raise ValueError("Provider not found")

        # if we're inside office hours then:
        # Randomly assign to provider who offers this category of service
        # else it should be the on duty provider
        data = request.data.copy()
        data["provider"] = p.pk
        form = ProviderAllocationForm(case=obj, data=data, providers=suitable_providers)

        if form.is_valid():
            provider = form.save(request.user)

            ProviderPreAllocation.objects.clear(case=obj)
            notify_case_assigned(provider, form.case)

            provider_serialised = ProviderSerializer(provider)
            self.set_case_organisation(self.get_object())
            return DRFResponse(data=provider_serialised.data)

        return DRFResponse(dict(form.errors), status=status.HTTP_400_BAD_REQUEST)

    @detail_route(methods=["post"])
    def defer_assignment(self, request, **kwargs):
        obj = self.get_object()
        form = DeferAssignmentCaseForm(case=obj, data=request.data)
        if form.is_valid():
            form.save(request.user)
            return DRFResponse(status=status.HTTP_204_NO_CONTENT)

        return DRFResponse(dict(form.errors), status=status.HTTP_400_BAD_REQUEST)

    @detail_route(methods=["post"])
    def decline_help(self, request, reference=None, **kwargs):
        response = self._form_action(request, DeclineHelpCaseForm)
        self.set_case_organisation(self.get_object())
        return response

    @detail_route(methods=["post"])
    def suspend(self, request, reference=None, **kwargs):
        response = self._form_action(request, SuspendCaseForm)
        self.set_case_organisation(self.get_object())
        return response

    @detail_route(methods=["post"])
    def assign_alternative_help(self, request, reference=None, **kwargs):
        response = self._form_action(request, AlternativeHelpForm)
        self.set_case_organisation(self.get_object())
        return response

    def get_log_notes(self, obj):
        return "Case created"

    @detail_route(methods=["get"])
    def search_for_personal_details(self, request, reference=None, **kwargs):
        """
            You can only call this endpoint if the case doesn't have any
            personal_details record attached.
            This is by design as it feels slighly more secure than allowing
            clients to use a dedicated endpoint that they can call whenever
            they want.

            If things change in the future, feel free to add a dedicated
            endpoint for this though.

            Should return just ('reference', 'full_name', 'postcode', 'dob')
            and should NOT include vulnerable users.
        """
        obj = self.get_object()
        if obj.personal_details:
            return DRFResponse(
                {"error": "This case is already linked to a Person"}, status=status.HTTP_400_BAD_REQUEST
            )

        person_q = request.query_params.get("person_q", "") or ""
        if len(person_q) >= 3:
            users = PersonalDetails.objects.filter(full_name__icontains=person_q).exclude(vulnerable_user=True)
        else:
            users = []
        data = [BarePersonalDetailsSerializer(user).data for user in users]

        return DRFResponse(data)

    @detail_route(methods=["post"])
    def link_personal_details(self, request, reference=None, **kwargs):
        """
        * if not DATA.personal_details => return 400
        * if obj.personal_details != None => return 400
        * if personal_details does not exist => return 400
        * else link personal details
        """

        def error_response(msg):
            return DRFResponse({"error": msg}, status=status.HTTP_400_BAD_REQUEST)

        obj = self.get_object()

        # check PARAM exists
        pd_ref = request.data.get("personal_details", None)
        if not pd_ref:
            return error_response('Param "personal_details" required')

        # check that case doesn't have personal_details
        if obj.personal_details:
            return error_response("A person is already linked to this case")

        # check that personal details exists
        try:
            pd_ref = UUID(pd_ref, version=4)

            personal_details = PersonalDetails.objects.get(reference=pd_ref)
        except (ValueError, PersonalDetails.DoesNotExist):
            return error_response('Person with reference "%s" not found' % pd_ref)

        # link personal details to case
        obj.personal_details = personal_details
        obj.save(update_fields=["personal_details", "modified"])
        self.set_case_organisation(self.get_object())
        return DRFResponse(status=status.HTTP_204_NO_CONTENT)

    @detail_route(methods=["post"])
    def call_me_back(self, request, reference=None, **kwargs):
        response = self._form_action(request, CallMeBackForm)
        self.set_case_organisation(self.get_object())
        return response

    @detail_route(methods=["post"])
    def stop_call_me_back(self, request, reference=None, **kwargs):
        response = self._form_action(request, StopCallMeBackForm)
        self.set_case_organisation(self.get_object())
        return response

    @detail_route(methods=["post"])
    def start_call(self, request, reference=None, **kwargs):
        obj = self.get_object()
        event = event_registry.get_event("case")()
        event.process(obj, status="call_started", created_by=request.user, notes="Call started")
        self.set_case_organisation(obj)
        return DRFResponse(status=status.HTTP_204_NO_CONTENT)


class ProviderViewSet(CallCentrePermissionsViewSetMixin, viewsets.ReadOnlyModelViewSet):
    model = Provider
    serializer_class = ProviderSerializer

    queryset = Provider.objects.active()

    filter_backends = (DjangoFilterBackend,)
    filter_fields = ("law_category__code",)


class OutOfHoursRotaViewSet(
    CallCentreManagerPermissionsViewSetMixin,
    ClaCreateModelMixin,
    ClaUpdateModelMixin,
    mixins.RetrieveModelMixin,
    mixins.DestroyModelMixin,
    mixins.ListModelMixin,
    CompatGenericViewSet,
):
    model = OutOfHoursRota
    serializer_class = OutOfHoursRotaSerializer
    queryset = OutOfHoursRota.objects.all()


class UserViewSet(CallCentrePermissionsViewSetMixin, BaseUserViewSet):
    model = Operator
    queryset = Operator.objects.all()

    permission_classes = (CallCentreClientIDPermission, IsManagerOrMePermission)
    serializer_class = OperatorSerializer

    filter_backends = (DjangoFilterBackend,)
    filter_fields = ("is_manager",)

    def get_queryset(self):
        qs = super(BaseUserViewSet, self).get_queryset()
        operator = self.get_logged_in_user_model()
        if operator.organisation:
            query = Q(organisation__isnull=True)
            query.add(Q(organisation=operator.organisation.id), Q.OR)
            qs = qs.filter(query)
        return qs

    def get_logged_in_user_model(self):
        return self.request.user.operator

    @transaction.atomic
    def create(self, request, *args, **kwargs):
        operator = self.get_logged_in_user_model()
        if operator.organisation:
            request.data["organisation"] = operator.organisation.id
        return super(UserViewSet, self).create(request, *args, **kwargs)


class PersonalDetailsViewSet(
    CallCentrePermissionsViewSetMixin,
    FormActionMixin,
    CaseOrganisationAssignCurrentOrganisationMixin,
    FullPersonalDetailsViewSet,
):
    serializer_class = PersonalDetailsSerializer

    @detail_route(methods=["post"])
    def set_diversity(self, request, reference=None, **kwargs):
        return self._form_action(request, DiversityForm)


class ThirdPartyDetailsViewSet(
    CallCentrePermissionsViewSetMixin, CaseOrganisationAssignCurrentOrganisationMixin, BaseThirdPartyDetailsViewSet
):
    serializer_class = ThirdPartyDetailsSerializer


class AdaptationDetailsViewSet(
    CallCentrePermissionsViewSetMixin, CaseOrganisationAssignCurrentOrganisationMixin, BaseAdaptationDetailsViewSet
):
    serializer_class = AdaptationDetailsSerializer


class AdaptationDetailsMetadataViewSet(CallCentrePermissionsViewSetMixin, BaseAdaptationDetailsMetadataViewSet):
    serializer_class = AdaptationDetailsSerializer


class EODDetailsViewSet(
    CallCentrePermissionsViewSetMixin, CaseOrganisationAssignCurrentOrganisationMixin, BaseEODDetailsViewSet
):
    serializer_class = EODDetailsSerializer


class EventViewSet(CallCentrePermissionsViewSetMixin, BaseEventViewSet):
    pass


class ArticleViewSet(CallCentrePermissionsViewSetMixin, BaseArticleViewSet):
    pass


class ArticleCategoryViewSet(CallCentrePermissionsViewSetMixin, BaseArticleCategoryViewSet):
    pass


class TimerViewSet(CallCentrePermissionsViewSetMixin, BaseTimerViewSet):
    pass


class DiagnosisViewSet(
    CallCentrePermissionsViewSetMixin, CaseOrganisationAssignCurrentOrganisationMixin, BaseDiagnosisViewSet
):
    pass


class LogViewSet(CallCentrePermissionsViewSetMixin, BaseLogViewSet):
    serializer_class = LogSerializer


class FeedbackViewSet(
    CallCentreManagerPermissionsViewSetMixin,
    mixins.ListModelMixin,
    ClaUpdateModelMixin,
    mixins.RetrieveModelMixin,
    CompatGenericViewSet,
):
    model = Feedback
    lookup_field = "reference"
    serializer_class = FeedbackSerializer

    filter_backends = (DjangoFilterBackend, OrderingFilter, DateRangeFilter)
    ordering = ("resolved", "-created")
    date_range_field = "created"
    filter_fields = ("resolved",)

    queryset = Feedback.objects.all().select_related("case", "created_by", "created_by__provider")

    pagination_class = StandardResultsSetPagination


class CaseArchivedSearchFilter(SearchFilter):
    def get_search_terms(self, request):
        terms = super(CaseArchivedSearchFilter, self).get_search_terms(request)
        return [term.upper() for term in terms]

    def construct_search(self, field_name):
        return "%s__contains" % field_name


class CaseArchivedViewSet(
    CallCentrePermissionsViewSetMixin, mixins.ListModelMixin, mixins.RetrieveModelMixin, CompatGenericViewSet
):

    lookup_field = "laa_reference"
    model = CaseArchived
    queryset = CaseArchived.objects.all()
    serializer_class = CaseArchivedSerializer

    search_fields = ["search_field"]

    filter_backends = (CaseArchivedSearchFilter,)
    pagination_class = StandardResultsSetPagination


class CaseNotesHistoryViewSet(CallCentrePermissionsViewSetMixin, BaseCaseNotesHistoryViewSet):
    serializer_class = CaseNotesHistorySerializer


class CSVUploadViewSet(CallCentreManagerPermissionsViewSetMixin, BaseCSVUploadReadOnlyViewSet):

    serializer_class = CSVUploadSerializer
    serializer_detail_class = CSVUploadDetailSerializer

    filter_backends = (DjangoFilterBackend, OrderingFilter)
    ordering = ("-month",)
    filter_fields = ("month", "provider_id")

    pagination_class = StandardResultsSetPagination

    def get_queryset(self, *args, **kwargs):
        # only return last 18 months worth
        after = (timezone.now() - relativedelta(months=18)).date().replace(day=1)

        qs = super(CSVUploadViewSet, self).get_queryset(*args, **kwargs).filter(month__gte=after)
        return qs


class GuidanceNoteViewSet(CallCentrePermissionsViewSetMixin, BaseGuidanceNoteViewSet):
    pass


class NotificationViewSet(CallCentrePermissionsViewSetMixin, BaseNotificationViewSet):
    pass


class ComplaintSearchFilter(SearchFilter):
    # LGA-2568 addition of select DISTINCT from DRF 3.0.5 onwards - see release-notes introduced because of this bug
    # https://github.com/encode/django-rest-framework/pull/2535
    # Later versions of DRF change this functionality so will need updating
    def filter_queryset(self, request, queryset, view):
        search_fields = getattr(view, 'search_fields', None)

        search_terms = self.get_search_terms(request)

        if not search_fields or not search_terms:
            return queryset

        orm_lookups = [
            self.construct_search(six.text_type(search_field))
            for search_field in search_fields
        ]

        for search_term in search_terms:
            queries = [
                models.Q(**{orm_lookup: search_term})
                for orm_lookup in orm_lookups
            ]
            queryset = queryset.filter(reduce(operator.or_, queries))
        return queryset


class ComplaintViewSet(
    CallCentrePermissionsViewSetMixin, CaseOrganisationAssignCurrentOrganisationMixin, BaseComplaintViewSet
):
    filter_backends = (DjangoFilterBackend, OrderingFilter, ComplaintSearchFilter)
    filter_fields = ("justified", "level", "category", "owner", "created_by")

    search_fields = (
        "eod__case__personal_details__full_name",
        "eod__case__personal_details__postcode",
        "eod__case__personal_details__street",
        "eod__case__personal_details__search_field",
        "eod__case__reference",
        "eod__case__laa_reference",
    )

    ordering_fields = (
        "created",
        "level",
        "justified",
        "closed",
        "holding_letter",
        "full_letter",
        "category__name",
        "eod__case__reference",
        "eod__case__personal_details__full_name",
    )
    ordering = ("-created",)
    pagination_class = StandardResultsSetPagination

    def get_queryset(self, **kwargs):
        dashboard = self.request.query_params.get("dashboard") == "True"
        show_closed = self.request.query_params.get("show_closed") == "True"
        return super(ComplaintViewSet, self).get_queryset(dashboard=dashboard, show_closed=show_closed)

    def get_case(self, validated_data, obj=None):
        # Cannot always depend on obj as it will be None when a new case is being created
        # when this method gets triggered in a perform_create before the instance is saved
        if "eod" in validated_data:
            return validated_data["eod"].case
        else:
            return obj.eod.case

    def retrieve(self, request, *args, **kwargs):
        response = super(ComplaintViewSet, self).retrieve(request, *args, **kwargs)
        self.instance.audit_log.add(AuditLog.objects.create(user=request.user, action=AuditLog.ACTIONS.VIEWED))
        return response


class ComplaintCategoryViewSet(CallCentrePermissionsViewSetMixin, BaseComplaintCategoryViewSet):
    pass


class ComplaintConstantsView(CallCentrePermissionsViewSetMixin, BaseComplaintConstantsView):
    pass


class ComplaintLogViewset(CallCentrePermissionsViewSetMixin, BaseComplaintLogViewset):
    pass

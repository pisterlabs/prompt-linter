from itertools import chain
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from django.shortcuts import render
from wagtail.models import Page
from wagtail.search.models import Query
from about.models import AboutPage, AboutSubPage, CaseStudyPage, HistoryPage, PeoplePage
from contact.models import ContactPage
from events.models import EventPage
from guidance_and_support.models import GuidanceAndSupportPage, GuidanceGroupPage, GuidancePage
# from guidance_and_support.models import KnowledgebasePage
from news.models import NewsPage
from home.models import AbstractBasePage, StandardPage
from iati_standard.models import IATIStandardPage


class SearchPage(AbstractBasePage):
    """A model for a seach page, to respond to query requests."""

    class Meta:
        verbose_name = 'Search'

    parent_page_types = ['home.HomePage']
    subpage_types = []

    max_count = 1

    def get_paginated(self, collection, page: int, per_page: int = 10):
        """Handle some error conditions and tries to return working pagination."""
        results = None
        paginator = None
        try:
            paginator = Paginator(collection, per_page)
        except Exception:
            pass

        try:
            results = paginator.page(page)
        except PageNotAnInteger:
            # If page is not an integer, deliver first page.
            results = paginator.page(1)
        except EmptyPage:
            # If page is out of range (e.g. 9999), deliver last page of results.
            results = paginator.page(paginator.num_pages)

        return results, paginator

    def serve(self, request, page=None):
        """Serve the search page with query info and paginated results."""
        template = self.get_template(request)
        context = self.get_context(request)

        per_page = 10
        searchable_models = [
            AboutPage, AboutSubPage, CaseStudyPage, HistoryPage,
            PeoplePage, ContactPage, EventPage, GuidanceAndSupportPage,
            GuidanceGroupPage, GuidancePage, NewsPage,
            StandardPage, IATIStandardPage,
        ]
        # TODO: add KnowledgebasePage back if activated

        query = request.GET
        search_query = request.GET.get('query', '')
        page = request.GET.get('page', 1)

        if search_query:
            search_results = [r for m in searchable_models
                              for r in m.objects.live().public().search(search_query).annotate_score('_score')]
            search_results = sorted(search_results, key=lambda x: x._score, reverse=True)

            promoted = [x.page.specific for x in Query.get(search_query).editors_picks.all() if x.page.live]
            promoted_page_ids = [promoted_page.id for promoted_page in promoted]
            filtered_search_results = [result for result in search_results if result.id not in promoted_page_ids]

            query = Query.get(search_query)
            query.add_hit()

            results = list(chain(promoted, filtered_search_results))

        else:
            results = Page.objects.none()

        search_results, paginator = self.get_paginated(results, page, per_page)

        total_pages = search_results.paginator.num_pages if search_results else 0

        range_start = search_results.number - 5 if search_results.number > 5 else 1
        if search_results.number < (total_pages - 4):
            range_end = search_results.number + 4
        else:
            range_end = total_pages

        context['search_query'] = search_query
        context['search_results'] = search_results
        context['paginator_range'] = [i for i in range(range_start, range_end + 1)]
        context['paginator'] = paginator

        return render(request, template, context)

from django.contrib import admin
from django.db.models import QuerySet
from django.http import HttpRequest
from django.utils.html import format_html

from .filters import (
    CompanySizeFilter,
    SalaryFilter,
    TechnologyFilter,
)
from .models import (
    Company,
    Job,
    JobCategory,
    JobLocation,
    JobSalary,
    Technology,
)
from .tasks import download_jobs


@admin.register(Job)
class JobAdmin(admin.ModelAdmin):
    list_display = (
        'title',
        'company',
        'company_size',
        'board',
        'category',
        'salary',
        'url_display',
    )
    actions = ('fix_nofluff_links',)
    list_filter = (
        CompanySizeFilter,
        'seniority',
        TechnologyFilter,
        SalaryFilter,
        'company__status',
    )

    def get_queryset(self, request):
        qs = super().get_queryset(request)
        return qs.prefetch_related('salary', 'category', 'board', 'company')

    @admin.display(description='Salary')
    def salary(self, obj):
        return f'{obj.salary.amount_from} - {obj.salary.amount_to}'

    @admin.display(description='Company size')
    def company_size(self, obj):
        return f'{obj.company.size_from} - {obj.company.size_to}'

    @admin.display(description='link')
    def url_display(self, obj: Job):
        return format_html(f'<a href="{obj.url}">Link</a>')

    @admin.action(description='Blacklist selected companies')
    def blacklist_company(self, request: HttpRequest, queryset: QuerySet[Job]) -> None:
        for job in queryset:
            job.company.is_blacklisted = True
            job.company.save()


@admin.register(JobSalary)
class JobSalaryAdmin(admin.ModelAdmin):
    """Register JobSalary model in admin panel."""


@admin.register(Technology)
class TechnologyAdmin(admin.ModelAdmin):
    """Register Technology model in admin panel."""


@admin.register(JobLocation)
class JobLocationAdmin(admin.ModelAdmin):
    """Register JobLocation model in admin panel."""


@admin.register(JobCategory)
class JobCategoryAdmin(admin.ModelAdmin):
    """Register JobCategory model in admin panel."""


@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ('name', 'industry', 'size_from', 'size_to', 'url')
    actions = ('deduplicate',)
    search_fields = ('name',)

    @admin.action(description='Deduplicate companies')
    def deduplicate(self, request: HttpRequest, queryset: QuerySet[Company]) -> None:
        for company in queryset:
            duplicated_companies = (
                Company.objects.get_possible_match(company.name).exclude(pk=company.pk)
            )
            for duplicate in duplicated_companies:
                duplicate.job_set.update(company=company)
                company.update_if_better(
                    industry=duplicate.industry,
                    size_from=duplicate.size_from,
                    size_to=duplicate.size_to,
                    url=duplicate.url,
                )
                duplicate.delete()

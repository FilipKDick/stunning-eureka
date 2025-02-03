from django.db import models

from .choices import HypeStatus, JobBoard
from .managers import (
    CompanyManager,
    JobManager,
)


class Company(models.Model):
    name = models.CharField(max_length=255)
    industry = models.CharField(max_length=255, blank=True)
    size_from = models.IntegerField()
    size_to = models.IntegerField()
    url = models.CharField(max_length=1024, blank=True)
    status = models.IntegerField(choices=HypeStatus.choices, default=HypeStatus.UNKNOWN)

    objects = CompanyManager()

    def __str__(self) -> str:
        return self.name

    def update_if_better(
        self,
        industry: str = '',
        size_from: int = 0,
        size_to: int = 0,
        url: str = '',
    ) -> 'Company':
        self.industry = self.industry or industry
        self.size_from = self.size_from or size_from
        self.size_to = self.size_to or size_to
        self.url = self.url or url
        self.save()
        return self


class JobLocation(models.Model):
    job = models.ForeignKey('grabbo.Job', on_delete=models.CASCADE)
    is_remote = models.BooleanField()
    is_covid_remote = models.BooleanField()
    city = models.CharField(max_length=32, null=True, blank=True)
    street = models.CharField(max_length=128, null=True, blank=True)

    def __str__(self) -> str:
        return f'{self.city}, {self.street}'


class JobCategory(models.Model):
    name = models.CharField(max_length=32)

    def __str__(self) -> str:
        return self.name


class Technology(models.Model):
    name = models.CharField(max_length=32)

    def __str__(self) -> str:
        return self.name


class JobSalary(models.Model):
    amount_from = models.IntegerField()
    amount_to = models.IntegerField()
    currency = models.CharField(max_length=4)
    job_type = models.CharField(max_length=32)

    def __str__(self) -> str:
        return f'{self.amount_from} - {self.amount_to} {self.currency}'


class Job(models.Model):
    board = models.IntegerField(choices=JobBoard.choices)
    category = models.ForeignKey(
        'grabbo.JobCategory',
        on_delete=models.SET_NULL,
        null=True,
    )
    technology = models.ForeignKey(
        'grabbo.Technology',
        on_delete=models.SET_NULL,
        null=True,
    )
    # TODO: maybe the relation should be the other way round?
    salary = models.ForeignKey('grabbo.JobSalary', on_delete=models.PROTECT, null=True)
    # because pracuj suxx
    salary_text = models.CharField(max_length=256, blank=True)
    seniority = models.CharField(max_length=256)
    original_id = models.CharField(max_length=256)
    company = models.ForeignKey('grabbo.Company', on_delete=models.SET_NULL, null=True)
    title = models.CharField(max_length=256)
    url = models.CharField(max_length=256)
    description = models.TextField(blank=True)
    status = models.IntegerField(choices=HypeStatus.choices, default=HypeStatus.UNKNOWN)
    responsibilities = models.TextField(blank=True)
    requirements = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    objects = JobManager()

    def __str__(self) -> str:
        return f'{self.title} in {self.company}'

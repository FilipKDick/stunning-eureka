import logging

from abc import ABC
from functools import cached_property

import requests

from bs4 import BeautifulSoup
from tqdm import tqdm

from .models import (
    Company,
    Job,
    JobBoard,
    JobCategory,
    JobSalary,
    Technology,
)

logger = logging.getLogger(__name__)


class BaseDownloader(ABC):
    companies_url: str = ''
    jobs_url: str = ''

    def download_companies(self):
        raise NotImplementedError('You must implement this method')

    def download_jobs(self):
        raise NotImplementedError('You must implement this method')

    def download(self):
        self.download_companies()
        self.download_jobs()


class NoFluffDownloader(BaseDownloader):
    companies_url = (
        'https://nofluffjobs.com/api/companies/search/all?'
        + 'salaryCurrency=PLN&salaryPeriod=month&region=pl'
    )
    jobs_url = (
        'https://nofluffjobs.com/api/search/posting?'
        + 'limit=4000&offset=0&salaryCurrency=PLN&salaryPeriod=month&region=pl'
    )

    @cached_property
    def job_board(self):
        return JobBoard.objects.get(name='nofluff')

    def download_companies(self):
        response = requests.get(self.companies_url)
        try:
            response.raise_for_status()
        except requests.HTTPError:
            logger.error('Whoops, couldnt get companies from nofluff.')

        resp_data = response.json()
        for company in tqdm(resp_data['items']):
            if Company.objects.filter(name=company['name']).exists():
                continue
            additional_company_data = self._scrap_company_page(company)

            Company.objects.create(
                name=company['name'],
                **additional_company_data,
            )

    def download_jobs(self):
        response = requests.post(
            self.jobs_url,
            json={'criteriaSearch': {'requirement': ['python']}, 'page': 1},
        )
        try:
            response.raise_for_status()
        except requests.HTTPError:
            logger.error('Whoops, couldnt get offers from NoFluff.')
        jobs = response.json()
        for job in tqdm(jobs['postings']):
            if Job.objects.filter(original_id=job['id']).exists():
                continue
            self._add_job(job)

    @staticmethod
    def _get_info_from_spans(spans: list, key_to_find: str) -> str:
        company_size_spans = [span for span in spans if key_to_find in span.string]
        try:
            # only first 32 chars because this is our max
            return company_size_spans[0].next_sibling.string[:32]  # noqa: WPS432
        except IndexError:
            return ''

    def _scrap_company_page(self, company):
        url = f'https://nofluffjobs.com/pl{company["url"]}'
        company_resp = requests.get(url)
        company_data = BeautifulSoup(company_resp.content, 'html.parser')
        spans = company_data.find(id='company-main').find_all('span')
        return {
            'url': url,
            'size': self._get_info_from_spans(spans, 'Wielkość firmy'),
            'industry': self._get_info_from_spans(spans, 'Branża'),
        }

    def _add_job(self, job):
        category, _ = JobCategory.objects.get_or_create(name=job['category'])
        salary = self._add_salary(job['salary'])
        company, _ = Company.objects.get_or_create(name=job['name'])
        technology, _ = Technology.objects.get_or_create(
            name=job.get('technology', 'Unknown'),
        )
        job_instance = Job.objects.create(
            original_id=job['id'],
            board=self.job_board,
            category=category,
            technology=technology,
            salary=salary,
            company=company,
            title=job['title'],
            url=job['url'],
        )
        job_instance.add_locations(job['location'])

    @staticmethod
    def _add_salary(salary_data):
        return JobSalary.objects.create(
            amount_from=salary_data['from'],
            amount_to=salary_data['to'],
            job_type=salary_data['type'],
            amount_currency=salary_data['currency'],
        )

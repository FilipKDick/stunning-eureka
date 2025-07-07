import factory

from factory import fuzzy  # noqa: WPS458 (need to import fuzzy separately)


class CompanyFactory(factory.django.DjangoModelFactory):
    name = factory.Faker('company')
    industry = fuzzy.FuzzyChoice(
        ['IT', 'Finance', 'Healthcare', 'Retail', 'Education', 'Other'],
    )
    size_from = fuzzy.FuzzyInteger(1, 100)
    size_to = fuzzy.FuzzyInteger(1, 100)
    url = factory.Faker('url')

    class Meta:
        model = 'grabbo.Company'


class JobLocationFactory(factory.django.DjangoModelFactory):
    job = factory.SubFactory('apps.grabbo.tests.factories.JobFactory')
    is_remote = fuzzy.FuzzyChoice([True, False])
    city = factory.Faker('city')
    street = factory.Faker('street_address')

    class Meta:
        model = 'grabbo.JobLocation'


class JobCategoryFactory(factory.django.DjangoModelFactory):
    name = fuzzy.FuzzyChoice(
        ['backend', 'frontend', 'devops', 'qa', 'design', 'other'],
    )

    class Meta:
        model = 'grabbo.JobCategory'


class TechnologyFactory(factory.django.DjangoModelFactory):
    name = fuzzy.FuzzyChoice(
        ['python', 'django', 'javascript', 'react', 'vue', 'ruby', 'rails', 'other'],
    )

    class Meta:
        model = 'grabbo.Technology'


class JobSalaryFactory(factory.django.DjangoModelFactory):
    amount_from = fuzzy.FuzzyInteger(1, 100)
    amount_to = fuzzy.FuzzyInteger(1, 100)
    currency = fuzzy.FuzzyChoice(['EUR', 'USD', 'GBP', 'PLN', 'CHF', 'CAD', 'AUD'])
    job_type = fuzzy.FuzzyChoice(['permanent', 'contract', 'internship'])

    class Meta:
        model = 'grabbo.JobSalary'


class JobFactory(factory.django.DjangoModelFactory):
    board = factory.FuzzyChoice([1,2])  # TODO: use JobBoard.choices
    category = factory.SubFactory('apps.grabbo.tests.factories.JobCategoryFactory')
    technology = factory.SubFactory('apps.grabbo.tests.factories.TechnologyFactory')
    salary = factory.SubFactory('apps.grabbo.tests.factories.JobSalaryFactory')
    original_id = factory.Faker('uuid4')
    company = factory.SubFactory('apps.grabbo.tests.factories.CompanyFactory')
    title = factory.Faker('sentence')
    url = factory.Faker('url')

    class Meta:
        model = 'grabbo.Job'

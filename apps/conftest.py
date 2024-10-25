from apps.grabbo.tests import factories as grabbo_factories

import pytest

from pytest_factoryboy import register

register(grabbo_factories.CompanyFactory)
register(grabbo_factories.JobLocationFactory)
register(grabbo_factories.JobCategoryFactory)
register(grabbo_factories.TechnologyFactory)
register(grabbo_factories.JobSalaryFactory)
register(grabbo_factories.JobFactory)


@pytest.fixture(autouse=True)
def media_storage(settings, tmpdir):
    settings.MEDIA_ROOT = tmpdir.strpath

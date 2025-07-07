from django.db import models


class CompanyManager(models.Manager):
    def get_possible_match(self, name: str) -> models.QuerySet:
        """There are different company names on different boards."""
        # TODO: matching by company url
        # TODO: non-polish: gmbh, llc
        stripped_name = name.replace('sp. z o.o.', '').strip()
        # if name didn't have sp. z o.o., let's add it to the check
        name_with_sp = f'{stripped_name} sp. z o.o.'
        return (
            self.filter(name__iexact=stripped_name)
            | self.filter(name__iexact=name_with_sp)
        )

    def create_or_update_if_better(
        self,
        name: str,
        url: str = '',
        **kwargs,
    ):
        url = url.strip()
        possible_matches = self.get_possible_match(name)
        if possible_matches.count() == 1:
            # if we have only one possible match, let's update it
            return possible_matches.first().update_if_better(url=url, **kwargs)
        # if 0 or more than 1 possible matches, we create a new company
        # if there's more than 1, we can't be sure that this is the same one,
        # so we create a new company
        return self.create(
            name=name.replace('sp. z o.o.', '').strip(),
            url=url,
            **kwargs,
        )


class JobManager(models.Manager):
    def create_if_not_duplicate(self, *args, **kwargs) -> models.QuerySet:
        duplicate = self.filter(
            title__iexact=kwargs['title'],
            company=kwargs['company'],
            salary__amount_from=kwargs['salary'].amount_from,
            salary__amount_to=kwargs['salary'].amount_to,
            technology=kwargs['technology'],
        )
        if duplicate.exists():
            kwargs['salary'].delete()
            return duplicate.first()
        return self.create(**kwargs)

# Generated by Django 4.2.2 on 2025-01-11 18:49

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('grabbo', '0012_add_pracuj'),
    ]

    operations = [
        migrations.AlterField(
            model_name='job',
            name='salary',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.PROTECT, to='grabbo.jobsalary'),
        ),
    ]

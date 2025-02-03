# Generated by Django 4.2.2 on 2025-01-11 18:31

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('grabbo', '0011_alter_job_board_delete_jobboard'),
    ]

    operations = [
        migrations.AddField(
            model_name='job',
            name='requirements',
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name='job',
            name='responsibilities',
            field=models.TextField(blank=True),
        ),
        migrations.AddField(
            model_name='job',
            name='salary_text',
            field=models.CharField(blank=True, max_length=256),
        ),
        migrations.AlterField(
            model_name='job',
            name='board',
            field=models.IntegerField(choices=[(1, 'No Fluff'), (2, 'Just Join It'), (3, 'Pracuj')]),
        ),
    ]

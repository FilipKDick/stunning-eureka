"""Base settings to build other settings files upon."""
from pathlib import Path

import environ

ROOT_DIR = Path(__file__).resolve(strict=True).parent.parent.parent
# jobbo/
APPS_DIR = ROOT_DIR / 'apps'
env = environ.Env()

READ_DOT_ENV_FILE = env.bool('DJANGO_READ_DOT_ENV_FILE', default=False)
if READ_DOT_ENV_FILE:
    # OS environment variables take precedence over variables from .env
    env.read_env(str(ROOT_DIR / '.env'))

# GENERAL
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#debug
DEBUG = env.bool('DJANGO_DEBUG', False)  # noqa: WPS425, no keyword is ok here
# Local time zone. Choices are
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
# though not all of them may be available with every OS.
# In Windows, this must be set to your system time zone.
TIME_ZONE = 'UTC'
# https://docs.djangoproject.com/en/dev/ref/settings/#language-code
LANGUAGE_CODE = 'en-us'
# https://docs.djangoproject.com/en/dev/ref/settings/#site-id
SITE_ID = 1
# https://docs.djangoproject.com/en/dev/ref/settings/#use-i18n
USE_I18N = True
# https://docs.djangoproject.com/en/dev/ref/settings/#use-l10n
USE_L10N = True
# https://docs.djangoproject.com/en/dev/ref/settings/#use-tz
USE_TZ = True
# https://docs.djangoproject.com/en/dev/ref/settings/#locale-paths
LOCALE_PATHS = [str(ROOT_DIR / 'locale')]

# DATABASES
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#databases
DATABASES = {'default': env.db('DATABASE_URL')}
DATABASES['default']['ATOMIC_REQUESTS'] = True
# https://docs.djangoproject.com/en/stable/ref/settings/#std:setting-DEFAULT_AUTO_FIELD
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# URLS
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#root-urlconf
ROOT_URLCONF = 'config.urls'
# https://docs.djangoproject.com/en/dev/ref/settings/#wsgi-application
WSGI_APPLICATION = 'config.wsgi.application'

# APPS
# ------------------------------------------------------------------------------
DJANGO_APPS = [
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.sites',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    # 'django.contrib.humanize', # Handy template tags
    'django.contrib.admin',
    'django.forms',
]
THIRD_PARTY_APPS = [
    'crispy_forms',
    'crispy_bootstrap5',
    'django_celery_beat',
    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
    'drf_spectacular',
]

LOCAL_APPS = [
    'apps.grabbo',
    'apps.users',
    # Your stuff: custom apps go here
]
# https://docs.djangoproject.com/en/dev/ref/settings/#installed-apps
INSTALLED_APPS = DJANGO_APPS + THIRD_PARTY_APPS + LOCAL_APPS

# AUTHENTICATION
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#authentication-backends
AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',
]
# https://docs.djangoproject.com/en/dev/ref/settings/#auth-user-model
AUTH_USER_MODEL = 'users.User'

# PASSWORDS
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#password-hashers
PASSWORD_HASHERS = [
    'django.contrib.auth.hashers.Argon2PasswordHasher',
    'django.contrib.auth.hashers.PBKDF2PasswordHasher',
    'django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher',
    'django.contrib.auth.hashers.BCryptSHA256PasswordHasher',
]
# https://docs.djangoproject.com/en/dev/ref/settings/#auth-password-validators
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},  # noqa: E501
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]

# MIDDLEWARE
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#middleware
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.locale.LocaleMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.common.BrokenLinkEmailsMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# STATIC
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#static-root
STATIC_ROOT = str(ROOT_DIR / 'staticfiles')
STATIC_URL = '/static/'
STATICFILES_DIRS = [str(APPS_DIR / 'static')]
STATICFILES_FINDERS = [
    'django.contrib.staticfiles.finders.FileSystemFinder',
    'django.contrib.staticfiles.finders.AppDirectoriesFinder',
]

# MEDIA
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#media-root
MEDIA_ROOT = str(APPS_DIR / 'media')
# https://docs.djangoproject.com/en/dev/ref/settings/#media-url
MEDIA_URL = '/media/'

# TEMPLATES
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#templates
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [str(APPS_DIR / 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.template.context_processors.i18n',
                'django.template.context_processors.media',
                'django.template.context_processors.static',
                'django.template.context_processors.tz',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# https://docs.djangoproject.com/en/dev/ref/settings/#form-renderer
FORM_RENDERER = 'django.forms.renderers.TemplatesSetting'

# http://django-crispy-forms.readthedocs.io/en/latest/install.html#template-packs
CRISPY_TEMPLATE_PACK = 'bootstrap5'
CRISPY_ALLOWED_TEMPLATE_PACKS = 'bootstrap5'

# FIXTURES
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#fixture-dirs
FIXTURE_DIRS = (str(APPS_DIR / 'fixtures'),)

# SECURITY
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#session-cookie-httponly
SESSION_COOKIE_HTTPONLY = True
# https://docs.djangoproject.com/en/dev/ref/settings/#csrf-cookie-httponly
CSRF_COOKIE_HTTPONLY = True
# https://docs.djangoproject.com/en/dev/ref/settings/#secure-browser-xss-filter
SECURE_BROWSER_XSS_FILTER = True
# https://docs.djangoproject.com/en/dev/ref/settings/#x-frame-options
X_FRAME_OPTIONS = 'DENY'

# EMAIL
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#email-backend
EMAIL_BACKEND = env(
    'DJANGO_EMAIL_BACKEND',
    default='django.core.mail.backends.smtp.EmailBackend',
)
# https://docs.djangoproject.com/en/dev/ref/settings/#email-timeout
EMAIL_TIMEOUT = 5
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = env('EMAIL_HOST_USER', default='')
EMAIL_HOST_PASSWORD = env('EMAIL_HOST_PASSWORD', default='')
DEFAULT_FROM_EMAIL = env('EMAIL_HOST_USER', default='')
SERVER_EMAIL = env('EMAIL_HOST_USER', default='')

# ADMIN
# ------------------------------------------------------------------------------
# Django Admin URL.
ADMIN_URL = 'admin/'
# https://docs.djangoproject.com/en/dev/ref/settings/#admins
ADMINS = [('KJagiela', 'kjagiela@example.com')]
# https://docs.djangoproject.com/en/dev/ref/settings/#managers
MANAGERS = ADMINS

# LOGGING
# ------------------------------------------------------------------------------
# https://docs.djangoproject.com/en/dev/ref/settings/#logging
# See https://docs.djangoproject.com/en/dev/topics/logging for
# more details on how to customize your logging configuration.
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': (
                '%(levelname)s %(asctime)s %(module)s '
                + '%(process)d %(thread)d %(message)s'
            ),
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
        },
    },
    'root': {'level': 'INFO', 'handlers': ['console']},
}

# Celery
# ------------------------------------------------------------------------------
if USE_TZ:
    CELERY_TIMEZONE = TIME_ZONE
CELERY_BROKER_URL = env('CELERY_BROKER_URL')
CELERY_RESULT_BACKEND = CELERY_BROKER_URL
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
# TODO: set to whatever value is adequate in your circumstances
CELERY_TASK_TIME_LIMIT = 5 * 60
# TODO: set to whatever value is adequate in your circumstances
CELERY_TASK_SOFT_TIME_LIMIT = 5 * 60
CELERY_BEAT_SCHEDULER = 'django_celery_beat.schedulers:DatabaseScheduler'

# django-rest-framework
# -------------------------------------------------------------------------------
# django-rest-framework - https://www.django-rest-framework.org/api-guide/settings/
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.TokenAuthentication',
    ),
    # it will never be deployed - I wouldn't use a blank AllowAny otherwise
    'DEFAULT_PERMISSION_CLASSES': ('rest_framework.permissions.AllowAny',),
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 100,
}

# django-cors-headers - https://github.com/adamchainz/django-cors-headers#setup
# normally a decent setup would be required, but I don't plan on deploying this
CORS_ALLOW_ALL_ORIGINS = True

SPECTACULAR_SETTINGS = {
    'TITLE': 'Jobbo API',
    'DESCRIPTION': 'Documentation of API endpoints of Jobbo',
    'VERSION': '1.0.0',
    'SERVE_PERMISSIONS': ['rest_framework.permissions.IsAdminUser'],
    'SERVERS': [
        {'url': 'https://127.0.0.1:8000', 'description': 'Local Development server'},
        {'url': 'https://example.com', 'description': 'Production server'},
    ],
}
# Your stuff...
# ------------------------------------------------------------------------------

from django.core.mail import EmailMessage


def send_mail_with_offers(offers: list):
    content = '<br><h2>Nowe oferty:</h2><br>'
    for offer in offers:
        company = offer.get('company__name', offer.get('company'))
        content += f'<a href="{offer["url"]}">{offer["title"]} w {company}</a><br>'
    msg = EmailMessage('Found a lot of new offers', body=content, to=['jan.zysko@gmail.com', 'olenansn@gmail.com'])
    msg.content_subtype = 'html'
    msg.send()

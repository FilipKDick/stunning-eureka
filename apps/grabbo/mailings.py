from django.core.mail import EmailMessage


def send_mail_with_offers(offers: list):
    content = '<h1>SIEMA LENA</h1><br><br><h2>Nowe oferty:</h2><br>'
    for offer in offers:
        company = offer.get('company__name', offer.get('company'))
        content += f'<a href="{offer["url"]}">{offer["title"]} w {company}</a><br>'
    msg = EmailMessage('Found a lot of new offers', body=content, to=['filipkjagiela@gmail.com'])
    msg.content_subtype = 'html'
    msg.send()

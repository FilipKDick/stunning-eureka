from django.core.mail import EmailMessage


def send_mail_with_offers(offers: list):
    content = '<h1>SIEMA LENA</h1><br><br><h2>Nowe oferty:</h2><br>'
    for offer in offers:
        content += f'<a href="{offer["url"]}">{offer["title"]} w {offer["company"]}</a><br>'
    msg = EmailMessage('YEYEYEY', body=content, to=['filipkjagiela@gmail.com'])
    msg.content_subtype = 'html'
    msg.send()

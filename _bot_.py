# openssl req -newkey rsa:2048 -sha256 -nodes -x509 -days 365 -keyout YOURPRIVATE.key -out YOURPUBLIC.pem -subj "/C=UA/ST=Donetsk/L=Donetsk/O=Vic/CN=178.158.131.41"

# -*- coding: utf-8 -*-
#import requests

#acc_key = "1227859397:AAH1TQc069YNhNUQ4RiOeJtS-CHI6qbEfDM" 
#files = {'certificate': open('YOURPUBLIC.pem', 'rb')}

###------------------------->
#my_l = "https://api.telegram.org/bot"+acc_key+"/setWebhook?url=https://178.158.131.41:8443"
#r = requests.post(my_l, files=files)
#print (r.text)
####---------------->
#my_l = "https://api.telegram.org/bot"+acc_key+"/getWebhookInfo"
#r = requests.get(my_l)
#print (r.text)
##---------------->
##DELETE
##my_l = "https://api.telegram.org/bot"+acc_key+"/deleteWebhook"
##r = requests.get(my_l)
##print (r.text)
##  


import requests

class TelegramWebhookManager:
    def __init__(self, acc_key, public_certificate_path):
        self.acc_key = acc_key
        self.public_certificate_path = public_certificate_path

    def set_webhook(self, url):
        files = {'certificate': open(self.public_certificate_path, 'rb')}
        my_l = f"https://api.telegram.org/bot{self.acc_key}/setWebhook?url={url}"
        r = requests.post(my_l, files=files)
        return r.text

    def get_webhook_info(self):
        my_l = f"https://api.telegram.org/bot{self.acc_key}/getWebhookInfo"
        r = requests.get(my_l)
        return r.text

    def delete_webhook(self):
        my_l = f"https://api.telegram.org/bot{self.acc_key}/deleteWebhook"
        r = requests.get(my_l)
        return r.text

# Example usage:
# Replace 'YOURPUBLIC.pem' with the path to your actual public certificate file
telegram_manager = TelegramWebhookManager("7877809876:AAHG3mujWFUcFDYRhz3z_jhlI316TB2mvP4", "YOURPUBLIC.pem")
webhook_url = "https://178.158.131.41:8443"

print (telegram_manager.set_webhook(webhook_url))
print (telegram_manager.get_webhook_info())
#print (telegram_manager.delete_webhook()) # Uncomment to delete the webhook
#server	 *	 8443	 192.168.1.50	 	 TCP

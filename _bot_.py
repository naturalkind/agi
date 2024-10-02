# -*- coding: utf-8 -*-
import requests

# openssl req -newkey rsa:2048 -sha256 -nodes -keyout YOURPRIVATE.key -x509 -days 365 -out YOURPUBLIC.pem -subj "/C=UA/ST=Donetsk/L=Donetsk/O=Vic/CN=178.158.131.41"


#1 openssl req -newkey rsa:2048 -sha256 -nodes -x509 -days 365 -keyout cert.key -out cert.crt -subj "/C=UA/ST=Donetsk/L=Donetsk/O=Vic/CN=178.158.131.41"
#2 openssl x509 -in cert.crt -out cert.pem -outform PEM
#3 sudo cp cert.crt /etc/ssl/certs/cert.crt
#4 sudo cp cert.key /etc/ssl/private/cert.key
#5 python3 _bot.py

#openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout ssl/cert.key -out ssl/cert.crt
acc_key = "1227859397:AAH1TQc069YNhNUQ4RiOeJtS-CHI6qbEfDM" 
files = {'certificate': open('YOURPUBLIC.pem', 'rb')}

##------------------------->
my_l = "https://api.telegram.org/bot"+acc_key+"/setWebhook?url=https://178.158.131.41:8443"
r = requests.post(my_l, files=files)
print (r.text)
###---------------->
my_l = "https://api.telegram.org/bot"+acc_key+"/getWebhookInfo"
r = requests.get(my_l)
print (r.text)
#---------------->
#DELETE
#my_l = "https://api.telegram.org/bot"+acc_key+"/deleteWebhook"
#r = requests.get(my_l)
#print (r.text)
#  




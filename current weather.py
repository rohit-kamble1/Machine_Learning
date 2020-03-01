# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:05:19 2020

@author: a
"""

import pyowm
owm = pyowm.OWM('<api key>') #  Replace <api_key> with your API key
city=input("enter the city=")
sf = owm.weather_at_place('city')
weather = sf.get_weather()
print(weather.get_temperature('celsius')['temp'])
print(weather.get_sunrise_time(timeformat='iso')) # Prints time in GMT timezone
print(weather.get_sunset_time(timeformat='iso')) # Prints time in GMT timezone
print(weather.get_humidity())
#!/usr/bin/env python

from __future__ import print_function

# set unbuffered output
import os
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

########################################
print('processing cmd line args')
import argparse

parser=argparse.ArgumentParser('generate stats of a data set')
parser.add_argument('--out_dir',type=str,default='stats')
parser.add_argument('--data',type=str,required=True)

args = parser.parse_args()

########################################
print('looping through files')

files_all=[]
for path_date in os.listdir(args.data):
    path_date_full=os.path.join(args.data,path_date)
    if os.path.isdir(path_date_full):
        for path_hour in os.listdir(path_date_full):
            files_all.append(os.path.join(path_date_full,path_hour))
files_all=sorted(files_all)

import datetime
import hash
import gzip
import simplejson as json
from collections import defaultdict

dict_country=defaultdict(lambda:0)
dict_lang=defaultdict(lambda:0)
dict_country_lang=defaultdict(lambda:defaultdict(lambda:0))
dict_lang_country=defaultdict(lambda:defaultdict(lambda:0))

dict_hr_country=defaultdict(lambda:defaultdict(lambda:0))
dict_hr_lang=defaultdict(lambda:defaultdict(lambda:0))
dict_country_hr=defaultdict(lambda:defaultdict(lambda:0))
dict_lang_hr=defaultdict(lambda:defaultdict(lambda:0))
dict_country_hr_lang=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0)))
dict_lang_hr_country=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0)))

dict_day_country=defaultdict(lambda:defaultdict(lambda:0))
dict_day_lang=defaultdict(lambda:defaultdict(lambda:0))
dict_country_day=defaultdict(lambda:defaultdict(lambda:0))
dict_lang_day=defaultdict(lambda:defaultdict(lambda:0))
dict_lang_day_country=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0)))
dict_country_day_lang=defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:0)))

try:
    while len(files_all)>0:
        filename=files_all.pop()
        print(datetime.datetime.now(),filename)
        file=gzip.open(filename,'rt')

        while True:
            nextline=file.readline()
            if nextline=='':
                break
            data=json.loads(nextline)

            def ms2hr(ms):
                return ms/1000/60/60
            hr=ms2hr(int(data['timestamp_ms']))%24
            day=ms2hr(int(data['timestamp_ms']))%(24*7)
            lang=data['lang']
            try:
                country=data['place']['country_code']
            except:
                country='geo'

            dict_country[country]+=1
            dict_lang[lang]+=1
            dict_country_lang[country][lang]+=1
            dict_lang_country[lang][country]+=1

            dict_hr_lang[hr][lang]+=1
            dict_hr_country[hr][country]+=1
            dict_lang_hr[lang][hr]+=1
            dict_country_hr[country][hr]+=1
            dict_lang_hr_country[lang][hr][country]+=1
            dict_country_hr_lang[country][hr][lang]+=1

            dict_day_lang[day][lang]+=1
            dict_day_country[day][country]+=1
            dict_lang_day[lang][day]+=1
            dict_country_day[country][day]+=1
            dict_lang_day_country[lang][day][country]+=1
            dict_country_day_lang[country][day][lang]+=1

except:
    pass

dict_hr_country[24]=dict_hr_country[0]
dict_hr_lang[24]=dict_hr_lang[0]
for country in dict_country_hr:
    dict_country_hr[country][24]=dict_country_hr[country][0]
for lang in dict_lang_hr:
    dict_lang_hr[lang][24]=dict_lang_hr[lang][0]
for lang in dict_lang_hr_country:
    dict_lang_hr_country[lang][24]=dict_lang_hr_country[lang][0]
for country in dict_country_hr_lang:
    dict_country_hr_lang[country][24]=dict_country_hr_lang[country][0]

dict_day_country[24*7]=dict_day_country[0]
dict_day_lang[24*7]=dict_day_lang[0]
for country in dict_country_day:
    dict_country_day[country][24*7]=dict_country_day[country][0]
for lang in dict_lang_day:
    dict_lang_day[lang][24*7]=dict_lang_day[lang][0]
for lang in dict_lang_day_country:
    dict_lang_day_country[lang][24*7]=dict_lang_day_country[lang][0]
for country in dict_country_day_lang:
    dict_country_day_lang[country][24*7]=dict_country_day_lang[country][0]

########################################
print('outputing country/lang stats')

def mkdir(dir):
    import errno
    try:
        os.makedirs(dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

mkdir(args.out_dir)

####################
print('  country')
filename=args.out_dir+"/country.dat"
file=open(filename,"w")
total=sum(dict_country.values())
for country in dict_country:
    country_str=country
    if country=='':
        country_str="blank"
    file.write("%s %d %d %f\n"%(
        country_str,
        dict_country[country],
        total,
        float(dict_country[country])/total
        ))
file.close()

####################
print('  lang')
filename=args.out_dir+"/lang.dat"
file=open(filename,"w")
total=sum(dict_lang.values())
for lang in dict_lang:
    lang_str=lang
    if lang=='':
        lang_str="blank"
    file.write("%s %d %d %f\n"%(
        lang_str,
        dict_lang[lang],
        total,
        float(dict_lang[lang])/total
        ))
file.close()

####################
print('  dict_country_lang')
country_dir=args.out_dir+"/country"
mkdir(country_dir)
for country in dict_country_lang:
    if country=='':
        filename=country_dir+"/blank.dat"
    else:
        filename=country_dir+"/"+country+".dat"
    file=open(filename,"w")
    total=sum(dict_country_lang[country].values())
    for lang in dict_country_lang[country]:
        file.write("%s %d %d %f\n"%(
            lang,
            dict_country_lang[country][lang],
            total,
            float(dict_country_lang[country][lang])/total
            ))
    file.close()

####################
print('  dict_lang_country')
lang_dir=args.out_dir+"/lang"
mkdir(lang_dir)
for lang in dict_lang_country:
    if lang=='':
        filename=lang_dir+"/blank.dat"
    else:
        filename=lang_dir+"/"+lang+".dat"
    file=open(filename,"w")
    total=sum(dict_lang_country[lang].values())
    for country in dict_lang_country[lang]:
        country_str=country
        if country=='':
            country_str='blank'
        file.write("%s %d %d %f\n"%(
            country_str,
            dict_lang_country[lang][country],
            total,
            float(dict_lang_country[lang][country])/total
            ))
    file.close()

####################
print('  dict_hr_country')
filename=args.out_dir+"/hr_country.dat"
file=open(filename,"w")
file.write('hr total ')
for country in hash.country_codes_uniq:
    country_str=country
    if country=='':
        country_str='blank'
    file.write('%s '%country_str)
file.write('\n')
for hr in dict_hr_country:
    total=sum(dict_hr_country[hr].values())
    file.write('%d %d '%(hr,total))
    for country in hash.country_codes_uniq:
        file.write('%d '%dict_hr_country[hr][country])
    file.write('\n')
file.close()

####################
print('  dict_hr_lang')
filename=args.out_dir+"/hr_lang.dat"
file=open(filename,"w")
file.write('hr total ')
for lang in hash.langs_uniq:
    file.write('%s '%lang)
file.write('\n')
for hr in dict_hr_lang:
    total=sum(dict_hr_lang[hr].values())
    file.write('%d %d '%(hr,total))
    for lang in hash.langs_uniq:
        file.write('%d '%dict_hr_lang[hr][lang])
    file.write('\n')
file.close()

####################
print('  dict_lang_hr_country')
lang_hr_country_dir=args.out_dir+"/lang_hr_country"
mkdir(lang_hr_country_dir)

for lang in dict_lang_hr_country:
    filename=lang_hr_country_dir+"/"+lang+".dat"
    file=open(filename,"w")
    file.write('hr total ')
    for country in hash.country_codes_uniq:
        country_str=country
        if country=='':
            country_str='blank'
        file.write('%s '%country_str)
    file.write('\n')
    for hr in dict_lang_hr_country[lang]:
        total=sum(dict_lang_hr_country[lang][hr].values())
        file.write('%d %d '%(hr,total))
        for country in hash.country_codes_uniq:
            file.write('%d '%dict_lang_hr_country[lang][hr][country])
        file.write('\n')
    file.close()

####################
print('  dict_day_country')
filename=args.out_dir+"/day_country.dat"
file=open(filename,"w")
file.write('day total ')
for country in hash.country_codes_uniq:
    country_str=country
    if country=='':
        country_str='blank'
    file.write('%s '%country_str)
file.write('\n')
for day in dict_day_country:
    total=sum(dict_day_country[day].values())
    file.write('%d %d '%(day,total))
    for country in hash.country_codes_uniq:
        file.write('%d '%dict_day_country[day][country])
    file.write('\n')
file.close()

####################
print('  dict_day_lang')
filename=args.out_dir+"/day_lang.dat"
file=open(filename,"w")
file.write('day total ')
for lang in hash.langs_uniq:
    file.write('%s '%lang)
file.write('\n')
for day in dict_day_lang:
    total=sum(dict_day_lang[day].values())
    file.write('%d %d '%(day,total))
    for lang in hash.langs_uniq:
        file.write('%d '%dict_day_lang[day][lang])
    file.write('\n')
file.close()

####################
print('  dict_lang_day_country')
lang_day_country_dir=args.out_dir+"/lang_day_country"
mkdir(lang_day_country_dir)

for lang in dict_lang_day_country:
    filename=lang_day_country_dir+"/"+lang+".dat"
    file=open(filename,"w")
    file.write('day total ')
    for country in hash.country_codes_uniq:
        country_str=country
        if country=='':
            country_str='blank'
        file.write('%s '%country_str)
    file.write('\n')
    for day in dict_lang_day_country[lang]:
        total=sum(dict_lang_day_country[lang][day].values())
        file.write('%d %d '%(day,total))
        for country in hash.country_codes_uniq:
            file.write('%d '%dict_lang_day_country[lang][day][country])
        file.write('\n')
    file.close()

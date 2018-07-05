#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

# set unbuffered output
import os
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

########################################
print('processing cmd line args')
import argparse

parser=argparse.ArgumentParser('find patterns in the tweets')

parser.add_argument('--data',type=str,required=True)

args = parser.parse_args()

########################################
print('generating queries')

cities_japan=[
    u'\xe7\xa6\x8f\xe5\xb2\xa1', #Fukuoka
    u'\xe6\x9c\xad\xe5\xb9\x8c', #Sapporo
    u'\xe6\x9d\xb1\xe4\xba\xac', #Tokyo
    u'\xe6\xa8\xaa\xe6\xb5\x9c', #Yokohama
]

german=[
    'der radio',
    'das radio'
    ]

klingon=[
    "vjIjatlh",
    "jlyaj",
    "Hlja'",
    "HiSlaH",
    "ghobe'",
    #"lu'",
    #"luq",
    #"Qo'",
    ]

dothraki=[
    u'Me zisosh disse',
    u'Hash yer ray tih',
    u'Me achrae athnajaharoon',
    u'Che dothras che drivos',
    u'Ifas maisi yerit',
    u'shekh ma shieraki anni',
    '#conlang ',
    'esperanto',
    'klingon'
    ]

#latin=[
    #"ardinalibus"
    #"eorsum"

mediterranean=['diterr']
elconfidencial=['@elconfidencial']
#muertos=['d\\u00eda de muertos','d\\u00eda de los muertos','D\\u00eda de Muertos','D\\u00eda de los Muertos','d\\u00edade','D\\u00edaDe']
muertos=['de muertos','de Muertos','demuertos','DeMuertos']

queries=dothraki

import unicodedata
standardize = lambda str: unicodedata.normalize('NFKC',unicode(str.lower()))
queries = map(standardize,queries)
print('queries=',queries)

########################################
print('processing')

# get files list
files_all=[]
for path_date in os.listdir(args.data):
    path_date_full=os.path.join(args.data,path_date)
    if os.path.isdir(path_date_full):
        for path_hour in os.listdir(path_date_full):
            files_all.append(os.path.join(path_date_full,path_hour))
files_all.sort()
files_all=list(reversed(files_all))

# loop through files
import simplejson as json
import gzip
while len(files_all)>0:

    # get the next line,
    # or open a new file if no line exists
    try:
        nextline=file.readline()
        tweet=json.loads(nextline)
    except:
        files_all=files_all[1:]
        print('file=',files_all[0])
        file=gzip.open(files_all[0])
        continue

    # perform tests
    for q in queries:
        if q in standardize(tweet['text']):
            print(nextline)


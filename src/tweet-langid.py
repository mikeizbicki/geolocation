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
files_all=filter(lambda x: x[-3:]=='.gz',files_all)

# get better pretty printing of defaultdict
import collections
class defaultdict(collections.defaultdict):
        __repr__ = dict.__repr__
def lambda0():
    return 0
langdict_id=defaultdict(lambda0)
langdict_tweet=defaultdict(lambda0)

# loop through files
import datetime
import langid
import simplejson as json
import gzip
i=0
while len(files_all)>0:
    i+=1

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
    try:
        langdict_tweet[tweet['lang']]+=1
        langdict_id[langid.classify(tweet['text'])[0]]+=1
    except:
        pass

    if i%10==0:
        print(datetime.datetime.now(),'i:',i,' keys: ',len(langdict_id.keys()),'/',len(langdict_tweet.keys()))

    if i%100000==0:
        print('keys=',langdict.keys())

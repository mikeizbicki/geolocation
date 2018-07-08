#!/usr/bin/env python

from __future__ import print_function

# command line arguments
import argparse
parser=argparse.ArgumentParser('print the contents')
parser.add_argument('--data',type=str,required=True)
args = parser.parse_args()

# get files list
import os
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
import pprint
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

    # process the tweet
    print('full tweet=')
    pprint.pprint(tweet)

    print('text=',tweet['text'].encode('utf-8'))
    print('lang=',tweet['lang'])
    print('country=',tweet['place']['country_code'])

    # wait for user input
    raw_input('press enter to continue')


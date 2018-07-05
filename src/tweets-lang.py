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

parser=argparse.ArgumentParser('filter tweets by Twitter assigned language')

parser.add_argument('--data',type=str,required=True)
parser.add_argument('--lang',type=str,required=True)
parser.add_argument('--output',type=str,default='data/lang')

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
    if tweet['lang']==args.lang:




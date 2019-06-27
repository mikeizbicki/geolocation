#!/bin/python

from __future__ import print_function

# set unbuffered output
import os
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

import itertools
hex=['1','2'] #,'3','4','5','6','7','8']
perms=list(map(''.join,itertools.product(hex,repeat=3)))
#=['/rhome/mizbicki/bigdata/geolocating/data/flickr/img2/'+perm+'/*.jpg' for perm in perms]

# get better pretty printing of defaultdict
import collections
class defaultdict(collections.defaultdict):
    __repr__ = dict.__repr__

lambda0 = lambda: 0.0
country_count=defaultdict(lambda0)
tot=0.0

import data
print('perms=',perms)
for perm in perms:
    dirname='/rhome/mizbicki/bigdata/geolocating/data/flickr/img2/'+perm
    print('dirname=',dirname)
    for filename in os.listdir(dirname):
        gps,country=data.imgpath2labels(dirname+'/'+filename)
        if country!=252:
            print('  tot=',tot,'; filename=',filename,'; gps=',gps,'; country=',country)
            country_count[country]+=1.0
            tot+=1.0
        else:
            print('filename=',dirname+'/'+filename)
            #asd

#import pprint
#country_percent=map(lambda x: x/tot, country_count)
#pprint.pprint(country_count)
for k,v in country_count.items():
    print(data.country_codes[k],v,v/tot)

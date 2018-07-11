#!/usr/bin/python

from __future__ import print_function
#import simplejson as json
#import simplejson as json
import json
from pprint import pprint
import sys
import datetime
import gzip
import pickle
import gzip

# set unbuffered output
import os
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

# get better pretty printing of defaultdict
import collections
class defaultdict(collections.defaultdict):
        __repr__ = dict.__repr__

files=(sys.argv[1:])
#files=['/data/twitter/geoTwitter17-11-06/geoTwitter17-11-06_14:05']
maxtweets=sys.maxint
#maxtweets=1000

def lambda0():
    return 0
num_pt=defaultdict(lambda0)
num_ct=defaultdict(lambda0)
num_lang=defaultdict(lambda0)

def lambda1():
    return defaultdict(lambda0)
num_fn=defaultdict(lambda1)
loc_fn=defaultdict(lambda1)

numlines=0
numtweets=0
numgeo=0
numca=0
nummx=0
for filename in files:
    print(datetime.datetime.now(),filename)
    f=gzip.open(filename,'rt')

    text='start'
    while text != '' and numlines < maxtweets:
        text=f.readline()
        numlines+=1
        try:
            data=json.loads(text)
            #pprint(data)
            #sys.exit(0)
            if 'limit' in data or not data['place']:
                continue
            numtweets+=1

            place_type=data['place']['place_type']
            full_name=data['place']['full_name']
            #print(place_type,': ',full_name)

            num_pt[place_type]+=1
            num_fn[place_type][full_name]+=1
            loc_fn[place_type][full_name]=data['place']['bounding_box']

            if 'country_code' in data['place']:
                num_ct[data['place']['country_code']]+=1
            if data['geo']:
                numgeo+=1
            if 'lang' in data:
                num_lang[data['lang']]+=1

        except:
            #sys.exit(0)
            e = sys.exc_info()[0]
            #print('error on line', numlines,': ',e)
            #pprint(data)

    f.close()

    print('numlines=',numlines,'; numtweets=',numtweets,'(',numtweets/float(numlines),'%); numgeo=',numgeo,'(',numgeo/float(numtweets),'%)')
    #print('numca=',numca,'(',numca/float(numtweets),'%)')
    #print('nummx=',nummx,'(',nummx/float(numtweets),'%)')

#print('num_pt:')
#pprint((num_pt))
print('num_ct:')
pprint((num_ct))
#print('num_fn:')
#pprint((num_fn))
print('num_lang:')
pprint((num_lang))

with open('summary-country_codes.pkl','w') as f:
    pickle.dump(num_ct,f)

with open('summary-lang.pkl','w') as f:
    pickle.dump(num_lang,f)

#pickle.dump(num_pt,f)
#pickle.dump(num_fn,f)
#pickle.dump(loc_fn,f)
#pickle.dump(num_lang,f)
#f.close()


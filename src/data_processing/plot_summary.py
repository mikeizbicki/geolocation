#!/usr/bin/python
from __future__ import print_function

# set unbuffered output
import os
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

# cmd line args
import argparse
parser=argparse.ArgumentParser('summarize data')
parser.add_argument('--summary',type=str,default='data/BillionTwitter/summary-small.pkl')
args = parser.parse_args()

# get better pretty printing of defaultdict
import collections
class defaultdict(collections.defaultdict):
        __repr__ = dict.__repr__

# data summaries
def lambda0():
    return 0
def lambda1():
    return defaultdict(lambda0)

# process file
import pickle
import pprint
with open(args.summary,'r') as f:

    print('unpickling')
    num_ct=pickle.load(f)
    num_lang=pickle.load(f)
    #num_pt=pickle.load(f)
    #num_user_lang=pickle.load(f)
    #num_fn=pickle.load(f)
    #loc_fn=pickle.load(f)

    # total tweets
    total_tweets=sum(num_ct.values())
    print('total tweets=',total_tweets)

    # num_user_lang
    #print('analyzing num_user_lang')
    #pprint.pprint(num_user_lang)
    #for v in num_user_lang:
        #print(v.keys())
        #sys.exit(0)

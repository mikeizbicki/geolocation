#!/usr/bin/python
from __future__ import print_function

# set unbuffered output
import os
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

# cmd line args
import argparse
parser=argparse.ArgumentParser('summarize data')
parser.add_argument('--files',type=str,default=None,nargs='*')
parser.add_argument('--outfile',type=str,default='summary.pkl')
parser.add_argument('--verbose',action='store_true')
args = parser.parse_args()

# get better pretty printing of defaultdict
import collections
class defaultdict(collections.defaultdict):
        __repr__ = dict.__repr__

# data summaries
def lambda0():
    return 0
num_pt=defaultdict(lambda0)
num_ct=defaultdict(lambda0)
num_lang=defaultdict(lambda0)

def lambda1():
    return defaultdict(lambda0)
num_fn=defaultdict(lambda1)
loc_fn=defaultdict(lambda1)
num_user_lang=defaultdict(lambda1)

# main loop
import pickle
import datetime
for filename in args.files:
    print(datetime.datetime.now(),filename)
    with open(filename) as f:

        tmp_ct=pickle.load(f)
        for k in tmp_ct.keys():
            num_ct[k]+=tmp_ct[k]

        tmp_lang=pickle.load(f)
        for k in tmp_lang.keys():
            num_lang[k]+=tmp_lang[k]

        tmp_pt=pickle.load(f)
        for k in tmp_pt.keys():
            num_pt[k]+=tmp_pt[k]

        tmp_user_lang=pickle.load(f)
        for k in tmp_user_lang.keys():
            for k2 in tmp_user_lang[k].keys():
                num_user_lang[k][k2]+=tmp_user_lang[k][k2]

        tmp_fn=pickle.load(f)
        for k in tmp_fn.keys():
            for k2 in tmp_user_lang[k].keys():
                num_fn[k][k2]=tmp_fn[k][k2]

        tmp_fn=pickle.load(f)
        for k in tmp_fn.keys():
            for k2 in tmp_user_lang[k].keys():
                loc_fn[k][k2]=tmp_fn[k][k2]

if args.verbose:
    from pprint import pprint
    print('num_ct:')
    pprint((num_ct))
    print('num_lang:')
    pprint((num_lang))

with open(args.outfile,'w') as f:
    #pickle.dump(num_ct,f)
    #pickle.dump(num_lang,f)
    #pickle.dump(num_user,f)
    pickle.dump(num_ct,f)
    pickle.dump(num_lang,f)
    pickle.dump(num_pt,f)
    pickle.dump(num_user_lang,f)
    pickle.dump(num_fn,f)
    pickle.dump(loc_fn,f)


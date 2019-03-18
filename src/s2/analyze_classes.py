
########################################
print('processing cmd line args')
import argparse

parser=argparse.ArgumentParser('print statistics of class file')
parser.add_argument('--filename',type=str,required=True)
args=parser.parse_args()

########################################
print('processing class')

import s2sphere
import pickle
with open(args.filename,'rb') as f:
    s2cells=pickle.load(f)

counts={}
for i in range(0,31):
    counts[i]=0

for cell in s2cells:
    counts[cell.level()]+=1

import pprint
pprint.pprint(counts)

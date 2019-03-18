#!/bin/python

from __future__ import print_function

########################################
print('processing cmd line args')
import argparse

parser=argparse.ArgumentParser('convert gps coordinates to s2 class labels')
parser.add_argument('--output_prefix',type=str,default='level')
args=parser.parse_args()

########################################
print('importing')
import pickle
import datetime
import heapq
import queue
import s2sphere as s2

for level in range(0,10):
    print('level=',level)
    cells=s2.CellId.walk(level)
    with open(args.output_prefix+'-'+str(level),'wb') as output_file:
        cellids = list(cells)
        #print('cellids=',cellids)
        pickle.dump(cellids,output_file)


#!/bin/python

from __future__ import print_function

########################################
print('processing cmd line args')
import argparse

parser=argparse.ArgumentParser('convert gps coordinates to s2 class labels')
parser.add_argument('--tfrecords',type=str,default='tmp.tfrecord')
parser.add_argument('--output',type=str,default='gps.list')
parser.add_argument('--max_entries',type=int,default=None)
args=parser.parse_args()

########################################
print('importing')
import tensorflow as tf
import pickle
import datetime

########################################
print('loading data points')

record_iterator = tf.python_io.tf_record_iterator(path=args.tfrecords)
gps_coords=[]

num_entries=0
for string_record in record_iterator:
    # debug message
    if num_entries%10000==0:
        print('%s  num_entries=%d' %
            ( datetime.datetime.now()
            , num_entries
            ))

    # early stopping
    if args.max_entries:
        if num_entries>=args.max_entries:
            break

    # process next entry
    num_entries+=1
    example = tf.train.Example()
    example.ParseFromString(string_record)
    gps = list(example.features.feature['train/gps'].float_list.value)
    gps_coords.append(gps)

########################################
print('outputting file')
with open(args.output,'wb') as output_file:
    pickle.dump(gps_coords,output_file)


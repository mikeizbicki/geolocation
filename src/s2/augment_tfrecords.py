#!/bin/python

from __future__ import print_function

########################################
print('processing cmd line args')
import argparse

parser=argparse.ArgumentParser('convert gps coordinates to s2 class labels')
parser.add_argument('--tfrecord_input',type=str,default='tmp.tfrecord')
parser.add_argument('--tfrecord_output',type=str,required=True)
parser.add_argument('--parallel_max',type=int,default=1)
parser.add_argument('--parallel_num',type=int,default=0)
parser.add_argument('--maxrange',type=int,default=18)
args=parser.parse_args()

########################################
print('importing')
import tensorflow as tf
import pickle
import datetime
import s2sphere

########################################
print('loading s2cells')

s2cells={}
for i in range(4,args.maxrange):
    print('  i=',i)
    exp=2**i
    filename='s2/class_cells-'+str(exp)
    with open(filename,'rb') as f:
        s2cells_tmp=pickle.load(f)
    max_cells=len(s2cells_tmp)
    s2cells[exp]=zip(s2cells_tmp,range(0,max_cells))

########################################
print('processing data points')

record_iterator = tf.python_io.tf_record_iterator(path=args.tfrecord_input)
outfile=args.tfrecord_output+'_'+str(args.parallel_max)+'_'+str(args.parallel_num)
record_writer=tf.python_io.TFRecordWriter(outfile)

num_entries=0
for string_record in record_iterator:
    # debug message
    if num_entries%1000==0:
        print('%s  num_entries=%d' %
            ( datetime.datetime.now()
            , num_entries
            ))

    # process next entry
    num_entries+=1

    if num_entries%args.parallel_max==args.parallel_num:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        gps=list(example.features.feature['train/gps'].float_list.value)
        features = {
            'train/gps' : example.features.feature['train/gps'],
            'train/country' : example.features.feature['train/country'],
            'train/features' : example.features.feature['train/features']
        }

        def gps2cellid(gps,s2cells):
            gps_cell=s2sphere.CellId.from_lat_lng(s2sphere.LatLng.from_degrees(gps[0],gps[1]))
            #print('debug=',gps_cell.to_lat_lng())
            def heap_sort(unsorted):
                import heapq
                unsorted = unsorted[:]
                def heap_sort_destructive(items):
                    heapq.heapify(items)
                    while items:
                        yield heapq.heappop(items)
                return heap_sort_destructive(unsorted)
            s2cells_ordered=heap_sort([(abs(cell.id()-gps_cell.id()),cell,id) for (cell,id) in s2cells])
            for (dist,cell,id) in s2cells_ordered:
                if cell.contains(gps_cell):
                    return id
            return max_cells+1

        #print('num_entries=',num_entries)
        for i in s2cells.keys():
            cellid=gps2cellid(gps,s2cells[i])
            feature=tf.train.Feature(int64_list=tf.train.Int64List(value=[cellid]))
            features['train/s2/'+str(i)] = feature
            #s2cells[i][cellid]
            #gps_s2=[
                #s2cells[i][cellid][0].to_lat_lng().lat().degrees,
                #s2cells[i][cellid][0].to_lat_lng().lng().degrees,
                #]
            #print('  i:',i,'gps=',gps,'gps_s2=',gps_s2)

        example = tf.train.Example(features=tf.train.Features(feature=features))
        record_writer.write(example.SerializeToString())


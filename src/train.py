#!/usr/bin/env python

from __future__ import print_function

########################################
print('processing cmd line args')
import argparse
import sys

parser=argparse.ArgumentParser('train a model')

# administration variables
parser.add_argument('--maxtweets',type=int,default=sys.maxint,required=False)
parser.add_argument('--logdir',type=str,default='log')
parser.add_argument('--pickle',type=str)
parser.add_argument('filenames', metavar='N', type=str, nargs='+',
                    help='gzipped files containing tweets in json format')

# model hyperparameters
parser.add_argument('--hashsize',type=int,default=20)
parser.add_argument('--batchsize',type=int,default=100)
parser.add_argument('--learningrate',type=float,default=0.05)
parser.add_argument('--loss',choices=['l2','angular','xentropy'],required=True)
parser.add_argument('--output',choices=['gps','loc'],required=True)

args = parser.parse_args()

if args.output=='loc' and args.pickle==None:
    print('must specify a pickle file for "loc" outputs')
    sys.exit(1)

if args.pickle != None:
    import pickle
    from collections import defaultdict
    lambda0 = lambda: 0
    lambda1 = lambda: 1
    f=open(args.pickle,'r')
    num_pt=pickle.load(f)
    num_ct=pickle.load(f)
    num_fn=pickle.load(f)
    loc_fn=pickle.load(f)
    f.close()

    maxlocs=10
    locsfreqs=list(reversed(sorted(zip(num_fn['city'].values(),num_fn['city'].keys()))))[0:maxlocs]
    print(locsfreqs)
    locs=map(lambda (a,b):b,list(reversed(sorted(zip(num_fn['city'].values(),num_fn['city'].keys()))))[0:maxlocs])
    locs.append('')


    #locs=sum(map(lambda x: x.keys(), loc_fn.values()),[])
    lochash={ loc : i for loc,i in zip(locs,xrange(len(locs))) }
    from pprint import pprint
    #pprint(num_fn['city'].values())
    #sys.exit(0)

########################################
print('importing libraries')

import datetime
import gzip
import math
import os
import simplejson as json
import time

import geopy.distance
import numpy as np
import scipy as sp

import sklearn.feature_extraction.text
hv=sklearn.feature_extraction.text.HashingVectorizer(n_features=2**args.hashsize,norm=None)

########################################
print('initializing tensorflow')
import tensorflow as tf

def mkSparseTensorValue(m):
    m2=sp.sparse.coo_matrix(m)
    return tf.SparseTensorValue(zip(m2.row,m2.col),m2.data,m2.shape)

# tf inputs
x_ = tf.sparse_placeholder(tf.float32)
loc_ = tf.placeholder(tf.int32, [args.batchsize,1])
gps_ = tf.placeholder(tf.float32, [args.batchsize,2])
op_lat_ = gps_[:,0]
op_lon_ = gps_[:,1]
op_lat_rad_ = op_lat_/360*2*math.pi
op_lon_rad_ = op_lon_/360*2*math.pi

# tf hidden units

# tf outputs (location buckets)
if args.output=='loc':
    w_loc = tf.Variable(tf.zeros([2**args.hashsize, len(lochash)]))
    b_loc = tf.Variable(tf.zeros([len(lochash)]))
    loc = tf.sparse_tensor_dense_matmul(x_,w_loc)+b_loc

    # FIXME: lat/lon not set properly
    op_lat = tf.zeros((1,))
    op_lon = tf.zeros((1,))
    op_lat_rad = tf.zeros((1,))
    op_lon_rad = tf.zeros((1,))

# tf outputs (gps coords)
if args.output=='gps':
    w_gps = tf.Variable(tf.zeros([2**args.hashsize, 2]))
    b_gps = tf.Variable([-118.243683,34.052235])
    gps = tf.sparse_tensor_dense_matmul(x_,w_gps) + b_gps
    op_lat = gps[:,0]
    op_lon = gps[:,1]
    op_lat_rad = op_lat/360*2*math.pi
    op_lon_rad = op_lon/360*2*math.pi

# set loss function
hav = lambda x: tf.sin(x/2)**2
havinv = lambda x: 2*3959*tf.asin(tf.sqrt(tf.abs(x)))
if args.loss=='l2':
    loss = tf.reduce_sum((gps - gps_) * (gps - gps_))
if args.loss=='angular':
    loss = tf.reduce_sum((hav(op_lat_rad-op_lat_rad_)
                        +tf.cos(op_lat_rad)*tf.cos(op_lat_rad_)*hav(op_lon_rad-op_lon_rad_)))
if args.loss=='xentropy':
    #labels = tf.to_int64(labels)
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            #labels=labels,
            #logits=logits,
            labels=tf.to_int64(tf.reshape(loc_,[args.batchsize])),
            logits=loc,
            name='xentropy'
            ))

#loss = tf.reduce_sum(havinv(hav(op_lat_rad-op_lat_rad_)+tf.cos(op_lat_rad)*tf.cos(op_lat_rad_)*hav(op_lon_rad-op_lon_rad_)))
#loss = tf.reduce_sum(tf.atan(
    #(tf.sqrt(tf.cos(op_lat)*tf.sin(op_lon-op_lon_)**2
          #+(tf.cos(op_lat)*tf.sin(op_lat_)-tf.sin(op_lat)*tf.cos(op_lat_)*tf.cos(op_lon-op_lon_))**2))/
    #(tf.sin(op_lat)*tf.sin(op_lat_)+tf.cos(op_lat)*tf.cos(op_lat_)*tf.cos(op_lon-op_lon_)
    #)))
#loss = tf.reduce_sum(3959*tf.acos(tf.sqrt(tf.sin(tf.abs(op_lat-op_lat_)/2)**2+tf.cos(op_lat)*tf.cos(op_lat_)*tf.sin(tf.abs(op_lon-op_lon_)/2)**2)))
#loss = tf.reduce_sum(            (       (tf.sin(tf.abs(op_lat-op_lat_)/2)**2+tf.cos(op_lat)*tf.cos(op_lat_)*tf.sin(tf.abs(op_lon-op_lon_)/2)**2)))
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, name='global_step', trainable=False)
optimizer = tf.train.AdamOptimizer(args.learningrate)
train_op = optimizer.minimize(loss, global_step=global_step)

# prepare logging
logdir=args.logdir
#local_log_dir=os.path.join(FLAGS.log_dir_out, '%s-%s.%d-%1.2f-%s.%d-%d'%(FLAGS.dataset,FLAGS.model,FLAGS.seed,FLAGS.induced_bias,FLAGS.same_seed,FLAGS.numproc,FLAGS.procid))
#if tf.gfile.Exists(logdir):
    #tf.gfile.DeleteRecursively(logdir)
tf.gfile.MakeDirs(logdir)

# create tf session
sess = tf.Session()
summary = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=1)
summary_writer = tf.summary.FileWriter(args.logdir, sess.graph)
sess.run(tf.global_variables_initializer())

########################################
print('training')

numlines=0
step=-1
start_time=time.time()
filename=args.filenames[0]
f=gzip.open(filename,'rt')

while True:
    step+=1

    #if step==0:
    if True:
        batch_x_=[]
        batch_gps_=[]
        batch_loc_=[]

        while len(batch_x_) < args.batchsize:

            # load and decode next json entry
            nextline=f.readline()
            if nextline=='':
                f.close()
                f=gzip.open(filename,'rt')
                continue
            data=json.loads(nextline)
            numlines+=1

            # only process entries that contain a tweet
            if data['text']:

                # get features
                batch_x_.append(hv.transform([data['text']]))

                # get gps coords
                if True: #args.output=='gps':
                    if data['geo']:
                        lat=data['geo']['coordinates'][0]
                        lon=data['geo']['coordinates'][1]
                        coord=(lat,lon)
                    else:
                        list=data['place']['bounding_box']['coordinates']
                        coords=[item for sublist in list for item in sublist]
                        lats=[coord[0] for coord in coords]
                        lons=[coord[1] for coord in coords]
                        lat=sum(lats)/float(len(lats))
                        lon=sum(lons)/float(len(lons))
                        coord=(lat,lon)
                    batch_gps_.append(np.array(coord))

                if args.output=='loc':
                    try:
                        full_name=data['place']['full_name']
                        locid=lochash[full_name]
                    except:
                        locid=lochash['']
                    batch_loc_.append(np.array([locid]))

    # create data dictionary
    feed_dict = {
        x_ : mkSparseTensorValue(sp.sparse.vstack(batch_x_)),
    }
    feed_dict[gps_] = np.vstack(batch_gps_)
    feed_dict[loc_] = np.vstack(batch_loc_) #.reshape((args.batchsize,))
    #if args.output=='gps':
        #feed_dict[gps_] = np.vstack(batch_gps_)
    #if args.output=='loc':
        #feed_dict[loc_] = np.vstack(batch_loc_).reshape([args.batchsize]),

    # run the model
    _, loss_value_total, lats, lons, lats_, lons_ = sess.run([train_op, loss, op_lat, op_lon, op_lat_,op_lon_],feed_dict=feed_dict)
    loss_value_ave=loss_value_total/args.batchsize
    coords=zip(lats,lons)
    coords_=zip(lats_,lons_)
    dists=map(lambda (a,b): geopy.distance.great_circle(a,b).miles,zip(coords,coords_))
    dists_ave=np.mean(dists)
    #print(dists_ave)

    # Write the summaries and print an overview fairly often.
    if step % 20 == 0:
        duration = time.time() - start_time
        #print(datetime.datetime.now(),filename,'  %8d: loss(graph) = %5.2f, true_dist = %5.2f' % (step, loss_value_ave, dists_ave))
        print(datetime.datetime.now(),filename,'  %8d: loss(graph) = %5.2f' % (step, loss_value_ave))
        start_time = time.time()
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

    # Save a checkpoint and evaluate the model periodically.
    if (step + 1) % 1000 == 0:
        checkpoint_file = os.path.join(logdir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)

f.close()

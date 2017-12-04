#!/usr/bin/env python

from __future__ import print_function

########################################
# helper functions

def centroid(xs):
    coords=xs[0]
    lats=[coord[0] for coord in coords]
    lons=[coord[1] for coord in coords]
    lat=sum(lats)/float(len(lats))
    lon=sum(lons)/float(len(lons))
    coord=(lat,lon)
    return coord

########################################
print('processing cmd line args')
import argparse
import sys

parser=argparse.ArgumentParser('train a model')

# administration variables
parser.add_argument('--maxsteps',type=int,required=False)
parser.add_argument('--logdir',type=str,default='log')
parser.add_argument('--pickle',type=str)
parser.add_argument('filenames', metavar='N', type=str, nargs='+',
                    help='gzipped files containing tweets in json format')

# model hyperparameters
parser.add_argument('--hashsize',type=int,default=18)
parser.add_argument('--batchsize',type=int,default=100)
parser.add_argument('--learningrate',type=float,default=0.005)
parser.add_argument('--loss',choices=['l2','angular','xentropy'],required=True)
parser.add_argument('--output',choices=['gps','loc'],required=True)

args = parser.parse_args()

if args.output=='loc' and args.pickle==None:
    print('must specify a pickle file for "loc" outputs')
    sys.exit(1)

if args.pickle != None:
    import pickle
    from pprint import pprint
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
    locs=map(lambda (a,b):b,list(reversed(sorted(zip(num_fn['city'].values(),num_fn['city'].keys()))))[0:maxlocs])

    # REMEMBER: twitter stores coordinates in (lon,lat) form instead of (lat,lon)
    locscoords=[centroid(loc_fn['city'][loc]['coordinates']) for loc in locs]
    lats_hash=[lat for (lon,lat) in locscoords]
    lons_hash=[lon for (lon,lat) in locscoords]

    lochash={ loc : i for loc,i in zip(locs,xrange(len(locs))) }
    from pprint import pprint

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

# tf inputs
with tf.name_scope('inputs'):
    x_ = tf.sparse_placeholder(tf.float32)
    loc_ = tf.placeholder(tf.int32, [args.batchsize,1])
    gps_ = tf.placeholder(tf.float32, [args.batchsize,2])
    op_lat_ = gps_[:,0]
    op_lon_ = gps_[:,1]
    op_lat_rad_ = op_lat_/360*2*math.pi
    op_lon_rad_ = op_lon_/360*2*math.pi
    input_size=2**args.hashsize

# tf hidden units
with tf.name_scope('hidden'):
    if False:
        w = tf.Variable(tf.truncated_normal([input_size, 10],
                                            stddev=1.0/math.sqrt(float(input_size))))
        b = tf.Variable(tf.truncated_normal([            10]))
        h = tf.nn.relu(tf.sparse_tensor_dense_matmul(x_,w)+b)

        final_layer=h
        final_layer_size=10
        matmul=tf.matmul

    else:
        final_layer=x_
        final_layer_size=2**args.hashsize
        matmul=tf.sparse_tensor_dense_matmul

# rf outputs
with tf.name_scope('output'):

    # loc buckets
    if args.output=='loc':
        w = tf.Variable(tf.zeros([final_layer_size, len(lochash)]))
        b = tf.Variable(tf.zeros([len(lochash)]))
        loc = matmul(final_layer,w)+b
        probs = tf.nn.softmax(loc)

        np_lats=np.array(lats_hash,dtype=np.float32)
        np_lons=np.array(lons_hash,dtype=np.float32)
        op_lats_hash=tf.convert_to_tensor(np_lats)
        op_lons_hash=tf.convert_to_tensor(np_lons)

        op_lat = tf.reduce_sum(tf.multiply(op_lats_hash,probs),1)
        op_lon = tf.reduce_sum(tf.multiply(op_lons_hash,probs),1)
        gps = tf.transpose(tf.stack([op_lat,op_lon]))

    # gps coords
    if args.output=='gps':
        w = tf.Variable(tf.zeros([final_layer_size, 2]))
        b = tf.Variable([34.052235,-118.243683])
        gps = matmul(final_layer,w) + b
        op_lat = gps[:,0]
        op_lon = gps[:,1]

    # common outputs
    op_lat_rad = op_lat/360*2*math.pi
    op_lon_rad = op_lon/360*2*math.pi

    hav = lambda x: tf.sin(x/2)**2
    squared_angular_dist = ( hav(op_lat_rad-op_lat_rad_)
                    +tf.cos(op_lat_rad)*tf.cos(op_lat_rad_)*hav(op_lon_rad-op_lon_rad_)
                   )
    miles_dist = 2*3959*tf.asin(tf.sqrt(squared_angular_dist))
    miles_dist_ave = tf.reduce_sum(miles_dist)/args.batchsize
    tf.summary.scalar('miles_dist_ave',miles_dist_ave)

# set loss function
if args.loss=='l2':
    loss = tf.reduce_sum((gps - gps_) * (gps - gps_))
if args.loss=='angular':
    loss = tf.reduce_sum(squared_angular_dist)
if args.loss=='xentropy':
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.to_int64(tf.reshape(loc_,[args.batchsize])),
            logits=loc,
            name='xentropy'
            ))
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, name='global_step', trainable=False)
optimizer = tf.train.AdamOptimizer(args.learningrate)
train_op = optimizer.minimize(loss, global_step=global_step)

# prepare logging
logdir=os.path.join(args.logdir,str(args))
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

    if step==0:
    #if True:
        batch_dict={
            x_ : [],
            loc_ : [],
            gps_ : []
        }
        tweets_total=0
        while len(batch_dict[x_]) < args.batchsize:

            # load and decode next json entry
            nextline=f.readline()
            if nextline=='':
                f.close()
                f=gzip.open(filename,'rt')
                continue
            data=json.loads(nextline)
            numlines+=1
            tweets_total+=1

            # only process entries that contain a tweet
            if data['text']:

                # possibly skip locations
                full_name=data['place']['full_name']
                #if args.output=='loc' and not (full_name in lochash):
                if not (full_name in lochash):
                    continue

                # get features
                batch_dict[x_].append(hv.transform([data['text']]))

                # get true output
                if True: #args.output=='gps':
                    if data['geo']:
                        lat=data['geo']['coordinates'][0]
                        lon=data['geo']['coordinates'][1]
                        coord=(lat,lon)
                    else:
                        coord=centroid(data['place']['bounding_box']['coordinates'])
                        # the twitter format stores bounding boxes as (lon,lat) pairs
                        # instead of (lat,lon) pairs, so we need to flip them around
                        coord=(coord[1],coord[0])
                    batch_dict[gps_].append(np.array(coord))

                if True: #args.output=='loc':
                    full_name=data['place']['full_name']
                    locid=lochash[full_name]
                    batch_dict[loc_].append(np.array([locid]))

    # create data dictionary
    def mkSparseTensorValue(m):
        m2=sp.sparse.coo_matrix(m)
        return tf.SparseTensorValue(zip(m2.row,m2.col),m2.data,m2.shape)

    feed_dict = {
        x_ : mkSparseTensorValue(sp.sparse.vstack(batch_dict[x_])),
        gps_ : np.vstack(batch_dict[gps_]),
        loc_ : np.vstack(batch_dict[loc_]),
    }

    # run the model
    _, loss_value_total, lats, lons, lats_, lons_, miles = sess.run([train_op, loss, op_lat, op_lon, op_lat_,op_lon_,miles_dist_ave],feed_dict=feed_dict)
    loss_value_ave=loss_value_total/args.batchsize
    coords=zip(lats,lons)
    coords_=zip(lats_,lons_)
    dists=map(lambda (a,b): geopy.distance.great_circle(a,b).miles,zip(coords,coords_))
    dists_ave=np.mean(dists)

    # Write the summaries and print an overview fairly often.
    if step % 10 == 0:
        duration = time.time() - start_time
        output='  %8d: loss=%1.2E  dist=%1.2E,%1.2E  good=%1.2f' % (
              step
            , loss_value_ave
            , dists_ave
            , miles
            , args.batchsize/float(tweets_total)
            )
        print(datetime.datetime.now(),filename,output)
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        start_time = time.time()

    # Save a checkpoint and evaluate the model periodically.
    if (step + 1) % 1000 == 0:
        checkpoint_file = os.path.join(logdir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)

f.close()

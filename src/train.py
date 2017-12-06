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
parser.add_argument('--output',choices=['naive','aglm','aglm2','proj3d','loc'],required=True)
parser.add_argument('--l1',type=float,default=0.0)
parser.add_argument('--maxloc',type=int)
parser.add_argument('--hidden1',type=int)
parser.add_argument('--lstm',type=int)
parser.add_argument('--cnn',type=int)

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
    if args.maxloc:
        maxlocs=args.maxloc
    else:
        maxlocs=len(num_fn['city'].keys())
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

    # true labels
    loc_ = tf.placeholder(tf.int32, [args.batchsize,1])
    gps_ = tf.placeholder(tf.float32, [args.batchsize,2])
    op_lat_ = gps_[:,0]
    op_lon_ = gps_[:,1]
    op_lat_rad_ = op_lat_/360*2*math.pi
    op_lon_rad_ = op_lon_/360*2*math.pi

    # hash bow inputs
    hash_ = tf.sparse_placeholder(tf.float32)
    hash_reg = args.l1*tf.sparse_reduce_sum(tf.abs(hash_))

        #W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="W")
        #b = tf.Variable(tf.constant(0.1, shape=[num_filters_per_size]), name="b")
        #conv = tf.nn.conv2d(self.input_x, W, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
        #h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        #pooled = tf.nn.max_pool(
            #h,
            #ksize=[1, 1, 3, 1],
            #strides=[1, 1, 3, 1],
            #padding='VALID',
            #name="pool1")

    # lstm inputs
    if args.lstm:
        cell = tf.nn.rnn_cell.BasicRNNCell(200)
        init_state = cell.zero_state(args.batchsize, tf.float32)
        rnn_outputs, final_state = tf.nn.rnn(cell, rnn_inputs, initial_state=init_state)

    # time inputs
    def wrapped(var,length):
        with tf.name_scope('wrapped'):
            scaled=var/length*2*math.pi
            return tf.sin(scaled)
    time_dow_ = tf.placeholder(tf.float32, [args.batchsize,1])
    time_tod_ = tf.placeholder(tf.float32, [args.batchsize,1])

    time_dow_wrapped_ = wrapped(time_dow_,7)
    time_tod_wrapped_ = wrapped(time_tod_,24)

    # overall input
    input_ = hash_
    input_size=2**args.hashsize

# FIXME: cnn inputs
if args.cnn:
    tweetlen=280
    vocabsize=32
    numfilters=32
    filterlen=7
    text_ = tf.placeholder(tf.float32, [args.batchsize,tweetlen,vocabsize])
    text_reshaped = tf.reshape(text_,[args.batchsize,tweetlen,vocabsize,1])

    with tf.name_scope('conv1'):
        w = tf.Variable(tf.truncated_normal([filterlen,vocabsize,1,numfilters],stddev=0.05))
        b = tf.Variable(tf.constant(0.1,shape=[numfilters]))
        conv = tf.nn.conv2d(text_reshaped, w, strides=[1,1,1,1], padding='VALID')
        h = tf.nn.tanh(tf.nn.bias_add(conv,b))
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, 3, 1, 1],
            strides=[1, 3, 1, 1],
            padding='VALID')

    with tf.name_scope('conv2'):
        w = tf.Variable(tf.truncated_normal([filterlen,1,numfilters,numfilters],stddev=0.05))
        b = tf.Variable(tf.constant(0.1,shape=[numfilters]))
        conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
        h = tf.nn.tanh(tf.nn.bias_add(conv,b))
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, 3, 1, 1],
            strides=[1, 3, 1, 1],
            padding='VALID')

    filterlen=3
    with tf.name_scope('conv3'):
        w = tf.Variable(tf.truncated_normal([filterlen,1,numfilters,numfilters],stddev=0.05))
        b = tf.Variable(tf.constant(0.1,shape=[numfilters]))
        conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
        h = tf.nn.tanh(tf.nn.bias_add(conv,b))
        pooled = h

    with tf.name_scope('conv4'):
        w = tf.Variable(tf.truncated_normal([filterlen,1,numfilters,numfilters],stddev=0.05))
        b = tf.Variable(tf.constant(0.1,shape=[numfilters]))
        conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
        h = tf.nn.tanh(tf.nn.bias_add(conv,b))
        pooled = h

    with tf.name_scope('conv5'):
        w = tf.Variable(tf.truncated_normal([filterlen,1,numfilters,numfilters],stddev=0.05))
        b = tf.Variable(tf.constant(0.1,shape=[numfilters]))
        conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
        h = tf.nn.tanh(tf.nn.bias_add(conv,b))
        pooled = h

    with tf.name_scope('conv6'):
        w = tf.Variable(tf.truncated_normal([filterlen,1,numfilters,numfilters],stddev=0.05))
        b = tf.Variable(tf.constant(0.1,shape=[numfilters]))
        conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
        h = tf.nn.tanh(tf.nn.bias_add(conv,b))
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, 3, 1, 1],
            strides=[1, 3, 1, 1],
            padding='VALID')

    #w = tf.Variable(tf.truncated_normal([tweetlen,vocabsize,numfilters],stddev=0.05))
    #b = tf.Variable(tf.constant(0.1,shape=[numfilters]))
    #conv = tf.nn.conv1d(text_, w, stride=1, padding='VALID')
    #h = tf.nn.relu(tf.nn.bias_add(conv,b))

    #print(w.get_shape())
    #print(conv.get_shape())
    #print(h.get_shape())
    #print(pooled.get_shape())
    #sys.exit(0)

    #last=text_reshaped
    last=pooled

    matmul=tf.matmul
    #final_layer_size=int(last.get_shape()[1]*last.get_shape()[3])
    #final_layer=tf.reshape(last,[args.batchsize,final_layer_size])
    input_size=int(last.get_shape()[1]*last.get_shape()[2]*last.get_shape()[3])
    input_=tf.reshape(last,[args.batchsize,input_size])

# tf hidden units
with tf.name_scope('hidden1'):
    if args.hidden1:
        w = tf.Variable(tf.truncated_normal([input_size, args.hidden1],
                                            stddev=1.0/math.sqrt(float(input_size))))
        b = tf.Variable(tf.truncated_normal([args.hidden1]))
        #h = tf.nn.relu(tf.sparse_tensor_dense_matmul(input_,w)+b)
        h = tf.nn.relu(tf.matmul(input_,w)+b)

        final_layer=h
        final_layer_size=args.hidden1
        matmul=tf.matmul

    else:
        final_layer=input_
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

    # treat gps coords as R^2
    if args.output=='naive':
        w = tf.Variable(tf.zeros([final_layer_size, 2]))
        b = tf.Variable([34.052235,-118.243683])
        #b = tf.Variable(tf.zeros([2]))
        gps = matmul(final_layer,w) + b
        op_lat = gps[:,0]
        op_lon = gps[:,1]

    # angular generalized linear model
    # See: "Regression Models for Angular Response" by Fisher and Lee
    if args.output=='aglm':
        w = tf.Variable(tf.zeros([final_layer_size, 2]))
        #b = tf.Variable([0.6745,-2])
        b = tf.Variable(tf.zeros([2]))
        response = matmul(final_layer,w) + b
        op_lat = tf.atan(response[:,0])*360/2/math.pi
        op_lon = tf.atan(response[:,1])*360/math.pi
        gps = tf.stack([op_lat,op_lon],1)
        #op_lat = tf.Print(op_lat,[response,gps])

    # model gps coordinates in R^3
    if args.output=='proj3d':
        w = tf.Variable(tf.zeros([final_layer_size, 3]))
        b = tf.Variable([0.1,0,0])
        r3 = matmul(final_layer,w) + b
        norm = tf.sqrt(tf.reduce_sum(r3*r3,1))
        r3normed = r3/tf.stack([norm,norm,norm],1)
        #op_lon = tf.asin(tf.minimum(1.0,r3normed[:,2]))
        #op_lat = tf.asin(tf.minimum(1.0,r3normed[:,1]))*tf.acos(tf.minimum(1.0,r3normed[:,2]))
        op_lon = tf.asin(r3normed[:,2])
        op_lat = tf.asin(r3normed[:,1])*tf.acos(r3normed[:,2])
        gps = tf.stack([op_lat,op_lon],1)

        op_lon=tf.Print(op_lon,[gps,b,norm])

    # fancy gps coords
    #if args.output=='atlas':
        #numatlas=10
        #w = tf.Variable(tf.zeros([final_layer_size,2]))
        #b = tf.Variable([1.0,1.0])
        #x = matmul(final_layer,w)+b
        #xnorm_squared = tf.reduce_sum(x*x)

        #pad=tf.zeros([args.batchsize,1])
        #print(x.get_shape())
        #print(pad.get_shape())
        #print(tf.stack([x,pad]).get_shape())

        #x3 = tf.stack([x,tf.zeros([1])])
        #p3 = tf.Variable([1.0,0,0])
        #q3 = p*(xnorm_squared-1)/(xnorm_squared+1)+x3*2/(xnorm_squared+1)

        op_lat = tf.asin(q3[2])         *360/(2*math.pi)
        op_lon = tf.atan(q3[1]/q3[0])   *360/(2*math.pi)
        gps = tf.stack([op_lat,op_lon],1)

        pass

    # common outputs
    op_lat_rad = op_lat/360*2*math.pi
    op_lon_rad = op_lon/360*2*math.pi

    hav = lambda x: tf.sin(x/2)**2
    squared_angular_dist = ( hav(op_lat_rad-op_lat_rad_)
                    +tf.cos(op_lat_rad)*tf.cos(op_lat_rad_)*hav(op_lon_rad-op_lon_rad_)
                   )
    miles_dist = 2*3959*tf.asin(tf.sqrt(squared_angular_dist))
    miles_dist_ave = tf.reduce_sum(miles_dist)/args.batchsize

    #test=tf.Print(miles_dist_ave,[miles_dist_ave])
    #tf.summary.scalar('miles_dist_ave',test)

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

loss_regularized=loss+hash_reg

loss_sum=tf.summary.scalar('loss', loss)

# optimization nodes
global_step = tf.Variable(0, name='global_step', trainable=False)
optimizer = tf.train.AdamOptimizer(args.learningrate)
train_op = optimizer.minimize(loss, global_step=global_step)

# prepare logging
localdir='output=%s,maxloc=%s,loss=%s,hidden1=%s,hashsize=%d,learningrate=%f,batchsize=%d'%(
    args.output,
    str(args.maxloc),
    args.loss,
    str(args.hidden1),
    args.hashsize,
    args.learningrate,
    args.batchsize
    )
logdir=os.path.join(args.logdir,localdir)
tf.gfile.MakeDirs(logdir)

# create tf session
sess = tf.Session()
summary = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=1)
summary_writer = tf.summary.FileWriter(logdir, sess.graph)
sess.run(tf.global_variables_initializer())

########################################
print('training')

numlines=0
step=-1
epochs=0
loss_value_total=0
miles_total=0
start_time=time.time()
filename=args.filenames[0]
f=gzip.open(filename,'rt')

while True:
    step+=1

    #if step==0:
    if True:
        batch_dict={
            hash_ : [],
            text_ : [],
            loc_ : [],
            gps_ : [],
        }
        tweets_total=0
        while len(batch_dict[hash_]) < args.batchsize:

            # load and decode next json entry
            nextline=f.readline()
            if nextline=='':
                epochs+=1
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

                # hash features
                batch_dict[hash_].append(hv.transform([data['text']]))

                # text features
                bytes=data['text'].encode('utf-8')
                encodedtext=np.zeros([1,tweetlen,vocabsize])
                #for i in range(min(tweetlen,len(bytes))):
                    #encodedtext[0][i][ord(bytes[i])%vocabsize]=1
                #if len(bytes)>tweetlen:
                    #print('len(bytes)=',len(bytes),'; text=')
                    #print(bytes)
                def myhash(i):
                    if i>=ord('a') or i<=ord('z'):
                        val=i-ord('a')
                    else:
                        val=5381*i
                    return val%vocabsize

                for i in range(len(data['text'])):
                    encodedtext[0][i][myhash(ord(data['text'][i]))%vocabsize]=1
                batch_dict[text_].append(encodedtext)

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
        hash_ : mkSparseTensorValue(sp.sparse.vstack(batch_dict[hash_])),
        text_ : np.vstack(batch_dict[text_]),
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

    loss_value_total+=loss_value_ave
    miles_total+=dists_ave

    # Write the summaries and print an overview fairly often.
    stepdelta=100
    if step % stepdelta == 0:
        duration = time.time() - start_time
        output='  %8d/%2d: loss=%1.2E  dist=%1.2E  good=%1.2f' % (
              step
            , epochs
            , loss_value_total/stepdelta
            , miles_total/stepdelta
            , args.batchsize/float(tweets_total)
            )
        print(datetime.datetime.now(),filename,output)
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        #summary_writer.flush()
        start_time = time.time()

        loss_value_total=0
        miles_total=0

    # Save a checkpoint and evaluate the model periodically.
    if (step + 1) % 1000 == 0:
        checkpoint_file = os.path.join(logdir, 'model.ckpt')
        saver.save(sess, checkpoint_file, global_step=step)

f.close()

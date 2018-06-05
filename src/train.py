#!/usr/bin/env python

from __future__ import print_function

# set unbuffered output
import os
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

########################################
print('processing cmd line args')
import argparse

parser=argparse.ArgumentParser('train a model')

# administration variables
parser.add_argument('--log_dir',type=str,default='log')
parser.add_argument('--log_name',type=str,default=None)
parser.add_argument('--stepsave',type=int,default=10000)
parser.add_argument('--seed',type=int,default=0)
parser.add_argument('--name',type=str)
parser.add_argument('--task',choices=['train','vis'],default='train')

# debug variables
parser.add_argument('--no_checkpoint',action='store_true')
parser.add_argument('--repeat_batch',action='store_true')

# model hyperparameters
parser.add_argument('--data',type=str,required=True)
parser.add_argument('--data_summary',type=str,default=None)
parser.add_argument('--data_sample',choices=['uniform','fancy'],default='uniform')
parser.add_argument('--data_style',choices=['online','batch'],default='batch')
parser.add_argument('--max_open_files',type=int,default=96)
parser.add_argument('--initial_weights',type=str,default=None)

import model
model.update_parser(parser)

args = parser.parse_args()

if args.data_style=='online':
    args.max_open_files=1

if args.task=='vis':
    args.batchsize=1

print('args=',args)

########################################
print('importing libraries')

import copy
from collections import defaultdict
import datetime
import gzip
import math
import os
import simplejson as json
import time

#import geopy.distance
import numpy as np
import scipy as sp

import sklearn.feature_extraction.text
hv=sklearn.feature_extraction.text.HashingVectorizer(n_features=2**args.bow_hashsize,norm=None)

import random
random.seed(args.seed)

########################################
print('initializing tensorflow')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.set_random_seed(args.seed)

import hash

input_tensors={
    'country_' : tf.placeholder(tf.int64, [args.batchsize,1],name='country_'),
    'text_' : tf.placeholder(tf.float32, [args.batchsize,model.tweetlen,args.cnn_vocabsize],name='text_'),
    'gps_' : tf.placeholder(tf.float32, [args.batchsize,2], name='gps_'),
    'loc_' : tf.placeholder(tf.int64, [args.batchsize,1],name='loc_'),
    'timestamp_ms_' : tf.placeholder(tf.float32, [args.batchsize,1], name='timestamp_ms_'),
    'lang_' : tf.placeholder(tf.int32, [args.batchsize,1], 'lang_'),
    'newuser_' : tf.placeholder(tf.float32, [args.batchsize,1],name='newuser_')
}

op_metrics,op_loss_regularized = model.inference(args,input_tensors)

# optimization nodes
with tf.name_scope('optimization'):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    if args.decay is None:
        learningrate = args.learningrate
    else:
        learningrate = tf.train.inverse_time_decay(
            args.learningrate,
            global_step,
            args.decay,
            1.0,
            staircase=args.loss_staircase)
    tf.summary.scalar('learningrate',learningrate)
    if args.optimizer=='adam':
        optimizer = tf.train.AdamOptimizer(learningrate)
    elif args.optimizer=='sgd':
        optimizer = tf.train.MomentumOptimizer(learningrate,args.momentum)
    train_op = optimizer.minimize(op_loss_regularized, global_step=global_step)

########################################
print('preparing logging')
if args.log_name is None:
    args.log_name='data=%s,style=%s,input=%s,output=%s,loss_weights=%s,loss=%s,full=%s,learningrate=%1.1E,l1=%1.1E,l2=%1.1E,dropout=%1.1f,bow_hashsize=%d,loc_hashsize=%s,batchsize=%d'%(
        os.path.basename(args.data),
        args.data_style,
        args.input,
        args.output,
        args.loss_weights,
        args.pos_loss,
        str(args.full),
        args.learningrate,
        args.l1,
        args.l2,
        args.dropout,
        args.bow_hashsize,
        str(args.loc_hashsize),
        args.batchsize
        )
log_dir=os.path.join(args.log_dir,args.log_name)
if not args.no_checkpoint:
    tf.gfile.MakeDirs(log_dir)
print('log_dir=',log_dir)

# create tf session
config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config=config)

saver = tf.train.Saver(max_to_keep=100000)
if not args.no_checkpoint:
    for k,(v,_) in op_metrics.iteritems():
        tf.summary.scalar(k,v)
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)

reset_global_vars=tf.global_variables_initializer()
reset_local_vars=tf.local_variables_initializer()

if args.initial_weights:
    saver.restore(sess,args.initial_weights)
    print('model restored from ',args.initial_weights)

sess.graph.finalize()
sess.run(reset_global_vars)
sess.run(reset_local_vars)

########################################
print('allocate dataset files')

files_all=[]
for path_date in os.listdir(args.data):
    path_date_full=os.path.join(args.data,path_date)
    if os.path.isdir(path_date_full):
        for path_hour in os.listdir(path_date_full):
            files_all.append(os.path.join(path_date_full,path_hour))

files_all.sort()
files_all=files_all[:3000]

files_train=[]
files_test=[]
files_valid=[]

file_count=2
file_step=13
if args.data_style == 'online':
    files_train=files_all
elif args.data_style == 'batch':
    for file in files_all:
        if file_count%file_step==0:
            files_test.append(file)
        elif file_count%file_step==1:
            files_valid.append(file)
        else:
            files_train.append(file)

########################################
print('looping through data')
print('  task: ',args.task)

stats_epoch={'count':0}
def reset_stats_epoch():
    stats_epoch['steps']=0
    stats_epoch['new']=False
    stats_epoch['loss']=0
    stats_epoch['dist']=0
    stats_epoch['err']=defaultdict(float)
    stats_epoch['start_time']=time.time()
    stats_epoch['decoding_time']=0
reset_stats_epoch()
stats_epoch_prev=stats_epoch

stats_step={'count':0,'numlines':0}
def reset_stats_step():
    stats_step['loss']=0
    stats_step['dist']=0
    stats_step['err']=defaultdict(float)
    stats_step['validtweets']=0
    stats_step['start_time']=time.time()
    stats_step['decoding_time']=0
    if not args.repeat_batch:
        stats_step['numlines']=0
reset_stats_step()

userids=set([])

files_remaining=copy.deepcopy(files_train)
open_files=[]

while True:
    stats_step['count']+=1
    stats_epoch['steps']+=1

    if not args.repeat_batch or stats_step['count']==0:
        decoding_time_start=time.time()
        batch=[]
        while len(batch) < args.batchsize:

            # load and decode next json entry
            while len(open_files)<args.max_open_files and len(files_remaining)>0:
                if args.data_style=='online':
                    index=0
                else:
                    if args.data_sample == 'uniform':
                        index=random.randrange(len(files_remaining))
                    elif args.data_sample == 'fancy':
                        choices=3
                        choice=np.random.randint(choices)+1
                        index=random.randrange(1+int((choice/float(choices))*len(files_remaining)))
                        print('index=',index)

                filename=files_remaining[index]
                files_remaining.pop(index)
                try:
                    open_files.append(gzip.open(filename,'rt'))
                except Exception as e:
                    print('gzip.open failed on ',filename,file=sys.stderr)
                    print(e,file=sys.stderr)
                    continue
                print('  opening [%s]; files remaining: %d/%d; buffer state: %d/%d'%(
                    filename,
                    len(files_remaining),
                    len(files_train),
                    len(open_files),
                    args.max_open_files,
                    ))

            if len(open_files)==0:
                if args.data_style=='online':
                    print('done.')
                    sys.exit(0)
                else:
                    stats_epoch['new']=True
                    stats_epoch['count']+=1
                    files_remaining=copy.deepcopy(files_train)
                    continue

            index=random.randrange(len(open_files))
            try:
                nextline=open_files[index].readline()
                if nextline=='':
                    raise ValueError('done')
            except Exception as e:
                open_files[index].close()
                open_files.pop(index)
                continue

            try:
                stats_step['numlines']+=1
                data=model.json2dict(args,nextline)
                batch.append(data)
                stats_step['validtweets']+=1
            except Exception as e:
                print('current file=',open_files[index].name)
                print(e)
                continue

    feed_dict=model.mk_feed_dict(args,batch)
    decoding_time_stop=time.time()
    stats_step['decoding_time']+=decoding_time_stop-decoding_time_start
    stats_epoch['decoding_time']+=decoding_time_stop-decoding_time_start

    ####################
    if args.task=='vis':
        [country_orig, gps_orig] = sess.run(
            [country_softmax,gps]
            , feed_dict=feed_dict
            )
        print('data[text]=',data['text'])
        print('data[lang]=',data['lang'])
        print('data[place][country_code]=',data['place']['country_code'])
        print('data[gps]=',feed_dict[gps_])

        for i in range(0,len(data['text'])):
            feed_dict_i=feed_dict
            feed_dict_i[text_][0][i] = np.zeros([args.cnn_vocabsize])
            [country_i,gps_i] = sess.run(
                [country_softmax,gps]
                , feed_dict=feed_dict_i
                )
            country_diff=np.linalg.norm(country_orig-country_i)
            gps_diff=np.linalg.norm(gps_orig-gps_i)
            print('%4d , %c : %e %e'%(i,data['text'][i],country_diff,gps_diff))
        sys.exit(0)

    ####################
    if args.task=='train':

        # run the model
        _, metrics = sess.run(
            [ train_op, op_metrics ]
            , feed_dict=feed_dict
            )

        # Write the summaries and print an overview fairly often.
        if (stats_step['count'] < 10 or
           (stats_step['count'] < 100 and stats_step['count']%10 == 0) or
           (stats_step['count'] < 10000 and stats_step['count']%100 == 0) or
           (stats_step['count']%1000 == 0)):
            output='  %8d/%4d: good=%1.2f  dec=%1.2f %s' % (
                  stats_step['count']
                , stats_epoch['count']
                , stats_step['validtweets']/float(stats_step['numlines'])
                , stats_step['decoding_time']/float(time.time()-stats_step['start_time'])
                , '' #str(metrics)
                )
            print(datetime.datetime.now(),output)

            # quit if we've gotten nan values
            if math.isnan(stats_step['loss']):
                raise ValueError('NaN loss')

            # save summaries if not debugging
            if not args.no_checkpoint:
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, stats_step['count']*args.batchsize)
                summary_writer.flush()

            # save model if not debugging
            if stats_step['count'] % args.stepsave == 0 and not args.no_checkpoint:
                checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=stats_step['count'])

            # reset step variables and update epoch counters
            for k,(v,_) in metrics.iteritems():
                stats_epoch['err'][k]+=metrics[k][0]
            reset_stats_step()
            sess.run(reset_local_vars)

        if stats_epoch['new']:
            err=''
            for k,v in stats_epoch['err'].iteritems():
                if k=='dist':
                    err='%s %s:%1.2E'%(err,k,v)
                else:
                    err='%s %s:%1.4f'%(err,k,v)

            print('--------------------------------------------------------------------------------')
            print('epoch %d' % stats_epoch['count'])
            print('  time:  %s ' % str(datetime.timedelta(seconds=time.time() - stats_epoch['start_time'])))
            print('  steps: %d ' % stats_epoch['steps'])
            print('  err:  %s' % err )
            print('--------------------------------------------------------------------------------')
            stats_epoch_prev=copy.deepcopy(stats_epoch)

            # save model if not debugging
            if not args.no_checkpoint:
                checkpoint_file = os.path.join(log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=stats_step['count'])

            # reset epoch variables
            reset_stats_epoch()

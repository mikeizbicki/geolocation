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

# debug variables
parser.add_argument('--no_checkpoint',action='store_true')
parser.add_argument('--repeat_batch',action='store_true')
parser.add_argument('--tfdbg',action='store_true')

# model hyperparameters
parser.add_argument('--data',type=str)
parser.add_argument('--data_summary',type=str,default=None)
parser.add_argument('--data_sample',choices=['uniform','fancy'],default='uniform')
parser.add_argument('--data_style',choices=['online','batch'],default='batch')
parser.add_argument('--max_open_files',type=int,default=96)
parser.add_argument('--initial_weights',type=str,default=None)
parser.add_argument('--multiepoch',action='store_true')

import model
model.update_parser(parser)

args = parser.parse_args()

if args.data_style=='online':
    args.max_open_files=1

# load arguments from file if applicable
if args.initial_weights:
    print('  loading arguments from %s/args.json'%args.initial_weights)
    import simplejson as json
    with open(args.initial_weights+'/args.json','r') as f:
        args_str=f.readline()

    args=argparse.Namespace(**json.loads(args_str))
    args=parser.parse_args(namespace=args)

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

import random
random.seed(args.seed)

########################################
print('initializing tensorflow')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.set_random_seed(args.seed)

import myhash

input_tensors={
    'country_' : tf.placeholder(tf.int64, [args.batchsize,1],name='country_'),
    'text_' : tf.placeholder(tf.float32, [args.batchsize,model.tweetlen,args.cnn_vocabsize],name='text_'),
    'gps_' : tf.placeholder(tf.float32, [args.batchsize,2], name='gps_'),
    'loc_' : tf.placeholder(tf.int64, [args.batchsize,1],name='loc_'),
    'timestamp_ms_' : tf.placeholder(tf.float32, [args.batchsize,1], name='timestamp_ms_'),
    'lang_' : tf.placeholder(tf.int32, [args.batchsize,1], 'lang_'),
    'newuser_' : tf.placeholder(tf.float32, [args.batchsize,1],name='newuser_'),
    'hash_' : tf.sparse_placeholder(tf.float32,name='hash_'),
}

op_metrics,op_loss_regularized,op_losses,op_outputs = model.inference(args,input_tensors)
op_summaries=model.metrics2summaries(args,op_metrics)

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
    #train_op = optimizer.minimize(op_loss_regularized, global_step=global_step)

    gradients, variables = zip(*optimizer.compute_gradients(op_loss_regularized))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

########################################
print('preparing logging')
if args.log_name is None:
    args.log_name='data=%s,style=%s,input=%s,output=%s,pos_type=%s,gmm_components=%d,loss_weights=%s,loss=%s,full=%s,learningrate=%1.1E,l1=%1.1E,l2=%1.1E,dropout=%1.1f,batchsize=%d'%(
        os.path.basename(args.data),
        args.data_style,
        args.input,
        args.output,
        args.pos_type,
        args.gmm_components,
        args.loss_weights,
        args.pos_loss,
        str(args.full),
        args.learningrate,
        args.l1,
        args.l2,
        args.dropout,
        #args.bow_hashsize,
        #str(args.loc_hashsize),
        args.batchsize
        )
log_dir=os.path.join(args.log_dir,args.log_name)
if not args.no_checkpoint:
    tf.gfile.MakeDirs(log_dir)
print('log_dir=',log_dir)

# save args
import simplejson as json
args_str=json.dumps(vars(args))
with open(log_dir+'/args.json','w') as f:
    f.write(args_str)

# create tf session
config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config=config)

if args.tfdbg:
    from tensorflow.python import debug as tf_debug
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    def my_filter_callable(datum, tensor):
        if ('dbgloss' in datum.node_name or
            'kappa' in datum.node_name
            #'optimization' in datum.node_name
           ) and 'summar' not in datum.node_name:
            return tf_debug.has_inf_or_nan(datum,tensor)
        #print('datum=',datum)
        #print('tensor=',tensor)
        #return len(tensor.shape) == 0 and tensor == 0.0
    sess.add_tensor_filter('my_filter', my_filter_callable)


saver = tf.train.Saver(max_to_keep=10)
if not args.no_checkpoint:
    for k,(v,_) in op_summaries.iteritems():
        tf.summary.scalar(k,v)
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)

reset_global_vars=tf.global_variables_initializer()
reset_local_vars=tf.local_variables_initializer()

sess.run(reset_global_vars)
sess.run(reset_local_vars)

if args.initial_weights:
    # see https://stackoverflow.com/questions/41621071/restore-subset-of-variables-in-tensorflow
    print('restoring model')
    chkpt_file=tf.train.latest_checkpoint(args.initial_weights)
    print('  chkpt_file=',chkpt_file)
    reader = tf.pywrap_tensorflow.NewCheckpointReader(chkpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    var_dict={}
    for k in sorted(var_to_shape_map):
        try:
            var=tf.get_default_graph().get_tensor_by_name(k+":0")
            if var.get_shape() == var_to_shape_map[k]:
                var_dict[k]=var
            else:
                print('  not restoring',k,'; old=',var_to_shape_map[k],'; new=',var.get_shape())
        except:
            pass
    #print('  restored_vars=',var_dict)
    loader = tf.train.Saver(var_dict)
    loader.restore(sess, chkpt_file)
    #loader.restore(sess, args.initial_weights)
    #saver.restore(sess,args.initial_weights)
    print('  model restored from ',args.initial_weights)

sess.graph.finalize()

########################################
print('allocate dataset files')

files_all=[]
for path_date in os.listdir(args.data):
    path_date_full=os.path.join(args.data,path_date)
    if os.path.isdir(path_date_full):
        for path_hour in os.listdir(path_date_full):
            files_all.append(os.path.join(path_date_full,path_hour))

files_all.sort()
#files_all=files_all[:3000]

#files_train=[]
#files_test=[]
#files_valid=[]
#
#file_count=2
#file_step=13
#if args.data_style == 'online':
    #files_train=files_all
#elif args.data_style == 'batch':
    #for file in files_all:
        #if file_count%file_step==0:
            #files_test.append(file)
        #elif file_count%file_step==1:
            #files_valid.append(file)
        #else:
            #files_train.append(file)

########################################
print('looping through data')

stats_epoch={'count':0}
def reset_stats_epoch():
    stats_epoch['steps']=0
    stats_epoch['new']=False
    stats_epoch['loss']=0
    stats_epoch['dist']=0
    stats_epoch['err']=defaultdict(float)
    stats_epoch['start_time']=time.time()
    stats_epoch['decoding_time']=0
    stats_epoch['run_time']=0
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
    stats_step['decoding_time_model']=0
    stats_step['run_time']=0
    if not args.repeat_batch:
        stats_step['numlines']=0
reset_stats_step()

userids=set([])

files_remaining=copy.deepcopy(files_all)
open_files=[]

random.seed(args.seed)
np.random.seed(args.seed)

# function that returns the next batch
def mk_batch():
    global files_remaining
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
                len(files_all),
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
                files_remaining=copy.deepcopy(files_all)
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

        #if batch==[]:
            #print('nextline=',nextline)

        #try:
        if True:
            decoding_time_model_start=time.time()
            stats_step['numlines']+=1
            data=model.json2dict(args,nextline)
            batch.append(data)
            stats_step['validtweets']+=1
            decoding_time_model_stop=time.time()
            stats_step['decoding_time_model']+=decoding_time_model_stop-decoding_time_model_start
        #except Exception as e:
            #print('current file=',open_files[index].name)
            #print(e)
            #continue

    feed_dict=model.mk_feed_dict(args,batch)
    return feed_dict

# setup multiprocessing
import multiprocessing as mp
queue = mp.Queue(maxsize=20)

def mk_batches():
    while True:
        feed_dict=mk_batch()
        queue.put(feed_dict)

process=mp.Process(target=mk_batches)
process.start()

# loop through training data
while True:
    stats_step['count']+=1
    stats_epoch['steps']+=1

    decoding_time_start=time.time()
    if not args.repeat_batch or stats_step['count']==0:
        feed_dict=queue.get() #mk_batch()
    decoding_time_stop=time.time()
    stats_step['decoding_time']+=decoding_time_stop-decoding_time_start
    stats_epoch['decoding_time']+=decoding_time_stop-decoding_time_start

    # run the model
    run_time_start=time.time()
    _, metrics = sess.run(
        [ train_op, op_summaries ]
        , feed_dict=feed_dict
        )
    run_time_stop=time.time()
    stats_step['run_time']+=run_time_stop-run_time_start

    # Write the summaries and print an overview fairly often.
    if (stats_step['count'] < 10 or
       (stats_step['count'] < 1000 and stats_step['count']%10 == 0) or
       (stats_step['count'] < 10000 and stats_step['count']%100 == 0) or
       (stats_step['count']%1000 == 0)):
        output='  %8d/%4d: good=%1.2f  dec=%1.2f  dec_m=%1.2f run=%1.2f' % (
              stats_step['count']
            , stats_epoch['count']
            , 0.0 #stats_step['validtweets']/float(stats_step['numlines'])
            , stats_step['decoding_time']/float(time.time()-stats_step['start_time'])
            , stats_step['decoding_time_model']/float(time.time()-stats_step['start_time'])
            , stats_step['run_time']/float(time.time()-stats_step['start_time'])
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
        if (stats_step['count'] in [1,100,1000] or stats_step['count'] % args.stepsave == 0) and not args.no_checkpoint:
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

        # exit if needed
        if not args.multiepoch:
            sys.exit(0)

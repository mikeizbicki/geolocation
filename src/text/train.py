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
parser.add_argument('--no_staging_area',action='store_true')
parser.add_argument('--loss_output',action='store_true')
parser.add_argument('--no_optimizer',action='store_true')

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
parser.add_argument('--multiepoch',action='store_true')

parser.add_argument('--initial_weights',type=str,default=None)
parser.add_argument('--train_list',type=str,nargs='*',default=[])

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

# get cuda devices
try:
    cuda_devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if cuda_devices==['NoDevFiles']:
        cuda_devices=['no_cuda']
except KeyError:
    cuda_devices=['no_cuda']
#cuda_devices=['3'] #,'2','3']
num_devices=float(len(cuda_devices))
print('cuda_devices=',cuda_devices)

# inputs
placeholders={
    'country_' : tf.placeholder(tf.int64, [args.batchsize,1],name='country_'),
    'wnut2016_' : tf.placeholder(tf.int64, [args.batchsize,1],name='wnut2016_'),
    'text_' : tf.placeholder(tf.float32, [args.batchsize,model.tweetlen,args.cnn_vocabsize],name='text_'),
    'gps_' : tf.placeholder(tf.float32, [args.batchsize,2], name='gps_'),
    #'loc_' : tf.placeholder(tf.int64, [args.batchsize,1],name='loc_'),
    'timestamp_ms_' : tf.placeholder(tf.float32, [args.batchsize,1], name='timestamp_ms_'),
    'lang_' : tf.placeholder(tf.int32, [args.batchsize,1], 'lang_'),
    'newuser_' : tf.placeholder(tf.float32, [args.batchsize,1],name='newuser_'),
    #'hash_' : tf.sparse_placeholder(tf.float32,name='hash_'),
    }
#input_queue=tf.contrib.staging.StagingArea(
    #capacity=2*num_devices,
    #dtypes=map(lambda x: x.dtype, placeholders.values()),
    #shapes=map(lambda x: x.get_shape(), placeholders.values()),
    #names=placeholders.keys(),
    #)
#input_queue_size=input_queue.size()
#input_queue_enqueue=input_queue.put(placeholders)

# generate model on each device
device_inputs={}
device_metrics={}
device_loss_reg={}
device_grads={}
device_grads_summary={}
device_set_vars={}
device_input_queue={}
device_input_queue_size={}
device_input_queue_enqueue={}
device_input_queue_dequeue={}
device_input_queue_peek={}

reuse_variables=False
for device in cuda_devices:
    devicestr="/device:GPU:"+device
    if device=='no_cuda':
        devicestr='/cpu:0'
    print('devicestr=',devicestr)
    with tf.device(devicestr):
        with tf.variable_scope('input'+device):
            device_inputs[device]={
                'country_' : tf.placeholder(tf.int64, [args.batchsize,1],name='country_'),
                'wnut2016_' : tf.placeholder(tf.int64, [args.batchsize,1],name='wnut2016_'),
                'gps_' : tf.placeholder(tf.float32, [args.batchsize,2], name='gps_'),
                #'loc_' : tf.placeholder(tf.int64, [args.batchsize,1],name='loc_'),
                'newuser_' : tf.placeholder(tf.float32, [args.batchsize,1],name='newuser_'),
                #'hash_' : tf.sparse_placeholder(tf.float32,name='hash_'),
                'lang_' : tf.placeholder(tf.int32, [args.batchsize,1], 'lang_'),
            }
            if 'time' in args.input:
                device_inputs[device]['timestamp_ms_'] = tf.placeholder(tf.float32, [args.batchsize,1], name='timestamp_ms_')
            #if 'lang' in args.input:
                #device_inputs[device]['lang_'] = tf.placeholder(tf.int32, [args.batchsize,1], 'lang_')
            if 'cnn' in args.input:
                device_inputs[device]['text_'] =  tf.placeholder(tf.float32, [args.batchsize,model.tweetlen,args.cnn_vocabsize],name='text_')
            if 'bow' in args.input:
                if args.bow_dense:
                    device_inputs[device]['hash_'] = tf.placeholder(tf.float32,name='hash_')
                else:
                    device_inputs[device]['hash_'] = tf.sparse_placeholder(tf.float32,name='hash_')

        if not args.no_staging_area:
            device_input_queue[device]=tf.contrib.staging.StagingArea(
                dtypes=map(lambda x: x.dtype, device_inputs[device].values()),
                shapes=map(lambda x: x.get_shape(), device_inputs[device].values()),
                names=device_inputs[device].keys(),
                )
            device_input_queue_size[device]=device_input_queue[device].size()
            device_input_queue_enqueue[device]=device_input_queue[device].put(device_inputs[device])
            device_input_queue_dequeue[device]=device_input_queue[device].get()
            device_input_queue_peek[device]=device_input_queue[device].peek(0)
            device_input=device_input_queue_dequeue[device]
        else:
            device_input=device_inputs[device]

        device_metrics[device],device_loss_reg[device],op_losses,op_losses_unreduced,op_outputs = model.inference(
            args,
            device_input,
            reuse_variables,
            )

        reuse_variables=True
        with tf.variable_scope('optimization'):
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

            if args.train_list == []:
                trainable_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                print('training all variables')
            else:
                trainable_vars=[]
                print('trainable variables:')
                for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                    for r in args.train_list:
                        if r in v.name:
                            print('  ',v.name)
                            trainable_vars.append(v)

            gradients, variables = zip(*optimizer.compute_gradients(
                device_loss_reg[device],
                var_list=trainable_vars,
                colocate_gradients_with_ops=True,
                ))
            device_grads_summary[device]=sum(map(lambda x: tf.reduce_sum(x),gradients))
            #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            #device_grads[device]=(optimizer.compute_gradients(device_loss_reg[device]))
            device_grads[device]=zip(gradients,variables)

# combine gradients from each device
# see: https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
          # Add 0 dimension to the gradients to represent the tower.
          expanded_g = tf.expand_dims(g, 0)

          # Append on a 'tower' dimension which we will average over below.
          grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads,0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

with tf.device('/cpu:0'):
    with tf.variable_scope('merge_losses'):
        averaged_grads = average_gradients(device_grads.values())
        train_op = optimizer.apply_gradients(
            averaged_grads,
            global_step=global_step
            )
        if args.no_optimizer:
            train_op=()

# combine summaries
with tf.variable_scope('summaries'):
    metrics={}
    for k in device_metrics[cuda_devices[0]].keys():
        metric=[]
        for device in cuda_devices:
            metric.append(device_metrics[device][k])
        try:
            w_total=1e-6
            metric_tmp=[]
            for (w,v) in metric:
                w_total+=w
            for (w,v) in metric:
                metric_tmp.append((w/w_total)*v)
            metric=metric_tmp
        except:
            metric=map(lambda x:x/num_devices,metric)
        metrics[k]=sum(metric)
    op_summaries=model.metrics2summaries(args,metrics)

# optimization nodes
#with tf.variable_scope('optimization'):
    #global_step = tf.Variable(0, name='global_step', trainable=False)
    #if args.decay is None:
        #learningrate = args.learningrate
    #else:
        #learningrate = tf.train.inverse_time_decay(
            #args.learningrate,
            #global_step,
            #args.decay,
            #1.0,
            #staircase=args.loss_staircase)
    #tf.summary.scalar('learningrate',learningrate)
    #if args.optimizer=='adam':
        #optimizer = tf.train.AdamOptimizer(learningrate)
    #elif args.optimizer=='sgd':
        #optimizer = tf.train.MomentumOptimizer(learningrate,args.momentum)
    ##train_op = optimizer.minimize(op_loss_regularized, global_step=global_step)
#
    #gradients, variables = zip(*optimizer.compute_gradients(
        #op_loss_regularized,
        #colocate_gradients_with_ops=True,
        #))
    #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    #train_op = optimizer.apply_gradients(
        #zip(gradients, variables),
        #global_step=global_step
        #)

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

# create loss output
if args.loss_output:
    loss_output=open(log_dir+'/loss_output.txt','w')

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

    def restore(suffix):
        var_dict={}
        for k in sorted(var_to_shape_map):
            ksplits=k.split('/')
            if ksplits[0][-2] == '_':
                continue
            try:
                k0=ksplits[0]+suffix
                k2='/'.join([k0]+ksplits[1:])
                var=tf.get_default_graph().get_tensor_by_name(k2+":0")
                if var.get_shape() == var_to_shape_map[k]:
                    var_dict[k]=var
                    print('  restoring',k)
                else:
                    print('  not restoring',k,'; old=',var_to_shape_map[k],'; new=',var.get_shape())
            except Exception as e:
                print('  variable not found in graph: ',k)
                #print('    e=',e)
        loader = tf.train.Saver(var_dict)
        loader.restore(sess, chkpt_file)

    restore('')
    #for i in range(1,len(cuda_devices)):
        #restore('_'+str(i))

    #print('vars=',tf.trainable_variables())
    #for var in tf.trainable_variables():
        #print('var=',var)

    #raise ValueError('poop')
    #print('  restored_vars=',var_dict)
    #loader = tf.train.Saver(var_dict)
    #loader.restore(sess, chkpt_file)
    #loader.restore(sess, args.initial_weights)
    #saver.restore(sess,args.initial_weights)
    print('  model restored from ',args.initial_weights)

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


# setup multiprocessing
import multiprocessing as mp
#if cuda_devices==['no_cuda']:
    #fmq=mp
#else:
import fmq
mp_queue = fmq.Queue(maxsize=20*len(cuda_devices))

def mk_batches(device_files,deviceid):

    files_remaining=copy.deepcopy(device_files)

    # function that returns the next batch
    # loop
    while True:
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
                    #try:
                        #open_files.append(open(filename,'rt'))
                    #except Exception as e:
                    print('open failed on ',filename,file=sys.stderr)
                    print(e,file=sys.stderr)
                    continue
                print('  opening [%s]; files remaining: %d/%d; buffer state: %d/%d'%(
                    filename,
                    len(files_remaining),
                    len(device_files),
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
                data=model.json2dict(args,nextline)
                batch.append(data)
            #except Exception as e:
                #pass
                #print('current file=',open_files[index].name)
                #if str(e)!="'text'":
                    #print('  exception:',e)

        feed_dict=model.mk_feed_dict(args,batch)
        #print('mp_queue.qsize()=',mp_queue.qsize())
        mp_queue.put(feed_dict)

cpu_per_gpu=1
for i in range(0,cpu_per_gpu*len(cuda_devices)):
    device_files=files_remaining[
        (cpu_per_gpu*i+0)*len(files_remaining)/len(cuda_devices):
        (cpu_per_gpu*i+1)*len(files_remaining)/len(cuda_devices)
        ]
    process=mp.Process(target=mk_batches,args=[device_files,i])
    process.start()

# finalize graph
#coord = tf.train.Coordinator()
#enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
sess.graph.finalize()

# loop through training data
firstLoop=True
while True:

    # load a feed_dict from the processes
    # on the first loop, these will get stored in the staging area
    decoding_time_start=time.time()
    if not args.repeat_batch or stats_step['count']==0:
        feed_dict={}
        for device in cuda_devices:
            queue_size=mp_queue.qsize()
            if queue_size<5:
                print('mp_queue.qsize()=',mp_queue.qsize())
            device_dict=mp_queue.get()
            for k,v in device_dict.iteritems():
                feed_dict['input'+device+'/'+k]=v
    decoding_time_stop=time.time()
    stats_step['decoding_time']+=decoding_time_stop-decoding_time_start
    stats_epoch['decoding_time']+=decoding_time_stop-decoding_time_start

    if firstLoop and not args.no_staging_area:
        sess.run(device_input_queue_enqueue,feed_dict=feed_dict)
        firstLoop=False
        continue

    # update loop counters
    # Note: this must be after the firstLoop check above due to the continue
    stats_step['count']+=1
    stats_epoch['steps']+=1

    # Write the summaries and print an overview fairly often.
    record_summary=(
        (stats_step['count'] < 10) or
        (stats_step['count'] < 1000 and stats_step['count']%10 == 0) or
        (stats_step['count'] < 10000 and stats_step['count']%100 == 0) or
        (stats_step['count']%1000 == 0))

    # run the model
    run_time_start=time.time()
    if record_summary:
        #print('feed_dict.keys()=',feed_dict.keys())
        _a,_b,_c,summary_str,_metrics = sess.run(
            [ train_op , op_summaries, device_input_queue_enqueue, summary, metrics ]
        #print('device_inputs=',device_inputs[device_inputs.keys()[0]].keys())
        #print('feed_dict=',feed_dict.keys())
        #_a,_b,_c,_metrics,[_run_country,_run_country_out]=sess.run(
            #[ train_op , op_summaries, device_input_queue_enqueue, summary, metrics, [device_inputs[device_inputs.keys()[0]]['country_'],feed_dict['input1/country_:0']] ]
            , feed_dict=feed_dict
            )
        summary_writer.add_summary(summary_str, stats_step['count']*args.batchsize*num_devices)
        summary_writer.flush()

    else:
        _a,_b,_c,_metrics=sess.run(
            [ train_op , op_summaries, device_input_queue_enqueue, metrics ]
            , feed_dict=feed_dict
            )

    #print('countries=',_run_country,_run_country_out)

    if args.loss_output:
        #loss_output.write('%0.4f %0.4f %0.4f\n'%(
        #print('_metrics=',_metrics.keys())
        loss_output.write('%0.4f %0.4f %0.4f %0.4f\n'%(
            _metrics['all/dist'],
            _metrics['all/country_acc'][0],
            _metrics['all/country_acc'][1],
            _metrics['all/k100'][0],
            #_metrics['all/k100'][1],
            ))
        loss_output.flush()
        sess.run(reset_local_vars)
        #asd

    run_time_stop=time.time()
    stats_step['run_time']+=run_time_stop-run_time_start

    # Write the summaries and print an overview fairly often.
    if record_summary:
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
        #if not args.no_checkpoint:
            #summary_str = sess.run(summary) #, feed_dict=feed_dict)
            #summary_writer.add_summary(summary_str, stats_step['count']*args.batchsize*num_devices)
            #summary_writer.flush()

        # save model if not debugging
        if (stats_step['count'] in [1,100,1000] or stats_step['count'] % args.stepsave == 0) and not args.no_checkpoint:
            checkpoint_file = os.path.join(log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=stats_step['count'])

        # reset step variables and update epoch counters
        reset_stats_step()
        sess.run(reset_local_vars)

    # process epoch
    # FIXME: should this be removed?
    if stats_epoch['new']:
        err=''

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

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
parser.add_argument('--step_save',type=int,default=10000)
parser.add_argument('--seed',type=int,default=0)

parser.add_argument('--inputs',choices=['middles_all','middles_last','model'],default=['middles_last'],nargs='*')
parser.add_argument('--outputs',choices=['gps','gps2','gps2b','country'],default=['gps','country'],nargs='+')

parser.add_argument('--cyclelength',type=int,default=79)
parser.add_argument('--shufflemul',type=int,default=20)

parser.add_argument('--initial_weights',type=str,default=None)
parser.add_argument('--reset_global_step',action='store_true')

# hyperparams
parser.add_argument('--optimizer',choices=['Adam','RMSProp'],default='Adam')
parser.add_argument('--batchsize',type=int,default=64)
parser.add_argument('--learningrate',type=float,default=1e-4)
parser.add_argument('--l2',type=float,default=1e-5)
parser.add_argument('--l2gps',type=float,default=1e-5)
parser.add_argument('--model',type=str,
    choices=[
        'ResNet50',
        'ResNet152v2',
        'ResNet200v2',
        'ResNeXt50c32',
        'ResNeXt101c32',
        'ResNeXt101c64',
        'WideResNet50',
        'DenseNet201',
        'Inception3',
        'Inception4',
        'InceptionResNet2',
        'VGG16',
        'PNASNetlarge',
        ]
    )

parser.add_argument('--pretrain',action='store_true')
parser.add_argument('--train_only_last_until_step',type=float,default=0.0)
parser.add_argument('--gmm_type',choices=['simple','verysimple'],default='simple')
parser.add_argument('--gmm_minimizedist',action='store_true')
parser.add_argument('--gmm_splitter',action='store_true')

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'
parser.add_argument('--trainable_mu',type=boolean_string,default=False)
parser.add_argument('--trainable_kappa',type=boolean_string,default=False)
parser.add_argument('--trainable_weights',type=boolean_string,default=True)
parser.add_argument('--lores_gmm_prekappa0',type=float,default=10.0)
parser.add_argument('--hires_gmm_prekappa0',type=float,default=12.0)

args = parser.parse_args()

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
print('loading libraries')

import tensorflow as tf
tf.set_random_seed(args.seed)

import tensornets as nets
import datetime

global_step=tf.train.create_global_step()

########################################
print('creating model pipeline')

import itertools
hex=['1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f']
perms=list(map(''.join,itertools.product(hex,repeat=3)))
files=['/rhome/mizbicki/bigdata/geolocating/data/flickr/img_train/'+perm+'/*.jpg' for perm in perms]

import model
iter=model.mkDataset(args,files,is_training=True)
file_,(gps_,country_),image_=iter.get_next()
net,loss,loss_regularized,op_metrics=model.mkModel(args,image_,country_,gps_,is_training=True)

########################################
print('creating optimizer')

if args.optimizer=='Adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=args.learningrate)
elif args.optimizer=='RMSProp':
    optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learningrate)

# create train_op for all layers
trainable_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
train_op_alllayers = optimizer.minimize(
    loss_regularized,
    global_step=global_step,
    #colocate_gradients_with_ops=True
    )

# create train_op for last layers
trainable_vars=[]
print('  last_layer variables:')
for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    if 'gps_loss' in v.name or 'loss_country' in v.name:
        print('    ++ ',v.name)
        trainable_vars.append(v)
train_op_lastlayers = optimizer.minimize(
    loss_regularized,
    global_step=global_step,
    var_list=trainable_vars,
    #colocate_gradients_with_ops=True
    )

########################################
print('creating session')

config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config=config)

# create log dir
if args.log_name is None:
    args.log_name='inputs=%s,outputs=%s,model=%s,batchsize=%d,learningrate=%1.1E,l2=%1.1E,l2gps=%1.1E,%s'%(
        '_'.join(args.inputs),
        '_'.join(args.outputs),
        args.model,
        args.batchsize,
        args.learningrate,
        args.l2,
        args.l2gps,
        ',pretrain' if args.pretrain else '',
        #',train_only_last' if args.train_only_last else ''
        )
log_dir=os.path.join(args.log_dir,args.log_name)
tf.gfile.MakeDirs(log_dir)
print('  log_dir=',log_dir)

# save args
import simplejson as json
args_str=json.dumps(vars(args))
with open(log_dir+'/args.json','w') as f:
    f.write(args_str)

# create summaries
# FIXME: should be abstracted between models
op_summaries={}
with tf.variable_scope('streaming_mean'):
    for k,v in op_metrics.iteritems():
        try:
            (weights,metric)=v
        except:
            metric=v
        op_summaries[k]=tf.contrib.metrics.streaming_mean(metric,name=k)
for k,(v,_) in op_summaries.iteritems():
    tf.summary.scalar(k,v)
summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)

saver_step = tf.train.Saver(max_to_keep=2)
saver_epoch = tf.train.Saver(max_to_keep=50)

# initialization
print('  initializing vars')
reset_global_vars=tf.global_variables_initializer()
reset_local_vars=tf.local_variables_initializer()
sess.run([reset_global_vars,reset_local_vars])

if args.pretrain:
    print('  loading pretrained weights')
    sess.run(net.pretrained())

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
    print('  model restored from ',args.initial_weights)

if args.reset_global_step:
    print('  reset global step')
    sess.run(tf.assign(global_step,0))

########################################
print('training loop')
sess.graph.finalize()
sess.run(iter.initializer)
local_epoch=0
local_step=0
while True:
    try:

        res_step=sess.run(global_step)
        local_step+=1

        # perform one training step
        if res_step<=args.train_only_last_until_step:
            train_op=train_op_lastlayers
            train_op_msg='last layers'
        else:
            train_op=train_op_alllayers
            train_op_msg='all layers'

        #res_loss,_=sess.run([loss_regularized,[train_op,op_summaries]])
        res_loss,_=sess.run([
            loss_regularized,
            [ train_op,
              op_summaries,
              tf.get_collection('gps_loss_updates'),
              #[tf.get_collection('gps_loss_splitter')],
              [tf.get_collection('gps_loss_splitter')] if local_step%100==0 and local_step>=1000 and args.gmm_splitter else [],
            ]
            ])

        # record summaries occasionally
        if (
           ( local_step<=10 ) or
           ( local_step%10==0 and local_step<1000 ) or
           ( local_step%100==0 and local_step<10000 ) or
           ( local_step%1000==0 )
           ):

            summary_str=sess.run(summary)
            summary_writer.add_summary(summary_str, res_step*args.batchsize)
            summary_writer.flush()
            sess.run(reset_local_vars)

            print('%s  step=%d (%d) loss=%g  %s' %
                ( datetime.datetime.now()
                , res_step
                , local_step
                , res_loss
                , train_op_msg
                ))

        # save model occasionally
        if (local_step in [1,100,1000] or res_step%args.step_save == 0):
            checkpoint_file = os.path.join(log_dir, 'step.ckpt')
            saver_step.save(sess, checkpoint_file, global_step=res_step)

    except tf.errors.OutOfRangeError:
        print('  epoch',local_epoch,'done!')
        local_epoch+=1
        checkpoint_file = os.path.join(log_dir, 'epoch.ckpt')
        saver_epoch.save(sess, checkpoint_file, global_step=local_epoch)
        sess.run(iter.initializer)

    except Exception as e:
        sys.stderr.write('Exception: '+str(e)+'\n')

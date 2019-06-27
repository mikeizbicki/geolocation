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
parser.add_argument('--features_file',type=str,default=None)
parser.add_argument('--jpg_dir',type=str,default='/rhome/mizbicki/bigdata/geolocating/data/flickr/img_train/')

parser.add_argument('--inputs',choices=['middles_all','middles_last','model'],default=['middles_last'],nargs='*')
parser.add_argument('--outputs',choices=['gps','gps2','gps2b','s2','country'],default=['gps','country'],nargs='+')
parser.add_argument('--s2size',type=int,default=12)
parser.add_argument('--input_format',type=str,choices=['jpg','tfrecord'],default='jpg')
parser.add_argument('--image_size',type=int,default=224)


parser.add_argument('--initial_weights',type=str,default=None)
parser.add_argument('--reset_global_step',action='store_true')
parser.add_argument('--cyclelength',type=int,default=20)

# hyperparams
parser.add_argument('--optimizer',choices=['Adam','RMSProp'],default='Adam')
parser.add_argument('--batchsize',type=int,default=64)
parser.add_argument('--shufflemul',type=int,default=20)
parser.add_argument('--learningrate',type=float,default=1e-4)
parser.add_argument('--lrb',type=float,default=10.0)
parser.add_argument('--lr_boundaries',nargs='*',type=float,default=[])
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
parser.add_argument('--train_em',action='store_true')
parser.add_argument('--gmm_type',choices=['simple','verysimple'],default='simple')
parser.add_argument('--gmm_distribution',choices=['fvm','fvm2'],default='fvm2')
parser.add_argument('--gmm_minimizedist',action='store_true')
parser.add_argument('--gmm_xentropy',action='store_true')
parser.add_argument('--gmm_no_logloss',action='store_true')
parser.add_argument('--gmm_splitter',action='store_true')
parser.add_argument('--gmm_components',type=int,default=8192)
parser.add_argument('--gmm_gradient_method',choices=['all','stop','efam','main'])
parser.add_argument('--s2warmstart_mu',action='store_true')
parser.add_argument('--s2warmstart_kappa',action='store_true')
parser.add_argument('--s2warmstart_kappa_s',type=float,default=0.95)
parser.add_argument('--mu_max_meters',type=float,default=0.1)

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
    args.log_dir='log'
    args.log_name=None
    args=parser.parse_args(namespace=args)

args.num_crops=1
print('args=',args)

args.reset_global_step=False

########################################
print('loading libraries')

import tensorflow as tf
tf.set_random_seed(args.seed)

import tensornets as nets
import datetime

global_step=tf.train.create_global_step()

########################################
print('creating model pipeline')

import model
is_training=args.features_file is None

if args.input_format=='jpg':
    import itertools
    hex=['1','2','3','4','5','6','7','8','9','0','a','b','c','d','e','f']
    perms=list(map(''.join,itertools.product(hex,repeat=3)))
    files=[args.jpg_dir+'/'+perm+'/*.jpg' for perm in perms]
    #files=[args.jpg_dir+'/'+'*.jpg']

    #print('files=',files)
    #asd

    iter=model.mkDataset_jpg(args,files,is_training=is_training)
    file_,(gps_,country_),image_=iter.get_next()
    if is_training:
        image_=tf.reshape(image_,[-1,1,args.image_size,args.image_size,3])
    else:
        image_=tf.reshape(image_,[-1,10,args.image_size,args.image_size,3])
        image_=image_[:,4,:,:,:]
        image_=tf.reshape(image_,[-1,1,args.image_size,args.image_size,3])
    #image_=tf.tile(image_,[1,10,1,1,1])
    features_=None
    s2_=None

elif args.input_format=='tfrecord':
    #files='tmp.tfrecord'
    files=['/rhome/mizbicki/bigdata/geolocating/data/image_tfrecord/out_tfrecord_32_'+str(i) for i in range(0,32)]
    #files='qqq'
    iter=model.mkDataset_tfrecord_features(args,files)
    gps_,country_,features_,s2_=iter.get_next()
    image_=None

net,features_,loss,loss_regularized,op_metrics,op_endpoints=model.mkModel(
    args,
    image_,
    country_,
    gps_,
    is_training=is_training,
    features=features_,
    s2_=s2_
    )

########################################
print('creating optimizer')

# set learning rate
if len(args.lr_boundaries) > 0:
    lr_values=[ args.learningrate/(args.lrb**i) for i in range(0,len(args.lr_boundaries)+1) ]
    lr_boundaries=map(int,args.lr_boundaries)
    print('lr_values=',lr_values)
    learningrate=tf.train.piecewise_constant(
        global_step,
        lr_boundaries,
        lr_values,
        )
else:
    learningrate=args.learningrate

op_metrics['learningrate']=learningrate

# create optimizer
if args.optimizer=='Adam':
    optimizer = tf.train.AdamOptimizer(learning_rate=learningrate)
elif args.optimizer=='RMSProp':
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learningrate)

# create train_op for all layers
trainable_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#train_op_alllayers = optimizer.minimize(
    #loss_regularized,
    #global_step=global_step,
    #)

gvs = optimizer.compute_gradients(loss_regularized)
gvs_clipped = []
for grad,var in gvs:
    if 'pre_mu' not in var.name:
        gvs_clipped.append((grad,var))
    else:
        max_meters=args.mu_max_meters
        max_grad=max_meters/6371000.0
        grad_clipped=tf.minimum(max_grad,tf.maximum(-max_grad,grad))
        print('var=',var,'grad=',grad)
        gvs_clipped.append((grad_clipped,var))
        op_metrics['gps_loss_grad/pre_mu_grad_max']=tf.reduce_max(tf.abs(grad))
        op_metrics['gps_loss_grad/pre_mu_grad_min']=tf.reduce_min(tf.abs(grad))
        op_metrics['gps_loss_grad/pre_mu_grad_mean']=tf.reduce_mean(tf.abs(grad))
        op_metrics['gps_loss_grad/pre_mu_grad_clipped_max']=tf.reduce_max(tf.abs(grad_clipped))
        op_metrics['gps_loss_grad/pre_mu_grad_clipped_min']=tf.reduce_min(tf.abs(grad_clipped))
        op_metrics['gps_loss_grad/pre_mu_grad_clipped_mean']=tf.reduce_mean(tf.abs(grad_clipped))
train_op_alllayers = optimizer.apply_gradients(gvs_clipped,global_step=global_step)

# create train_op for last layers
trainable_vars=[]
print('  last_layer variables:')
for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    if 'gps_loss' in v.name or 'loss_country' in v.name or 'loss_s2' in v.name:
        print('    ++ ',v.name)
        trainable_vars.append(v)
train_op_lastlayers = optimizer.minimize(
    loss_regularized,
    global_step=global_step,
    var_list=trainable_vars,
    )

# create train_op for expectation maximization
#print('  EM variables:')
#trainable_vars_e=[]
#trainable_vars_m=[]
#for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
    #if 'mu' in v.name or 'kappa' in v.name:
        #print('    e: ',v.name)
        #trainable_vars_e.append(v)
    #else:
        #print('    m: ',v.name)
        #trainable_vars_m.append(v)
#train_op_e = optimizer.minimize(
    #loss_regularized,
    #global_step=global_step,
    #var_list=trainable_vars_e,
    #)
#train_op_m = optimizer.minimize(
    #loss_regularized,
    #global_step=global_step,
    #var_list=trainable_vars_e,
    #)

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
    if not net is None:
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
                if var.get_shape() != var_to_shape_map[k]:
                    print('  >> not restoring',k,'; old=',var_to_shape_map[k],'; new=',var.get_shape())
                elif '_constant' in k:
                    print('  >> not restoring',k)

                else:
                    var_dict[k]=var
                    print('  restoring',k)
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

        # extract features
        if args.features_file:
            #print('local_step=',local_step)
            print('%s  step=%d (%d)' %
                ( datetime.datetime.now()
                , res_step
                , local_step
                ))
            try:
                writer
            except:
                writer=tf.python_io.TFRecordWriter(args.features_file)

            file,gps,country,features=sess.run([file_,gps_,country_,features_])
            for i in range(0,args.batchsize):
                feature = {
                    'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=file[i])),
                    'train/gps': tf.train.Feature(float_list=tf.train.FloatList(value=gps[i,:])),
                    'train/country': tf.train.Feature(int64_list=tf.train.Int64List(value=[country[i]])),
                    'train/features': tf.train.Feature(float_list=tf.train.FloatList(value=features[i,:])),
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
            continue

        # select training op
        if args.train_em:
            if (res_step/1000)%2==0:
                train_op=train_op_m
                train_op_msg='m'
            else:
                train_op=train_op_e
                train_op_msg='e'
        elif res_step<=args.train_only_last_until_step:
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
              tf.get_collection(tf.GraphKeys.UPDATE_OPS),
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

        if args.features_file:
            sys.exit(0)

    except Exception as e:
        sys.stderr.write('Exception: '+str(e)+'\n')

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

parser.add_argument('--initial_weights',type=str,required=True)
parser.add_argument('--outputdir',type=str,default='log/img')
parser.add_argument('--images',type=str,default='data/im2gps')
parser.add_argument('--images_subdir',action='store_true')
parser.add_argument('--xaxis',type=str,default='local_step*args.batchsize')
parser.add_argument('--batchsize',type=int,default=400)
parser.add_argument('--num_crops',type=int,choices=[1,10],default=1)
parser.add_argument('--input_format',type=str,choices=['jpg','tfrecord'],default='jpg')

parser.add_argument('--lores_gmm_prekappa0',type=float,default=10.0)
parser.add_argument('--hires_gmm_prekappa0',type=float,default=12.0)
parser.add_argument('--gmm_xentropy',action='store_true')

args = parser.parse_args()

# load arguments from file if applicable
print('  loading arguments from %s/args.json'%args.initial_weights)
import simplejson as json
with open(args.initial_weights+'/args.json','r') as f:
    args_str=f.readline()

args=argparse.Namespace(**json.loads(args_str))
args=parser.parse_args(namespace=args)
args.gmm_log2=True

print('args=',args)

args.gmm_minimizedist=False
args.gmm_gradient_method=False

########################################
print('loading libraries')

import tensorflow as tf
tf.set_random_seed(args.seed)

import tensornets as nets
import datetime

########################################
print('creating model pipeline')

import model
if args.input_format=='jpg':
    if args.images_subdir:
        files=[args.images+'/a*/*.jpg']
    else:
        files=[args.images+'/*.jpg']
    iter=model.mkDataset_jpg(args,files,is_training=False)
    file_,(gps_,country_),image_=iter.get_next()
    image_=tf.reshape(image_,[-1,10,args.image_size,args.image_size,3])
    if args.num_crops==1:
        image_=image_[:,4,:,:,:]
        image_=tf.reshape(image_,[-1,1,args.image_size,args.image_size,3])
    print('image_=',image_)
    features_=None
    s2_=None

elif args.input_format=='tfrecord':
    #files=['/rhome/mizbicki/bigdata/geolocating/data/im2gps_tfrecord/out_tfrecord_1_0']
    #files=['/rhome/mizbicki/bigdata/geolocating/data/im2gps_tfrecord/out_tfrecord_1_0']
    files=['/rhome/mizbicki/bigdata/geolocating/data/image_tfrecord/out_tfrecord_32_'+str(i) for i in [0]]
    #files=['/rhome/mizbicki/bigdata/geolocating/data/image_tfrecord_test/out_tfrecord_8_'+str(i) for i in range(0,8)]
    iter=model.mkDataset_tfrecord_infer(args,files,is_training=False)
    #iter=model.mkDataset_tfrecord_features(args,files,is_training=True)
    gps_,country_,features_,s2_=iter.get_next()
    image_=None

net,features,loss,loss_regularized,op_metrics,op_endpoints=model.mkModel(
    args,
    image_,
    country_,
    gps_,
    is_training=False,
    #is_training=True,
    gmm_log2=True,
    features=features_,
    s2_=s2_
    )

#tf.summary.image('input_image',image_)

########################################
print('creating session')

config = tf.ConfigProto(allow_soft_placement = True)
sess = tf.Session(config=config)

# make log dir
log_dir=args.outputdir #'log/infer'
tf.gfile.MakeDirs(log_dir)
print('  log_dir=',log_dir)

# save args
import simplejson as json
args_str=json.dumps(vars(args))
with open(log_dir+'/args.json','w') as f:
    f.write(args_str)

# create summaries
# FIXME: should be abstracted between models
mean_updates={}
with tf.variable_scope('streaming_mean'):
    for k,v in op_metrics.iteritems():
        try:
            (weights,metric)=v
        except:
            metric=v
        mean_updates[k]=tf.contrib.metrics.streaming_mean(metric,name=k)
for k,(v,_) in mean_updates.iteritems():
    tf.summary.scalar(k,v)
summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)

# initialization
print('  initializing vars')
reset_global_vars=tf.global_variables_initializer()
reset_local_vars=tf.local_variables_initializer()
sess.run([reset_global_vars,reset_local_vars])

if args.initial_weights:
    # see https://stackoverflow.com/questions/41621071/restore-subset-of-variables-in-tensorflow
    print('restoring model')
    chkpt_file=tf.train.latest_checkpoint(args.initial_weights)
    #chkpt_file=args.initial_weights+'/epoch.ckpt-3'
    print('  chkpt_file=',chkpt_file)
    reader = tf.pywrap_tensorflow.NewCheckpointReader(chkpt_file)
    var_to_shape_map = reader.get_variable_to_shape_map()

    def restore(suffix):
        var_dict={}
        for k in sorted(var_to_shape_map):
            for j in range(0,args.num_crops):
                #knew=k
                knew=k.replace('wideresnet50','wideresnet50_'+str(j))
                ksplits=knew.split('/')

                #if ksplits[0][-2] == '_':
                    #continue
                try:
                    k0=ksplits[0]+suffix
                    k2='/'.join([k0]+ksplits[1:])
                    var=tf.get_default_graph().get_tensor_by_name(k2+":0")
                    if var.get_shape() == var_to_shape_map[k]:
                        var_dict[k]=var
                        print('  restoring',knew)
                    else:
                        print('  not restoring',k,'; old=',var_to_shape_map[k],'; new=',var.get_shape())
                except Exception as e:
                    if 'Adam' in k or 'beta1_power' in k or 'beta2_power' in k:
                        pass
                    else:
                        print('  variable not found in graph: ',k)
                        print('    e=',e)
        loader = tf.train.Saver(var_dict)
        loader.restore(sess, chkpt_file)

    restore('')
    print('  model restored from ',args.initial_weights)

########################################
print('eval loop')
sess.run(reset_local_vars)
sess.graph.finalize()
sess.run(iter.initializer)

f_dist=open(args.outputdir+'/dist','w')

local_step=0
while True:
    try:

        local_step+=1
        res_loss,endpoints,gps,_=sess.run([loss_regularized,op_endpoints,gps_,[mean_updates]])

        print('%s  step=%d  loss=%g' %
            ( datetime.datetime.now()
            , local_step
            , res_loss
            ))

        for i in range(0,endpoints['dist'].shape[0]):
            #f_dist.write('%f\n'%(op_endpoints['dist'][i]))
            dist=endpoints['dist'][i]
            lat=endpoints['gps'][i][0]
            lon=endpoints['gps'][i][1]
            lat_=gps[i][0]
            lon_=gps[i][1]
            f_dist.write('%f (%f,%f) (%f,%f)\n'%(dist,lat,lon,lat_,lon_))

        #print('endpoints.keys()=',endpoints.keys())

    except tf.errors.OutOfRangeError:
        summary_str=sess.run(summary)
        summary_writer.add_summary(summary_str, eval(args.xaxis)) #local_step*args.batchsize)
        summary_writer.flush()
        print('done!')
        break

    #except Exception as e:
        #sys.stderr.write('Exception: '+str(e)+'\n')


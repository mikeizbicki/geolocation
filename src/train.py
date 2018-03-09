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
parser.add_argument('--warmstart',type=bool,default=True)
parser.add_argument('--stepdelta',type=int,default=100)
parser.add_argument('--stepsave',type=int,default=10000)
parser.add_argument('--seed',type=int,default=0)
parser.add_argument('--max_open_files',type=int,default=48)

# debug variables
parser.add_argument('--no_checkpoint',action='store_true')
parser.add_argument('--tf_debug',action='store_true')
parser.add_argument('--repeat_batch',action='store_true')

# model hyperparameters
parser.add_argument('--data',type=str,required=True)
parser.add_argument('--data_summary',type=str,default=None)
parser.add_argument('--hashsize',type=int,default=20)
parser.add_argument('--batchsize',type=int,default=100)
parser.add_argument('--learningrate',type=float,default=0.005)

parser.add_argument('--input',choices=['cltcc','bow','lang','time','const'],nargs='+',required=True)
parser.add_argument('--full',type=int,nargs='*',default=[])
parser.add_argument('--output',choices=['naive','aglm','aglm2','proj3d','loc'],required=True)
parser.add_argument('--loss',choices=['l2','chord','dist','dist2','angular','xentropy'],required=True)

parser.add_argument('--dropout',type=float,default=0.5)
parser.add_argument('--l1',type=float,default=0.0)
parser.add_argument('--l2',type=float,default=1e-5)

parser.add_argument('--bow_layersize',type=int,default=2)
parser.add_argument('--bow_dense',action='store_true')
parser.add_argument('--cltcc_vocabsize',type=int,default=128)
parser.add_argument('--cltcc_numfilters',type=int,default=256)

parser.add_argument('--maxloc',default=10,type=int)
parser.add_argument('--filter_locations',action='store_true')

parser.add_argument('--calc_loc',action='store_true')
parser.add_argument('--calc_gps',action='store_true')

args = parser.parse_args()

# ensure valid argument combinations

if (args.output in ['naive','aglm','aglm2','proj3d'] or
    args.loss in ['l2','chord','dist','dist2','angular']
    ):
    args.calc_gps=True

if args.output in ['loc'] or args.filter_locations:
    args.filter_locations=True
    args.calc_loc=True
    if not args.data_summary:
        raise ValueError('--data_summary must be specified when calculating locations')

print('args=',args)

########################################
if args.data_summary:
    print('loading data summary pickle')
    import pickle
    from pprint import pprint
    from collections import defaultdict
    lambda0 = lambda: 0
    lambda1 = lambda: 1
    f=open(args.data_summary,'r')
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

import copy
from collections import defaultdict
import datetime
import gzip
import math
import os
import simplejson as json
import time
import unicodedata

#import geopy.distance
import numpy as np
import scipy as sp

import sklearn.feature_extraction.text
hv=sklearn.feature_extraction.text.HashingVectorizer(n_features=2**args.hashsize,norm=None)

import random
random.seed(args.seed)

########################################
print('initializing tensorflow')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.set_random_seed(args.seed)
var_init = lambda shape,var: tf.truncated_normal(shape,stddev=var,seed=args.seed)

# tf inputs
with tf.name_scope('inputs'):

    regularizers=[]
    inputs=[]

    # true labels
    loc_ = tf.placeholder(tf.int32, [args.batchsize,1])
    gps_ = tf.placeholder(tf.float32, [args.batchsize,2])
    op_lat_ = gps_[:,0]
    op_lon_ = gps_[:,1]
    op_lat_rad_ = op_lat_/360*2*math.pi
    op_lon_rad_ = op_lon_/360*2*math.pi

    # hash bow inputs
    if 'bow' in args.input:
        with tf.name_scope('bow'):
            input_size=2**args.hashsize
            if args.bow_dense:
                hash_ = tf.placeholder(tf.float32,[args.batchsize,input_size])
                matmul = tf.matmul
                hash_reg=args.l1*tf.reduce_sum(tf.abs(hash_))
            else:
                hash_ = tf.sparse_placeholder(tf.float32,name='hash_')
                matmul = tf.sparse_tensor_dense_matmul
                hash_reg=args.l1*tf.sparse_reduce_sum(tf.abs(hash_))
            regularizers.append(hash_reg)
            w = tf.Variable(var_init([input_size,args.bow_layersize],1.0))
            b = tf.constant(0.1,shape=[args.bow_layersize])
            inputs.append(matmul(hash_,w)+b)

    # cnn inputs
    # follows paper "character level convnets for text classification"
    # see also: Language-Independent Twitter Classification Using Character-Based Convolutional Networks
    tweetlen=280
    activation=tf.nn.relu
    if 'cltcc' in args.input:
        with tf.name_scope('cltcc'):
            text_ = tf.placeholder(tf.float32, [args.batchsize,tweetlen,args.cltcc_vocabsize])
            text_reshaped = tf.reshape(text_,[args.batchsize,tweetlen,args.cltcc_vocabsize,1])

            filterlen=7
            with tf.name_scope('conv1'):
                w = tf.Variable(var_init([filterlen,args.cltcc_vocabsize,1,args.cltcc_numfilters],0.05))
                b = tf.Variable(tf.constant(0.1,shape=[args.cltcc_numfilters]))
                conv = tf.nn.conv2d(text_reshaped, w, strides=[1,1,1,1], padding='VALID')
                h = activation(tf.nn.bias_add(conv,b))
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 3, 1, 1],
                    strides=[1, 3, 1, 1],
                    padding='VALID')

            with tf.name_scope('conv2'):
                w = tf.Variable(var_init([filterlen,1,args.cltcc_numfilters,args.cltcc_numfilters],0.05))
                b = tf.Variable(tf.constant(0.1,shape=[args.cltcc_numfilters]))
                conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
                h = activation(tf.nn.bias_add(conv,b))
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 3, 1, 1],
                    strides=[1, 3, 1, 1],
                    padding='VALID')

            filterlen=3
            with tf.name_scope('conv3'):
                w = tf.Variable(var_init([filterlen,1,args.cltcc_numfilters,args.cltcc_numfilters],0.05))
                b = tf.Variable(tf.constant(0.1,shape=[args.cltcc_numfilters]))
                conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
                h = activation(tf.nn.bias_add(conv,b))
                pooled = h

            with tf.name_scope('conv4'):
                w = tf.Variable(var_init([filterlen,1,args.cltcc_numfilters,args.cltcc_numfilters],0.05))
                b = tf.Variable(tf.constant(0.1,shape=[args.cltcc_numfilters]))
                conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
                h = activation(tf.nn.bias_add(conv,b))
                pooled = h

            with tf.name_scope('conv5'):
                w = tf.Variable(var_init([filterlen,1,args.cltcc_numfilters,args.cltcc_numfilters],0.05))
                b = tf.Variable(tf.constant(0.1,shape=[args.cltcc_numfilters]))
                conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
                h = activation(tf.nn.bias_add(conv,b))
                pooled = h

            with tf.name_scope('conv6'):
                w = tf.Variable(var_init([filterlen,1,args.cltcc_numfilters,args.cltcc_numfilters],0.05))
                b = tf.Variable(tf.constant(0.1,shape=[args.cltcc_numfilters]))
                conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
                h = activation(tf.nn.bias_add(conv,b))
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, 3, 1, 1],
                    strides=[1, 3, 1, 1],
                    padding='VALID')

            last=pooled
            input_size=int(last.get_shape()[1]*last.get_shape()[2]*last.get_shape()[3])
            input_=tf.reshape(last,[args.batchsize,input_size])
            inputs.append(input_)

    # language inputs
    if 'lang' in args.input:
        with tf.name_scope('lang'):
            langs_emperical=[u'am',u'ar',u'bg',u'bn',u'bo',u'ckb',u'cs',u'cy',u'da',u'de',u'dv',u'el',u'en',u'es',u'et',u'eu',u'fa',u'fi',u'fr',u'gu',u'hi',u'ht',u'hu',u'hy',u'in',u'is',u'it',u'iw',u'ja',u'ka',u'km',u'kn',u'ko',u'lo',u'lt',u'lv',u'ml',u'mr',u'my',u'ne',u'nl',u'no',u'or',u'pa',u'pl',u'ps',u'pt',u'ro',u'ru',u'sd',u'si',u'sl',u'sr',u'sv',u'ta',u'te',u'th',u'tl',u'tr',u'uk',u'und',u'ur',u'vi',u'zh']
            langs_iso_639_1=['ab','aa','af','ak','sq','am','ar','an','hy','as','av','ae','ay','az','bm','ba','eu','be','bn','bh','bi','nb','bs','br','bg','my','es','ca','km','ch','ce','ny','ny','zh','za','cu','cu','cv','kw','co','cr','hr','cs','da','dv','dv','nl','dz','en','eo','et','ee','fo','fj','fi','nl','fr','ff','gd','gl','lg','ka','de','ki','el','kl','gn','gu','ht','ht','ha','he','hz','hi','ho','hu','is','io','ig','id','ia','ie','iu','ik','ga','it','ja','jv','kl','kn','kr','ks','kk','ki','rw','ky','kv','kg','ko','kj','ku','kj','ky','lo','la','lv','lb','li','li','li','ln','lt','lu','lb','mk','mg','ms','ml','dv','mt','gv','mi','mr','mh','ro','ro','mn','na','nv','nv','nd','nr','ng','ne','nd','se','no','nb','nn','ii','ny','nn','ie','oc','oj','cu','cu','cu','or','om','os','os','pi','pa','ps','fa','pl','pt','pa','ps','qu','ro','rm','rn','ru','sm','sg','sa','sc','gd','sr','sn','ii','sd','si','si','sk','sl','so','st','nr','es','su','sw','ss','sv','tl','ty','tg','ta','tt','te','th','bo','ti','to','ts','tn','tr','tk','tw','ug','uk','ur','ug','uz','ca','ve','vi','vo','wa','cy','fy','wo','xh','yi','yo','za','zu']
            langs=['unknown']+langs_emperical+langs_iso_639_1
            def hash_lang(lang):
                try:
                    return langs.index(lang)
                except:
                    print('unk=',lang)
                    return 0
            lang_ = tf.placeholder(tf.int32, [args.batchsize,1])
            lang_one_hot = tf.one_hot(lang_,len(langs),axis=1)
            inputs.append(lang_one_hot)

    # time inputs
    if 'time' in args.input:
        with tf.name_scope('time'):

            timestamp_ms_ = tf.placeholder(tf.float32, [args.batchsize,1])

            def wrapped(var,length):
                scaled=var/length*2*math.pi
                return [tf.sin(scaled),tf.cos(scaled)]

            time1 = wrapped(timestamp_ms_,1000*60*60*24*7)
            time2 = wrapped(timestamp_ms_,1000*60*60*24)
            time3 = wrapped(timestamp_ms_,1000*60*60*8)

            inputs.append(tf.stack(time1+time2+time3,axis=1))

    # constant input, for debugging purposes
    if 'const' in args.input:
        with tf.name_scope('const'):
            const=tf.reshape(tf.tile(tf.constant([1.0]),[args.batchsize]),[args.batchsize,1])
            inputs.append(const)

# fully connected hidden layers
with tf.name_scope('full'):
    layerindex=0
    final_layer=tf.concat(map(tf.contrib.layers.flatten,inputs),axis=1)
    final_layer_size=int(final_layer.get_shape()[1])

    for layersize in args.full:
        with tf.name_scope('full%d'%layerindex):
            w = tf.Variable(var_init([final_layer_size, layersize],1.0/math.sqrt(float(layersize))))
            b = tf.constant(0.1,shape=[layersize])
            h = tf.nn.relu(tf.matmul(final_layer,w)+b)
            final_layer=tf.nn.dropout(h,args.dropout)
            final_layer_size=layersize
        layerindex+=1

# rf outputs
with tf.name_scope('output'):

    # loc buckets
    if args.output=='loc':
        w = tf.Variable(tf.zeros([final_layer_size, len(lochash)]))
        b = tf.Variable(tf.zeros([len(lochash)]))
        loc = tf.matmul(final_layer,w)+b
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
        if args.warmstart:
            b = tf.Variable([34.052235,-118.243683])
        else:
            b = tf.Variable(tf.zeros([2]))
        gps = tf.matmul(final_layer,w) + b
        op_lat = gps[:,0]
        op_lon = gps[:,1]

    # angular generalized linear model
    # See: "Regression Models for Angular Response" by Fisher and Lee
    if args.output=='aglm':
        w = tf.Variable(tf.zeros([final_layer_size, 2]))
        if args.warmstart:
            b = tf.Variable([0.6745,-2])
        else:
            b = tf.Variable(tf.zeros([2]))
        response = tf.matmul(final_layer,w) + b
        op_lat = tf.atan(response[:,0])*360/2/math.pi
        op_lon = tf.atan(response[:,1])*360/math.pi
        gps = tf.stack([op_lat,op_lon],1)

    # same as algm, but uses the bias outside of the atan embedding
    if args.output=='aglm2':
        w = tf.Variable(tf.zeros([final_layer_size, 2]))
        if args.warmstart:
            b0 = tf.Variable([34.052235])
            b1 = tf.Variable([-118.243683])
        else:
            b0 = tf.Variable(tf.zeros([1]))
            b1 = tf.Variable(tf.zeros([1]))
        response = tf.matmul(final_layer,w)
        op_lat = tf.atan(response[:,0])*360/math.pi/2 + b0
        op_lon = tf.atan(response[:,1])*360/math.pi   + b1
        gps = tf.stack([op_lat,op_lon],1)

    # model gps coordinates in R^3
    if args.output=='proj3d':
        w = tf.Variable(tf.zeros([final_layer_size, 3]))
        b = tf.Variable([0.1,0,0])
        r3 = tf.matmul(final_layer,w) + b
        norm = tf.sqrt(tf.reduce_sum(r3*r3,1))
        r3normed = r3/tf.stack([norm,norm,norm],1)
        #op_lon = tf.asin(tf.minimum(1.0,r3normed[:,2]))
        #op_lat = tf.asin(tf.minimum(1.0,r3normed[:,1]))*tf.acos(tf.minimum(1.0,r3normed[:,2]))
        op_lon = tf.asin(r3normed[:,2])
        op_lat = tf.asin(r3normed[:,1])*tf.acos(r3normed[:,2])
        gps = tf.stack([op_lat,op_lon],1)
        #op_lon=tf.Print(op_lon,[gps,b,norm])

    # fancy gps coords
    #if args.output=='atlas':
        #numatlas=10
        #w = tf.Variable(tf.zeros([final_layer_size,2]))
        #b = tf.Variable([1.0,1.0])
        #x = tf.matmul(final_layer,w)+b
        #xnorm_squared = tf.reduce_sum(x*x)

        #pad=tf.zeros([args.batchsize,1])
        #print(x.get_shape())
        #print(pad.get_shape())
        #print(tf.stack([x,pad]).get_shape())

        #x3 = tf.stack([x,tf.zeros([1])])
        #p3 = tf.Variable([1.0,0,0])
        #q3 = p*(xnorm_squared-1)/(xnorm_squared+1)+x3*2/(xnorm_squared+1)

        #op_lat = tf.asin(q3[2])         *360/(2*math.pi)
        #op_lon = tf.atan(q3[1]/q3[0])   *360/(2*math.pi)
        #gps = tf.stack([op_lat,op_lon],1)
#
        #pass

    # common outputs
    epsilon = 1e-6

    op_lat_rad = op_lat/360*2*math.pi
    op_lon_rad = op_lon/360*2*math.pi

    hav = lambda x: tf.sin(x/2)**2
    squared_angular_dist = ( hav(op_lat_rad-op_lat_rad_)
                    +tf.cos(op_lat_rad)*tf.cos(op_lat_rad_)*hav(op_lon_rad-op_lon_rad_)
                   )

    # radius of earth = 3959 miles, 6371 kilometers
    op_dist = 2*6371*tf.asin(tf.sqrt(epsilon + squared_angular_dist))
    op_dist_ave = tf.reduce_sum(op_dist)/args.batchsize

    op_delta_x = tf.cos(op_lat_rad)*tf.cos(op_lon_rad)-tf.cos(op_lat_rad_)*tf.cos(op_lon_rad_)
    op_delta_y = tf.cos(op_lat_rad)*tf.sin(op_lon_rad)-tf.cos(op_lat_rad_)*tf.sin(op_lon_rad_)
    op_delta_z = tf.sin(op_lat_rad) - tf.sin(op_lat_rad_)
    op_chord = tf.sqrt(epsilon + op_delta_x**2 + op_delta_y**2 + op_delta_z**2)
    #op_dist = 2*6371*tf.asin(op_chord/2)
    #op_dist_ave = tf.reduce_sum(op_dist)/args.batchsize

    threshold_dist=100
    op_err = tf.sign(op_dist-threshold_dist)/2+0.5
    op_err_ave = tf.reduce_sum(op_err)/args.batchsize

# set loss function
if args.loss=='l2':
    op_loss = tf.reduce_sum((gps - gps_) * (gps - gps_))
if args.loss=='chord':
    op_loss = tf.reduce_sum(op_chord)/args.batchsize
if args.loss=='dist':
    op_loss = op_dist_ave
if args.loss=='dist2':
    op_loss = tf.reduce_sum(op_dist*op_dist)/args.batchsize
if args.loss=='angular':
    op_loss = tf.reduce_sum(squared_angular_dist)
if args.loss=='xentropy':
    op_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.to_int64(tf.reshape(loc_,[args.batchsize])),
            logits=loc,
            name='xentropy'
            ))

# add regularizers

with tf.name_scope('l2_regularization'):
    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    for var in vars:
        regularizers.append(args.l2*tf.nn.l2_loss(var))

op_loss_regularized=op_loss+tf.reduce_sum(regularizers)

# optimization nodes
global_step = tf.Variable(0, name='global_step', trainable=False)
optimizer = tf.train.AdamOptimizer(args.learningrate)
train_op = optimizer.minimize(op_loss_regularized, global_step=global_step)

########################################
print('preparing logging')
if args.log_name is None:
    args.log_name='data=%s,input=%s,output=%s,loss=%s,full=%s,learningrate=%f,l1=%f,l2=%f,dropout=%f,hashsize=%d,maxloc=%s,batchsize=%d'%(
        os.path.basename(args.data),
        args.input,
        args.output,
        args.loss,
        str(args.full),
        args.learningrate,
        args.l1,
        args.l2,
        args.dropout,
        args.hashsize,
        str(args.maxloc),
        args.batchsize
        )
log_dir=os.path.join(args.log_dir,args.log_name)
if not args.no_checkpoint:
    tf.gfile.MakeDirs(log_dir)
print('log_dir=',log_dir)

# create tf session
if args.tf_debug:
    from tensorflow.python import debug as tf_debug
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
else:
    sess = tf.Session()

summary = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=100000)
if not args.no_checkpoint:
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
sess.run(tf.global_variables_initializer())
sess.graph.finalize()

########################################
print('allocate datset files')

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
for file in files_all:
    if file_count%file_step==0:
        files_test.append(file)
    elif file_count%file_step==1:
        files_valid.append(file)
    else:
        files_train.append(file)

########################################
print('training')

stats_epoch={'count':0}
def reset_stats_epoch():
    stats_epoch['steps']=0
    stats_epoch['new']=False
    stats_epoch['loss']=0
    stats_epoch['dist']=0
    stats_epoch['err']=0
    stats_epoch['start_time']=time.time()
    stats_epoch['decoding_time']=0
reset_stats_epoch()
stats_epoch_prev=stats_epoch

stats_step={'count':0,'numlines':0}
def reset_stats_step():
    stats_step['loss']=0
    stats_step['dist']=0
    stats_step['err']=0
    stats_step['validtweets']=0
    stats_step['start_time']=time.time()
    stats_step['decoding_time']=0
    if not args.repeat_batch:
        stats_step['numlines']=0
reset_stats_step()

files_remaining=copy.deepcopy(files_train)
open_files=[]

while True:
    stats_step['count']+=1
    stats_epoch['steps']+=1

    if not args.repeat_batch or stats_step['count']==0:
        decoding_time_start=time.time()
        batch_dict=defaultdict(list)
        batch_dict_size=0
        while batch_dict_size < args.batchsize:

            # load and decode next json entry
            while len(open_files)<args.max_open_files and len(files_remaining)>0:
                index=random.randrange(len(files_remaining))
                filename=files_remaining[index]
                files_remaining.pop(index)
                open_files.append(open(filename,'rt'))
                print('  opening [%s]; files remaining: %d/%d; buffer state: %d/%d'%(
                    filename,
                    len(files_remaining),
                    len(files_train),
                    len(open_files),
                    args.max_open_files,
                    ))

            if len(open_files)==0:
                stats_epoch['new']=True
                stats_epoch['count']+=1
                files_remaining=copy.deepcopy(files_train)
                continue

            index=random.randrange(len(open_files))
            nextline=open_files[index].readline()
            if nextline=='':
                open_files[index].close()
                open_files.pop(index)
                continue

            try:
                data=json.loads(nextline)
            except Exception as e:
                print('current file=',open_files[index].name)
                print(e)
                continue

            stats_step['numlines']+=1

            # only process entries that contain a tweet
            if 'text' in data:

                # simplify the unicode representation
                data['text']=unicodedata.normalize('NFKC',unicode(data['text'].lower()))

                # possibly skip locations
                if args.filter_locations:
                    try:
                        full_name=data['place']['full_name']
                        if not (full_name in lochash):
                            continue
                    except:
                        continue
                stats_step['validtweets']+=1
                batch_dict_size+=1

                # hash features
                if 'bow' in args.input:
                    batch_dict[hash_].append(hv.transform([data['text']]))

                # text features
                if 'cltcc' in args.input:
                    encodedtext=np.zeros([1,tweetlen,args.cltcc_vocabsize])
                    def myhash(i):
                        return (5381*i)%args.cltcc_vocabsize
                        #if i>=ord('a') or i<=ord('z'):
                            #val=i-ord('a')
                        #else:
                            #val=5381*i
                        #return val%args.cltcc_vocabsize

                    for i in range(min(tweetlen,len(data['text']))):
                        encodedtext[0][i][myhash(ord(data['text'][i]))%args.cltcc_vocabsize]=1
                    batch_dict[text_].append(encodedtext)

                # language features
                if 'lang' in args.input:
                    batch_dict[lang_].append(hash_lang(data['lang']))

                # time features
                if 'time' in args.input:
                    timestamp = np.array(float(data['timestamp_ms']))
                    batch_dict[timestamp_ms_].append(timestamp)

                # get true output
                if args.calc_gps:
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

                if args.calc_loc:
                    full_name=data['place']['full_name']
                    locid=lochash[full_name]
                    batch_dict[loc_].append(np.array([locid]))

    decoding_time_stop=time.time()
    stats_step['decoding_time']+=decoding_time_stop-decoding_time_start
    stats_epoch['decoding_time']+=decoding_time_stop-decoding_time_start

    # create data dictionary

    feed_dict = {}

    if 'bow' in args.input:
        def mkSparseTensorValue(m):
            m2=sp.sparse.coo_matrix(m)
            if args.bow_dense:
                return m2.toarray()
            else:
                return tf.SparseTensorValue(
                    zip(m2.row,m2.col),
                    m2.data,
                    m2.shape,
                    )
        feed_dict[hash_] = mkSparseTensorValue(sp.sparse.vstack(batch_dict[hash_]))

    if 'cltcc' in args.input:
        feed_dict[text_] = np.vstack(batch_dict[text_])

    if 'lang' in args.input:
        feed_dict[lang_] = np.vstack(batch_dict[lang_])

    if 'time' in args.input:
        feed_dict[timestamp_ms_] = np.vstack(batch_dict[timestamp_ms_])

    if args.calc_gps:
        feed_dict[gps_] = np.vstack(batch_dict[gps_])

    if args.calc_loc:
        feed_dict[loc_] = np.vstack(batch_dict[loc_])

    # run the model
    _, loss_ave, dist_ave, err_ave = sess.run(
        [ train_op, op_loss_regularized, op_dist_ave, op_err_ave]
        , feed_dict=feed_dict
        )

    stats_step['loss']+=loss_ave
    stats_step['dist']+=dist_ave
    stats_step['err']+=err_ave

    stats_epoch['loss']+=loss_ave
    stats_epoch['dist']+=dist_ave
    stats_epoch['err']+=err_ave

    # Write the summaries and print an overview fairly often.
    if stats_step['count'] % args.stepdelta == 0:
        output='  %8d/%4d: loss=%1.2E  dist=%1.2E  err=%1.4f  good=%1.2f  dec=%1.2f' % (
              stats_step['count']
            , stats_epoch['count']
            , stats_step['loss']/args.stepdelta
            , stats_step['dist']/args.stepdelta
            , stats_step['err']/args.stepdelta
            , stats_step['validtweets']/float(stats_step['numlines'])
            , stats_step['decoding_time']/float(time.time()-stats_step['start_time'])
            )
        print(datetime.datetime.now(),output)

        # quit if we've gotten nan values
        if math.isnan(stats_step['loss']):
            raise ValueError('NaN loss')

        # save summaries if not debugging
        if not args.no_checkpoint:
            summary_str = sess.run(summary, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, stats_step['count'])
            summary_writer.flush()

        # save model if not debugging
        if stats_step['count'] % args.stepsave == 0 and not args.no_checkpoint:
            checkpoint_file = os.path.join(log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=stats_step['count'])

        # reset step variables
        reset_stats_step()

    if stats_epoch['new']:
        print('--------------------------------------------------------------------------------')
        print('epoch %d' % stats_epoch['count'])
        print('  time:  %s ' % str(datetime.timedelta(seconds=time.time() - stats_epoch['start_time'])))
        print('  steps: %d ' % stats_epoch['steps'])
        print('  loss:  %E    diff: %E' % (stats_epoch['loss']/float(stats_epoch['steps']),stats_epoch['loss']/float(stats_epoch['steps'])-stats_epoch_prev['loss']/float(stats_epoch_prev['steps'])))
        print('  dist:  %E    diff: %E' % (stats_epoch['dist']/float(stats_epoch['steps']),stats_epoch['dist']/float(stats_epoch['steps'])-stats_epoch_prev['dist']/float(stats_epoch['steps'])))
        print('  err:   %E    diff: %E' % (stats_epoch['err' ]/float(stats_epoch['steps']),stats_epoch['err' ]/float(stats_epoch['steps'])-stats_epoch_prev['err' ]/float(stats_epoch['steps'])))
        print('--------------------------------------------------------------------------------')
        stats_epoch_prev=copy.deepcopy(stats_epoch)

        # save model if not debugging
        if not args.no_checkpoint:
            checkpoint_file = os.path.join(log_dir, 'model.ckpt')
            saver.save(sess, checkpoint_file, global_step=stats_step['count'])

        # reset epoch variables
        reset_stats_epoch()


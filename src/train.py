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
parser.add_argument('--stepsave',type=int,default=10000)
parser.add_argument('--seed',type=int,default=0)
parser.add_argument('--max_open_files',type=int,default=96)

# debug variables
parser.add_argument('--no_checkpoint',action='store_true')
parser.add_argument('--tf_debug',action='store_true')
parser.add_argument('--repeat_batch',action='store_true')

# model hyperparameters
parser.add_argument('--data',type=str,required=True)
parser.add_argument('--data_summary',type=str,default=None)

parser.add_argument('--batchsize',type=int,default=100)
parser.add_argument('--learningrate',type=float,default=0.005)
parser.add_argument('--optimizer',choices=['adam','sgd'],default='adam')
parser.add_argument('--momentum',type=float,default=0.9)
parser.add_argument('--decay',type=float,default=1e-5)
parser.add_argument('--dropout',type=float,default=0.5)
parser.add_argument('--l1',type=float,default=0.0)
parser.add_argument('--l2',type=float,default=1e-5)

parser.add_argument('--input',choices=['cnn','bow','lang','time','const'],nargs='+',required=True)
parser.add_argument('--bow_hashsize',type=int,default=20)
parser.add_argument('--bow_layersize',type=int,default=2)
parser.add_argument('--bow_dense',action='store_true')
parser.add_argument('--cnn_type',choices=['vdcnn','cltcc'],default='cltcc')
parser.add_argument('--cnn_vocabsize',type=int,default=128)
parser.add_argument('--cnn_khot',type=int,default=1)
parser.add_argument('--vdcnn_numfilters',type=int,default=64)
parser.add_argument('--vdcnn_variance',type=float,default=0.02)
parser.add_argument('--vdcnn_resnet',action='store_true')
parser.add_argument('--cltcc_numfilters',type=int,default=1024)
parser.add_argument('--cltcc_variance',type=float,default=0.02)

parser.add_argument('--full',type=int,nargs='*',default=[])

parser.add_argument('--output',choices=['pos','country','loc'],default=['pos','country','loc'],nargs='+')
parser.add_argument('--pos_type',choices=['naive','aglm','aglm2','aglm_mix','proj3d'],default='aglm')
parser.add_argument('--pos_loss',choices=['l2','chord','dist','dist2','angular'],default='dist')
parser.add_argument('--pos_shortcut',choices=['loc','country'],default=['country'],nargs='*')
parser.add_argument('--aglm_components',type=int,default=128)
parser.add_argument('--country_shortcut',choices=['bow','lang'],default=['lang'],nargs='*')
parser.add_argument('--loc_type',choices=['popular','hash'],default='hash')
parser.add_argument('--loc_max',default=10,type=int)
parser.add_argument('--loc_filter',action='store_true')
parser.add_argument('--loc_hashsize',type=int,default=12)
parser.add_argument('--loc_shortcut',choices=['bow','lang'],default=[],nargs='*')

args = parser.parse_args()

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
import unicodedata

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
#var_init = lambda shape,var: tf.truncated_normal(shape,stddev=var,seed=args.seed)

def var_init(shape,stddev):
    var_init.count+=1
    size=float(sum(shape))
    return tf.truncated_normal(shape,stddev=math.sqrt(2.0/size),seed=args.seed+var_init.count)
var_init.count=0

# tf inputs
with tf.name_scope('inputs'):

    regularizers=[]
    inputs=[]

    # hash bow inputs
    if 'bow' in args.input:
        with tf.name_scope('bow'):
            bow_size=2**args.bow_hashsize
            if args.bow_dense:
                hash_ = tf.placeholder(tf.float32,[args.batchsize,bow_size])
                matmul = tf.matmul
                hash_reg=args.l1*tf.reduce_sum(tf.abs(hash_))
            else:
                hash_ = tf.sparse_placeholder(tf.float32,name='hash_')
                matmul = tf.sparse_tensor_dense_matmul
                hash_reg=args.l1*tf.sparse_reduce_sum(tf.abs(hash_))
            regularizers.append(hash_reg)
            w = tf.Variable(var_init([bow_size,args.bow_layersize],1.0))
            b = tf.constant(0.1,shape=[args.bow_layersize])
            bow = matmul(hash_,w)+b
            inputs.append(bow)

    # cnn inputs
    tweetlen=280
    if 'cnn' in args.input:
        text_ = tf.placeholder(tf.float32, [args.batchsize,tweetlen,args.cnn_vocabsize])
        text_reshaped = tf.reshape(text_,[args.batchsize,tweetlen,args.cnn_vocabsize,1])

        # Very Deep Convolutional Neural Network
        # follows paper "very deep convolutional networks for text classification"
        if 'vdcnn' == args.cnn_type:
            s=tweetlen
            with tf.name_scope('vdcnn'):
                def mk_conv(prev,numin,numout,swapdim=False):
                    mk_conv.count+=1
                    with tf.name_scope('conv'+str(mk_conv.count)):
                        if swapdim:
                            shape=[3,1,numin,numout]
                        else:
                            shape=[3,numin,1,numout]
                        w = tf.Variable(var_init(shape,0.1))
                        #w = tf.get_variable('w',
                            #shape=shape,
                            #initializer=tf.glorot_normal_initializer(seed=2000+args.seed+mk_conv.count)
                            #initializer=tf.truncated_normal_initializer(0.1)
                            #)
                        b = tf.Variable(tf.constant(0.1,shape=[numout]))
                        conv = tf.nn.conv2d(prev, w, strides=[1,1,1,1], padding='SAME')
                        return tf.nn.bias_add(conv,b)
                mk_conv.count=0

                def mk_conv_block(input,numin,numout,size=2):
                    net=input
                    print('input=',input)
                    with tf.name_scope('conv_block'):
                        for i in range(0,size):
                            #net = mk_conv(net,numin,numout,swapdim=True)
                            w = tf.Variable(var_init([3,1,numin,numout],0.1))
                            b = tf.Variable(tf.constant(0.1,shape=[numout]))
                            conv = tf.nn.conv2d(net, w, strides=[1,1,1,1], padding='SAME')
                            net= tf.nn.bias_add(conv,b)
                            numin=numout
                            print('net=',net)
                            #net = tf.nn.relu(net)
                        if args.vdcnn_resnet:
                            paddims=np.zeros([4,2])
                            for i in range(0,4):
                                paddims[i][0]=0
                                diff=abs(int(net.get_shape()[i])-int(input.get_shape()[i]))
                                paddims[i][1]=diff
                            input2=tf.pad(input,paddims)
                            return net+input2
                        else:
                            return net

                def pool2(prev):
                    return tf.nn.max_pool(
                        prev,
                        ksize=[1, 2, 1, 1],
                        strides=[1, 2, 1, 1],
                        padding='VALID')

                net = mk_conv(text_reshaped,args.cnn_vocabsize,args.vdcnn_numfilters)
                net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                #net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                #net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                #net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                #net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                #net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                #net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                #net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                #net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                #net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                #net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                #net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                #net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                #net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                #net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                #net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                net = pool2(net)
                net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters*2)
                net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                #net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                #net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                #net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                #net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                #net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                #net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                #net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                #net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                #net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                #net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                #net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                #net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                #net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                #net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                net = pool2(net)
                net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*4)
                net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                #net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                #net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                #net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                #net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                #net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                #net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                #net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                #net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                net = pool2(net)
                net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*8)
                net = mk_conv_block(net,args.vdcnn_numfilters*8,args.vdcnn_numfilters*8)
                #net = mk_conv_block(net,args.vdcnn_numfilters*8,args.vdcnn_numfilters*8)
                #net = mk_conv_block(net,args.vdcnn_numfilters*8,args.vdcnn_numfilters*8)
                #net = mk_conv_block(net,args.vdcnn_numfilters*8,args.vdcnn_numfilters*8)
                #net = mk_conv_block(net,args.vdcnn_numfilters*8,args.vdcnn_numfilters*8)
                net = pool2(net)
                print('net=',net)

                input_size=int(net.get_shape()[1]*net.get_shape()[2]*net.get_shape()[3])
                input_=tf.reshape(net,[args.batchsize,input_size])
                inputs.append(input_)

        # follows paper "character level convnets for text classification"
        # see also: Language-Independent Twitter Classification Using Character-Based Convolutional Networks
        if 'cltcc' == args.cnn_type:
            activation=tf.nn.relu
            with tf.name_scope('cltcc'):
                filterlen=7
                with tf.name_scope('conv1'):
                    w = tf.Variable(var_init([filterlen,args.cnn_vocabsize,1,args.cltcc_numfilters],args.cltcc_variance))
                    b = tf.Variable(tf.constant(0.1,shape=[args.cltcc_numfilters]))
                    conv = tf.nn.conv2d(text_reshaped, w, strides=[1,1,1,1], padding='VALID')
                    h = activation(tf.nn.bias_add(conv,b))
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, 3, 1, 1],
                        strides=[1, 3, 1, 1],
                        padding='VALID')

                with tf.name_scope('conv2'):
                    w = tf.Variable(var_init([filterlen,1,args.cltcc_numfilters,args.cltcc_numfilters],args.cltcc_variance))
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
                    w = tf.Variable(var_init([filterlen,1,args.cltcc_numfilters,args.cltcc_numfilters],args.cltcc_variance))
                    b = tf.Variable(tf.constant(0.1,shape=[args.cltcc_numfilters]))
                    conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
                    h = activation(tf.nn.bias_add(conv,b))
                    pooled = h

                with tf.name_scope('conv4'):
                    w = tf.Variable(var_init([filterlen,1,args.cltcc_numfilters,args.cltcc_numfilters],args.cltcc_variance))
                    b = tf.Variable(tf.constant(0.1,shape=[args.cltcc_numfilters]))
                    conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
                    h = activation(tf.nn.bias_add(conv,b))
                    pooled = h

                with tf.name_scope('conv5'):
                    w = tf.Variable(var_init([filterlen,1,args.cltcc_numfilters,args.cltcc_numfilters],args.cltcc_variance))
                    b = tf.Variable(tf.constant(0.1,shape=[args.cltcc_numfilters]))
                    conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
                    h = activation(tf.nn.bias_add(conv,b))
                    pooled = h

                with tf.name_scope('conv6'):
                    w = tf.Variable(var_init([filterlen,1,args.cltcc_numfilters,args.cltcc_numfilters],args.cltcc_variance))
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
            lang_one_hot = tf.reshape(tf.one_hot(lang_,len(langs),axis=1),shape=[args.batchsize,len(langs)])
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
    with tf.name_scope('flattened_inputs'):
        final_layer=tf.concat(map(tf.contrib.layers.flatten,inputs),axis=1)
        final_layer_size=int(final_layer.get_shape()[1])

    for layersize in args.full:
        with tf.name_scope('full%d'%layerindex):
            w = tf.Variable(var_init([final_layer_size,layersize],0.1))
            #w = tf.get_variable('w',
                #shape=[final_layer_size, layersize],
                ##initializer=tf.contrib.layers.xavier_initializer())
                #initializer=tf.truncated_normal_initializer(0.1))
            b = tf.Variable(tf.constant(0.1,shape=[layersize]))
            h = tf.nn.relu(tf.matmul(final_layer,w)+b)
            final_layer=tf.nn.dropout(h,args.dropout)
            final_layer_size=layersize
        layerindex+=1

# rf outputs
with tf.name_scope('output'):

    op_losses={}
    op_metrics={}

    # country hash
    if 'country' in args.output:
        country_ = tf.placeholder(tf.int64, [args.batchsize,1])

        # hash function
        country_codes_iso=['AF','AL','DZ','AS','AD','AO','AI','AQ','AG','AR','AM','AW','AU','AT','AZ','BS','BH','BD','BB','BY','BE','BZ','BJ','BM','BT','BO','BA','BW','BR','IO','VG','BN','BG','BF','BI','KH','CM','CA','CV','KY','CF','TD','CL','CN','CX','CC','CO','KM','CK','CR','HR','CU','CW','CY','CZ','CD','DK','DJ','DM','DO','TL','EC','EG','SV','GQ','ER','EE','ET','FK','FO','FJ','FI','FR','PF','GA','GM','GE','DE','GH','GI','GR','GL','GD','GU','GT','GG','GN','GW','GY','HT','HN','HK','HU','IS','IN','ID','IR','IQ','IE','IM','IL','IT','CI','JM','JP','JE','JO','KZ','KE','KI','XK','KW','KG','LA','LV','LB','LS','LR','LY','LI','LT','LU','MO','MK','MG','MW','MY','MV','ML','MT','MH','MR','MU','YT','MX','FM','MD','MC','MN','ME','MS','MA','MZ','MM','NA','NR','NP','NL','AN','NC','NZ','NI','NE','NG','NU','KP','MP','NO','OM','PK','PW','PS','PA','PG','PY','PE','PH','PN','PL','PT','PR','QA','CG','RE','RO','RU','RW','BL','SH','KN','LC','MF','PM','VC','WS','SM','ST','SA','SN','RS','SC','SL','SG','SX','SK','SI','SB','SO','ZA','KR','SS','ES','LK','SD','SR','SJ','SZ','SE','CH','SY','TW','TJ','TZ','TH','TG','TK','TO','TT','TN','TR','TM','TC','TV','VI','UG','UA','AE','GB','US','UY','UZ','VU','VA','VE','VN','WF','EH','YE','ZM','ZW']
        country_codes_empirical=['GP','MQ','GF','BQ','AX','BV','TF','NF','UM','GS','HM']
        country_codes=['']+country_codes_iso+country_codes_empirical
        def hash_country(str):
            try:
                return country_codes.index(str)
            except:
                print('unknown country code = [',str,']')
                return 0

        with tf.name_scope('country'):

            # shortcuts
            final_layer_country = final_layer
            final_layer_country_size = final_layer_size

            if 'lang' in args.country_shortcut:
                try:
                    final_layer_country = tf.concat([final_layer_country,lang_one_hot],axis=1)
                    final_layer_country_size += len(langs)
                except:
                    pass

            if 'bow' in args.pos_shortcut:
                try:
                    final_layer_country = tf.concat([final_layer_country,bow],axis=1)
                    final_layer_country_size += args.bow_layersize
                except:
                    pass

            # layer
            w = tf.Variable(tf.zeros([final_layer_country_size,len(country_codes)]))
            b = tf.Variable(tf.zeros([len(country_codes)]))
            logits = tf.matmul(final_layer_country,w)+b

            xentropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.to_int64(tf.reshape(country_,[args.batchsize])),
                logits=logits,
                name='country_xentropy'
                ))
            op_losses['country_xentropy']=xentropy
            op_metrics['country_xentropy']=tf.contrib.metrics.streaming_mean(xentropy,name='country_xentropy')

            country_softmax=tf.nn.softmax(logits)
            country=tf.reshape(tf.argmax(logits,axis=1),shape=[args.batchsize,1])
            op_metrics['country_acc']=tf.contrib.metrics.streaming_accuracy(country,country_,name='country_acc')

            def topk(k):
                name='top'+str(k)
                with tf.name_scope(name):
                    topk=tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits,tf.reshape(country_,shape=[args.batchsize]),k=k),tf.float32))
                    op_metrics[name]=tf.contrib.metrics.streaming_mean(topk,name=name)

            topk(1)
            topk(3)
            topk(5)
            topk(10)

    # loc buckets
    if 'loc' in args.output:
        loc_ = tf.placeholder(tf.int64, [args.batchsize,1])

        # hash
        import hashlib
        numloc=2**args.loc_hashsize
        def hash_loc(str):
            return hash(str)%numloc

        with tf.name_scope('loc'):
            # shortcuts
            final_layer_pos = final_layer
            final_layer_pos_size = final_layer_size

            if 'lang' in args.country_shortcut:
                final_layer_pos = tf.concat([final_layer_pos,lang_one_hot],axis=1)
                final_layer_pos_size += len(langs)

            if 'bow' in args.pos_shortcut:
                final_layer_pos = tf.concat([final_layer_pos,bow],axis=1)
                final_layer_pos_size += args.bow_layersize

            if 'country' in args.pos_shortcut:
                final_layer_pos = tf.concat([final_layer_pos,country_softmax],axis=1)
                final_layer_pos_size += len(country_codes)

            # layer
            w = tf.Variable(tf.zeros([final_layer_pos_size, numloc]))
            b = tf.Variable(tf.zeros([numloc]))
            logits = tf.matmul(final_layer_pos,w)+b

            xentropy = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.to_int64(tf.reshape(loc_,[args.batchsize])),
                    logits=logits,
                    name='xentropy'
                    ))
            op_losses['loc_xentropy']=xentropy
            op_metrics['loc_xentropy']=tf.contrib.metrics.streaming_mean(xentropy,name='loc_xentropy')

            loc_softmax=tf.nn.softmax(logits)
            loc=tf.reshape(tf.argmax(logits,axis=1),shape=[args.batchsize,1])
            op_metrics['loc_acc']=tf.contrib.metrics.streaming_accuracy(loc,loc_,name='loc_acc')


            # FIXME: this is from the old way of inserting locs
            if False:
                #probs = tf.nn.softmax(loc)
                #np_lats=np.array(lats_hash,dtype=np.float32)
                #np_lons=np.array(lons_hash,dtype=np.float32)
                #op_lats_hash=tf.convert_to_tensor(np_lats)
                #op_lons_hash=tf.convert_to_tensor(np_lons)
                #op_lat = tf.reduce_sum(tf.multiply(op_lats_hash,probs),1)
                #op_lon = tf.reduce_sum(tf.multiply(op_lons_hash,probs),1)
                #gps = tf.transpose(tf.stack([op_lat,op_lon]))
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

                loc_maxs=10
                if args.loc_max:
                    loc_maxs=args.loc_max
                else:
                    loc_maxs=len(num_fn['city'].keys())
                locsfreqs=list(reversed(sorted(zip(num_fn['city'].values(),num_fn['city'].keys()))))[0:loc_maxs]
                locs=map(lambda (a,b):b,list(reversed(sorted(zip(num_fn['city'].values(),num_fn['city'].keys()))))[0:loc_maxs])

                # REMEMBER: twitter stores coordinates in (lon,lat) form instead of (lat,lon)
                locscoords=[centroid(loc_fn['city'][loc]['coordinates']) for loc in locs]
                lats_hash=[lat for (lon,lat) in locscoords]
                lons_hash=[lon for (lon,lat) in locscoords]

                lochash={ loc : i for loc,i in zip(locs,xrange(len(locs))) }
                from pprint import pprint

    # position based losses
    if 'pos' in args.output:
        gps_ = tf.placeholder(tf.float32, [args.batchsize,2])

        with tf.name_scope('pos'):

            # shortcuts
            pos_final_layer = final_layer
            pos_final_layer_size = final_layer_size

            if 'loc' in args.pos_shortcut:
                try:
                    pos_final_layer = tf.concat([pos_final_layer,loc_softmax],axis=1)
                    pos_final_layer_size += numloc
                except:
                    pass

            if 'country' in args.pos_shortcut:
                try:
                    pos_final_layer = tf.concat([pos_final_layer,country_softmax],axis=1)
                    pos_final_layer_size += len(country_codes)
                except:
                    pass

            # decompose true labels
            op_lat_ = gps_[:,0]
            op_lon_ = gps_[:,1]
            op_lat_rad_ = op_lat_/360*2*math.pi
            op_lon_rad_ = op_lon_/360*2*math.pi

            # treat gps coords as R^2
            if 'naive' == args.pos_type:
                w = tf.Variable(tf.zeros([pos_final_layer_size, 2]))
                if args.warmstart:
                    b = tf.Variable([34.052235,-118.243683])
                else:
                    b = tf.Variable(tf.zeros([2]))
                gps = tf.matmul(pos_final_layer,w) + b
                op_lat = gps[:,0]
                op_lon = gps[:,1]

            # angular generalized linear model
            # See: "Regression Models for Angular Response" by Fisher and Lee
            if 'aglm' == args.pos_type:
                w = tf.Variable(tf.zeros([pos_final_layer_size, 2]))
                if args.warmstart:
                    b = tf.Variable([0.6745,-2])
                else:
                    b = tf.Variable(tf.zeros([2]))
                response = tf.matmul(pos_final_layer,w) + b
                op_lat = tf.atan(response[:,0])*360/2/math.pi
                op_lon = tf.atan(response[:,1])*360/math.pi
                gps = tf.stack([op_lat,op_lon],1)

            # same as algm, but uses the bias outside of the atan embedding
            if 'aglm2' == args.pos_type:
                w = tf.Variable(tf.zeros([pos_final_layer_size, 2]))
                if args.warmstart:
                    b0 = tf.Variable([34.052235])
                    b1 = tf.Variable([-118.243683])
                else:
                    b0 = tf.Variable(tf.zeros([1]))
                    b1 = tf.Variable(tf.zeros([1]))
                response = tf.matmul(pos_final_layer,w)
                op_lat = tf.atan(response[:,0])*360/math.pi/2 + b0
                op_lon = tf.atan(response[:,1])*360/math.pi   + b1
                gps = tf.stack([op_lat,op_lon],1)

            # mixture of aglm responses
            if 'aglm_mix' == args.pos_type:
                w = tf.Variable(var_init([pos_final_layer_size,args.aglm_components],0.1))
                b = tf.Variable(tf.constant(0.1,shape=[args.aglm_components]))
                mixture = tf.nn.softmax(tf.matmul(pos_final_layer,w)+b)

                w = tf.Variable(var_init([pos_final_layer_size,args.aglm_components,2],0.1))
                b = tf.Variable(tf.constant(0.1,shape=[args.aglm_components, 2]))
                responses = tf.tensordot(pos_final_layer,w,axes=[[1],[0]])

                reshaped=tf.reshape(mixture,shape=[args.batchsize,args.aglm_components,1])
                tiled=tf.tile(reshaped,[1,1,2])
                response=tf.reduce_sum(responses*tiled,axis=1)

                op_lat = tf.atan(response[:,0])*360/2/math.pi
                op_lon = tf.atan(response[:,1])*360/math.pi
                gps = tf.stack([op_lat,op_lon],1)

            # model gps coordinates in R^3
            if 'proj3d' == args.pos_type:
                w = tf.Variable(tf.zeros([pos_final_layer_size, 3]))
                b = tf.Variable([0.1,0,0])
                r3 = tf.matmul(pos_final_layer,w) + b
                norm = tf.sqrt(tf.reduce_sum(r3*r3,1))
                r3normed = r3/tf.stack([norm,norm,norm],1)
                #op_lon = tf.asin(tf.minimum(1.0,r3normed[:,2]))
                #op_lat = tf.asin(tf.minimum(1.0,r3normed[:,1]))*tf.acos(tf.minimum(1.0,r3normed[:,2]))
                op_lon = tf.asin(r3normed[:,2])
                op_lat = tf.asin(r3normed[:,1])*tf.acos(r3normed[:,2])
                gps = tf.stack([op_lat,op_lon],1)
                #op_lon=tf.Print(op_lon,[gps,b,norm])

            # common outputs

            epsilon = 1e-6

            op_lat_rad = op_lat/360*2*math.pi
            op_lon_rad = op_lon/360*2*math.pi

            hav = lambda x: tf.sin(x/2)**2
            squared_angular_dist = ( hav(op_lat_rad-op_lat_rad_)
                            +tf.cos(op_lat_rad)*tf.cos(op_lat_rad_)*hav(op_lon_rad-op_lon_rad_)
                           )

            # radius of earth = 3959 miles, 6371 kilometers
            op_dist = 2*6371*tf.asin(tf.sqrt(tf.maximum(epsilon,squared_angular_dist)))
            op_dist_ave = tf.reduce_sum(op_dist)/args.batchsize

            op_delta_x = tf.cos(op_lat_rad)*tf.cos(op_lon_rad)-tf.cos(op_lat_rad_)*tf.cos(op_lon_rad_)
            op_delta_y = tf.cos(op_lat_rad)*tf.sin(op_lon_rad)-tf.cos(op_lat_rad_)*tf.sin(op_lon_rad_)
            op_delta_z = tf.sin(op_lat_rad) - tf.sin(op_lat_rad_)
            op_chord = tf.sqrt(epsilon + op_delta_x**2 + op_delta_y**2 + op_delta_z**2)
            #op_dist = 2*6371*tf.asin(op_chord/2)
            #op_dist_ave = tf.reduce_sum(op_dist)/args.batchsize

            # set loss function
            if args.pos_loss=='l2':
                op_loss = tf.reduce_sum((gps - gps_) * (gps - gps_))
            if args.pos_loss=='chord':
                op_loss = tf.reduce_sum(op_chord)/args.batchsize
            if args.pos_loss=='dist':
                op_loss = op_dist_ave
            if args.pos_loss=='dist2':
                op_loss = tf.reduce_sum(op_dist*op_dist)/args.batchsize
            if args.pos_loss=='angular':
                op_loss = tf.reduce_sum(squared_angular_dist)

            op_losses['dist']=op_loss
            op_metrics['dist']=tf.contrib.metrics.streaming_mean(op_dist_ave,name='dist')

            def mk_threshold(threshold):
                op_threshold = tf.sign(op_dist-threshold)/2+0.5
                op_threshold_ave = tf.reduce_mean(op_threshold)
                name='k'+str(threshold)
                op_metrics[name]=tf.contrib.metrics.streaming_mean(op_threshold_ave,name=name)
            mk_threshold(10)
            mk_threshold(50)
            mk_threshold(100)
            mk_threshold(500)
            mk_threshold(1000)
            mk_threshold(2000)
            mk_threshold(3000)

# set loss function
with tf.name_scope('loss'):

    #with tf.name_scope('combined_loss'):
        #loss_total=0
        #for k,v in op_losses.iteritems():
            #loss_total+=op_metrics[k][0]
        #op_loss=0
        #for k,v in op_losses.iteritems():
            #op_loss+=op_losses[k]*(op_metrics[k][0]/loss_total)

    op_loss = op_losses['dist']/1000 + op_losses['country_xentropy'] # + op_losses['loc_xentropy']/1000

    #op_loss = tf.reduce_mean(op_losses.values())

    op_metrics['op_loss']=tf.contrib.metrics.streaming_mean(op_loss,name='op_loss')

    # add regularizers
    with tf.name_scope('l2_regularization'):
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in vars:
            regularizers.append(args.l2*tf.nn.l2_loss(var))

    op_loss_regularized=op_loss+tf.reduce_sum(regularizers)
    op_losses['op_loss_regularized']=op_loss_regularized

# optimization nodes
with tf.name_scope('optimization'):
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learningrate = tf.train.inverse_time_decay(args.learningrate, global_step, 1, args.decay)
    if args.optimizer=='adam':
        optimizer = tf.train.AdamOptimizer(learningrate)
    elif args.optimizer=='sgd':
        optimizer = tf.train.MomentumOptimizer(learningrate,args.momentum)
    train_op = optimizer.minimize(op_loss_regularized, global_step=global_step)

########################################
print('preparing logging')
if args.log_name is None:
    args.log_name='data=%s,input=%s,output=%s,loss=%s,full=%s,learningrate=%f,l1=%f,l2=%f,dropout=%f,bow_hashsize=%d,loc_max=%s,batchsize=%d'%(
        os.path.basename(args.data),
        args.input,
        args.output,
        args.pos_loss,
        str(args.full),
        args.learningrate,
        args.l1,
        args.l2,
        args.dropout,
        args.bow_hashsize,
        str(args.loc_max),
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

print('XXXXX')
saver = tf.train.Saver(max_to_keep=100000)
print('XXXXX')
if not args.no_checkpoint:
    for k,(v,_) in op_metrics.iteritems():
        tf.summary.scalar(k,v)
    summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph)
print('XXXXX')

reset_global_vars=tf.global_variables_initializer()
reset_local_vars=tf.local_variables_initializer()
sess.graph.finalize()
print('XXXXX')
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

                # FIXME: possibly skip locations
                #if args.loc_filter:
                    #try:
                        #full_name=data['place']['full_name']
                        #if not (full_name in lochash):
                            #continue
                    #except:
                        #continue
                stats_step['validtweets']+=1
                batch_dict_size+=1

                # get inputs
                if 'bow' in args.input:
                    batch_dict[hash_].append(hv.transform([data['text']]))

                if 'cnn' in args.input:
                    encodedtext=np.zeros([1,tweetlen,args.cnn_vocabsize])
                    def myhash(i):
                        return (5381*i)%args.cnn_vocabsize

                    for i in range(min(tweetlen,len(data['text']))):
                        for k in range(0,args.cnn_khot):
                            index=(5381*i + 88499*k)%args.cnn_vocabsize
                            encodedtext[0][i][index]=1

                    batch_dict[text_].append(encodedtext)

                if 'lang' in args.input:
                    batch_dict[lang_].append(hash_lang(data['lang']))

                if 'time' in args.input:
                    timestamp = np.array(float(data['timestamp_ms']))
                    batch_dict[timestamp_ms_].append(timestamp)

                # get true output
                if 'pos' in args.output:
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

                if 'country' in args.output:
                    try:
                        country_code=hash_country(data['place']['country_code'])
                    except:
                        country_code=0
                    batch_dict[country_].append(np.array([country_code]))

                if 'loc' in args.output:
                    try:
                        loc_code=hash_loc(data['place']['full_name'])
                    except:
                        loc_code=0
                    batch_dict[loc_].append(np.array([loc_code]))

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

    if 'cnn' in args.input:
        feed_dict[text_] = np.vstack(batch_dict[text_])

    if 'lang' in args.input:
        feed_dict[lang_] = np.vstack(batch_dict[lang_])

    if 'time' in args.input:
        feed_dict[timestamp_ms_] = np.vstack(batch_dict[timestamp_ms_])

    if 'country' in args.output:
        feed_dict[country_] = np.vstack(batch_dict[country_])

    if 'pos' in args.output:
        feed_dict[gps_] = np.vstack(batch_dict[gps_])

    if 'loc' in args.output:
        feed_dict[loc_] = np.vstack(batch_dict[loc_])

    # run the model
    _, metrics = sess.run(
        [ train_op, op_metrics]
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
            summary_writer.add_summary(summary_str, stats_step['count'])
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


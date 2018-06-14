#!/usr/bin/env python

from __future__ import print_function

# set unbuffered output
import os
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

########################################
print('processing cmd line args')
import argparse

parser=argparse.ArgumentParser('infer using a model')

parser.add_argument('--modeldir',type=str)
parser.add_argument('--outputdir',type=str,default='tmp')
parser.add_argument('--tweets',type=str,default='data/twitter-early/geoTwitter17-10-21/geoTwitter17-10-21_01:05.gz')
#parser.add_argument('--tweets',type=str,default='data/MX/train.gz')
parser.add_argument('--maxtweets',type=int,default=10)

args = parser.parse_args()

########################################
print('loading model')

# load args
import simplejson as json
args_str=json.dumps(vars(args))
with open(args.modeldir+'/args.json','r') as f:
    args_str=f.readline()

    class Bunch(object):
        def __init__(self, adict):
            self.__dict__.update(adict)
    args_train=Bunch(json.loads(args_str))
    args_train.batchsize=1
    args_train.dropout=1

# load weights
hashfn=hash
if True:
    import tensorflow as tf
    import model
    import hash

    input_tensors={
        'country_' : tf.placeholder(tf.int64, [args_train.batchsize,1],name='country_'),
        'text_' : tf.placeholder(tf.float32, [args_train.batchsize,model.tweetlen,args_train.cnn_vocabsize],name='text_'),
        'gps_' : tf.placeholder(tf.float32, [args_train.batchsize,2], name='gps_'),
        'loc_' : tf.placeholder(tf.int64, [args_train.batchsize,1],name='loc_'),
        'timestamp_ms_' : tf.placeholder(tf.float32, [args_train.batchsize,1], name='timestamp_ms_'),
        'lang_' : tf.placeholder(tf.int32, [args_train.batchsize,1], 'lang_'),
        'newuser_' : tf.placeholder(tf.float32, [args_train.batchsize,1],name='newuser_'),
        'hash_' : tf.sparse_placeholder(tf.float32,name='hash_'),
    }

    op_metrics,op_loss_regularized,op_outputs = model.inference(args_train,input_tensors)

    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.Session(config=config)
    saver=tf.train.Saver()
    chkpt_file=tf.train.latest_checkpoint(args.modeldir)
    saver.restore(sess,chkpt_file)

########################################
print('plotting')

# load tweets
import datetime
import gzip
import math
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

f=gzip.open(args.tweets,'r')

for i in range(0,args.maxtweets):

    print(datetime.datetime.now(),'tweet',i)
    line=f.readline()

    # get tweet data
    tweet=json.loads(line)
    input=model.json2dict(args_train,line)
    print('  input[gps_]=',input['gps_'])
    lat_=input['gps_'][0]
    lon_=input['gps_'][1]

    # get model data
    feed_dict=model.mk_feed_dict(args_train,[input])
    output=sess.run(op_outputs, feed_dict=feed_dict)
    print('  output[gps]=',np.reshape(output['gps'],[2]))

    pre_mu_gps=output['aglm_mix/pre_mu']
    print('pre_mu_gps=',pre_mu_gps)
    print('pre_mu_gps[:,0]=',pre_mu_gps[:,0])
    print('pre_mu_gps[:,1]=',pre_mu_gps[:,1])


    mu=output['aglm_mix/mu']

    def gps2sphere(lat,lon):
        lat_rad=lat/360*2*math.pi
        lon_rad=lon/360*math.pi
        x = np.stack([ np.cos(lat_rad)
                     , np.sin(lat_rad) * np.cos(lon_rad)
                     , np.sin(lat_rad) * np.sin(lon_rad)
                     ])
        return x

    print('mu=',mu)
    mu=gps2sphere(np.arctan(pre_mu_gps[:,0]),np.arctan(pre_mu_gps[:,1]))
    print('mu=',mu)

    #pre_kappa=output['aglm_mix/pre_kappa']
    #kappa=output['aglm_mix/kappa']

    pre_kappa=5
    kappa=math.exp(pre_kappa)

    mixture=output['aglm_mix/mixture']
    print('mixture=',mixture)
    mixture_hard=(mixture == mixture.max(axis=1)[:,None]).astype(float)
    mixture=mixture_hard
    print('mixture_hard=',mixture_hard)

    mixture=np.ones(mixture.shape)
    #mixture[0][i]=1.0

    # setup plot
    plt.figure(i)
    m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,\
                llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
    m.drawcoastlines()
    #m.fillcontinents(color='white',lake_color='white')
    #m.drawmapboundary(fill_color='white')
    m.drawcountries()

    # plot contours
    def log_likelihood(lat,lon):
        #lat_rad=lat/360*2*math.pi
        #lon_rad=lon/360*math.pi
        #x = np.stack([ np.cos(lat_rad)
                     #, np.sin(lat_rad) * np.cos(lon_rad)
                     #, np.sin(lat_rad) * np.sin(lon_rad)
                     #])
        x = gps2sphere(lat,lon)
        x_reshape=np.reshape(x,[3,1,1])
        #return np.dot(kappa * np.sum(x_reshape*mu,axis=0),np.transpose(mixture))[0,0]
        #print('a=',x_reshape*mu)
        #print('b=',np.sum(x_reshape*mu,axis=0))
        #print('c=',pre_kappa - np.log(np.sinh(kappa)) + kappa*np.sum(x_reshape*mu,axis=0))
        return np.dot(np.exp(pre_kappa - np.log(np.sinh(kappa)) + kappa * np.sum(x_reshape*mu,axis=0)),np.transpose(mixture))[0,0]

    nx=100
    ny=50
    lons, lats = m.makegrid(nx, ny)
    z=np.zeros([nx,ny])
    for x in range(0,nx):
        for y in range(0,ny):
            #z[x][y]=log_likelihood(lats[y,x],lons[y,x])
            z[x][y]=log_likelihood(lats[y,x],lons[y,x])
    #z = np.fromfunction(lambda x,y: log_likelihood(lons[x,y],lats[x,y]),shape=[ny,nx],dtype=int)

    cs = m.contourf(lons,lats,np.transpose(z),latlon=True)
    cbar = m.colorbar(cs,location='right',pad="5%")
    cbar.set_label('log likelihood (unnormalized)')

    # plot points
    x, y = m(lon_, lat_)
    m.scatter(x, y, marker='D',color='m')

    x, y = m(pre_mu_gps[:,1], pre_mu_gps[:,0])
    m.scatter(x, y, marker='D',color='green')

    x, y = m(output['gps'][1], output['gps'][0])
    m.scatter(x, y, marker='.',color='red')

    # add tweet title
    wrappedtweet='\n'.join(textwrap.wrap(tweet['text'],60))
    plt.title(wrappedtweet)

    # add gps info
    str_gps="gps_ = %2.1f, %2.1f \ngps   = %2.1f, %2.1f"%(
        input['gps_'][0],
        input['gps_'][1],
        output['gps'][0],
        output['gps'][1]
        )
    plt.gcf().text(0.2, 0.0, str_gps, fontsize=10)

    # add country info
    str_country="country_ = %s\ncountry   = %s"%(
        hash.int2country(input['country_'][0]),
        hash.softmax2country(output['country_softmax'])
        )
    plt.gcf().text(0.5, 0.0, str_country, fontsize=10)

    # add language info
    str_lang="lang_ = %s\nlang   = %s"%(
        hash.int2lang(input['lang_']),
        hash.softmax2lang(output['lang_softmax'])
        )
    plt.gcf().text(0.8, 0.0, str_lang, fontsize=10)

    # save
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(args.outputdir+'/'+str(i)+'.png', bbox_inches='tight')

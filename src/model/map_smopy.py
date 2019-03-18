#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

# set unbuffered output
import os
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

########################################
print('processing cmd line args')
import argparse

parser=argparse.ArgumentParser('infer using a model')

parser.add_argument('--modeldir',type=str,required=True)
parser.add_argument('--outputdir',type=str,default='img/infer')
#parser.add_argument('--tweets',type=str,default='data/infer/riverside')
parser.add_argument('--tweets',type=str,default='data/infer/infer')
#parser.add_argument('--tweets',type=str,default='data/twitter-early/geoTwitter17-10-21/geoTwitter17-10-21_01:05.gz')
#parser.add_argument('--tweets',type=str,default='data/MX/train.gz')
parser.add_argument('--maxtweets',type=int,default=100)
parser.add_argument('--plot_res',choices=['hi','good','med','lo'],default='med')
parser.add_argument('--plot_true',type=bool,default=True)
parser.add_argument('--plot_mle',type=bool,default=False)
parser.add_argument('--plot_mu',type=bool,default=False)
parser.add_argument('--plot_numbars',type=int,default=5)
parser.add_argument('--overwrite',action='store_true')
parser.add_argument('--notext',action='store_true')
parser.add_argument('--format',type=str,choices=['png','pdf','pgf'],default='png')
parser.add_argument('--marker',type=str,choices=['.','o'],default='o')

args = parser.parse_args()

########################################
print('loading model args')

class MyNamespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

import simplejson as json
args_str=json.dumps(vars(args))
with open(args.modeldir+'/args.json','r') as f:
    args_str=f.readline()

    args_train=MyNamespace(json.loads(args_str))
    args_train.gmm_lrfactor=0.1
    args_train.summary_size='small'
    args_train.summary_newusers=False

    try:
        args_train.hashes_uniq
    except:
        args_train.hashes_uniq=False
        args_train.text_naive=True
        args_train.text_multichar_init_bit=True
        args_train.text_latin_bit=True
        args_train.text_transliterate=True
        args_train.text_hashsize_combiners=True

    try:
        args_train.gmm_distribution
    except:
        args_train.gmm_distribution='fvm'

    try:
        args_train.hashes_true
    except:
        args_train.hashes_true=False
        args_train.full_per_lang=False

########################################
print('loading model weights')

import tensorflow as tf
import os
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import model
import myhash

def restore_session(batchsize):
    print('  restore_session(batchsize=',batchsize,')')
    with tf.Graph().as_default() as graph_bs1:
        ret=MyNamespace({})
        ret.args=copy.deepcopy(args_train)
        ret.args.batchsize=batchsize
        ret.input_tensors={
            'country_' : tf.placeholder(tf.int64, [ret.args.batchsize,1],name='country_'),
            'text_' : tf.placeholder(tf.float32, [ret.args.batchsize,model.tweetlen,ret.args.cnn_vocabsize],name='text_'),
            'gps_' : tf.placeholder(tf.float32, [ret.args.batchsize,2], name='gps_'),
            'loc_' : tf.placeholder(tf.int64, [ret.args.batchsize,1],name='loc_'),
            'timestamp_ms_' : tf.placeholder(tf.float32, [ret.args.batchsize,1], name='timestamp_ms_'),
            'lang_' : tf.placeholder(tf.int32, [ret.args.batchsize,1], 'lang_'),
            'newuser_' : tf.placeholder(tf.float32, [ret.args.batchsize,1],name='newuser_'),
            'hash_' : tf.sparse_placeholder(tf.float32,name='hash_'),
        }

        ret.op_metrics,ret.op_loss_regularized,ret.op_losses,ret.op_outputs = model.inference(ret.args,ret.input_tensors)

        config = tf.ConfigProto(allow_soft_placement = True)
        ret.sess = tf.Session(config=config)
        saver=tf.train.Saver()
        chkpt_file=tf.train.latest_checkpoint(args.modeldir)
        #chkpt_file=args.modeldir+'/model.ckpt-130000'
        saver.restore(ret.sess,chkpt_file)
        return ret

bs1 = restore_session(1)

########################################
print('plotting')

# load tweets
import datetime
import gzip
import math
import textwrap

import matplotlib
import matplotlib.colors as colors
matplotlib.use('Agg')
#matplotlib.use('ps')
import matplotlib.pyplot as plt
import smopy
import numpy as np

#try:
    #file=gzip.open(args.tweets,'r')
#except:
file=open(args.tweets,'r')
#file.readline()

# generate outputdir directory if needed
dataname=os.path.basename(os.path.normpath(args.tweets))
outputdir=args.outputdir+'/'+dataname
try:
    os.makedirs(outputdir)
except:
    pass

# main loop
for tweetnum in range(0,args.maxtweets):

    print(datetime.datetime.now(),'tweet',tweetnum)
    line=file.readline()
    basename=str(tweetnum)+'-'+args.plot_res

    # skip if already computed
    if not args.overwrite:
        mainfilepath=outputdir+'/'+basename+'.'+args.format
        if os.path.exists(mainfilepath):
            continue

    # get tweet data
    tweet=json.loads(line)
    tweet['text']=model.preprocess_text(args_train,tweet['text'])
    print('tweet=',tweet)
    input=model.json2dict(args_train,line)
    print('input=',input)
    input['newuser_']=np.array([1])
    print('  input[gps_]=',input['gps_'])
    lat_=input['gps_'][0]
    lon_=input['gps_'][1]

    # get model data
    feed_dict=model.mk_feed_dict(bs1.args,[input])
    loss,losses,output=bs1.sess.run(
        [bs1.op_loss_regularized, bs1.op_losses, bs1.op_outputs],
        feed_dict=feed_dict
        )
    print('  output[gps]=',np.reshape(output['gps'],[2]))
    print('  loss=',loss)
    print('  losses=',losses)

    def gps2sphere(lat,lon):
        lat_rad=(lat+27)/360*2*math.pi/1.5
        lon_rad=((lon-80)*1.06+80)/360*2*math.pi*2
        x = np.stack([ np.sin(lat_rad)
                     , np.cos(lat_rad) * np.sin(lon_rad)
                     , np.cos(lat_rad) * np.cos(lon_rad)
                     ])
        return x

    pre_mu_gps=np.reshape(output['aglm_mix/pre_mu_gps'],[2,args_train.gmm_components])
    #print('pre_mu_gps=',pre_mu_gps[:,:10])
    mu=np.reshape(output['aglm_mix/mu'],[3,args_train.gmm_components])
    #mu=gps2sphere(pre_mu_gps[0,:],pre_mu_gps[1,:])

    pre_kappa=output['aglm_mix/pre_kappa']
    kappa=output['aglm_mix/kappa']
    pre_kappa=8.0
    kappa=math.exp(pre_kappa)

    mixture=output['aglm_mix/mixture']
    #print('mixture=',mixture)
    #mixture=np.ones(mixture.shape)/args_train.gmm_components
    #mixture_hard=(mixture == mixture.max(axis=1)[:,None]).astype(float)
    #mixture=mixture_hard

    # tweet mangling
    import copy
    all_losses_mangled=[]
    if not args.notext:
        for i in range(0,len(tweet['text'])):
            input_new=copy.deepcopy(input)
            arr1=input_new['text_'][:,:max(0,i-0),:]
            arr2=input_new['text_'][:,min(model.tweetlen,i+1):,:]
            input_new['text_']=np.concatenate(
                [arr1
                ,arr2
                ,np.zeros([
                    1,
                    model.tweetlen-arr1.shape[1]-arr2.shape[1],
                    input_new['text_'].shape[2]
                    ])
                ],axis=1)
            feed_dict=model.mk_feed_dict(bs1.args,[input_new])
            loss_mangled,losses_mangled,output_mangled=bs1.sess.run(
                [bs1.op_loss_regularized, bs1.op_losses, bs1.op_outputs],
                feed_dict=feed_dict
                )
            tweetchar=tweet['text'][i]
            if ord(tweetchar)>127:
                tweetchar='???'
            #print('  %3d'%i,tweet['text'][i],'loss_mangled=',loss_mangled)
            print('  %3d'%i,tweetchar,'loss_mangled=',loss_mangled)
            all_losses_mangled.append(losses_mangled)
        while len(all_losses_mangled) < model.tweetlen:
            all_losses_mangled.append(losses)

    # setup plot
    lat_offset=lat_
    lon_offset=lon_
    lon_km=2000
    import math
    range_lat=math.degrees(min(lon_km,15000)/6371.0)/2
    range_lon=math.degrees(lon_km/6371.0)/2

    box=(lat_offset-range_lat,
         lon_offset-range_lon,
         lat_offset+range_lat,
         lon_offset+range_lon,
         )
    print('box=',box)

    # plot densities
    def plot_density(ax):

        # helper function to measure density
        def log_likelihood(lat,lon):
            x = gps2sphere(lat,lon)
            x_reshape=np.reshape(x,[3,1])

            if args_train.gmm_distribution=='gaussian':
                vecsum=np.sum(np.abs(x_reshape-mu)**2.0,axis=0)
                loglikelihood_per_component = np.exp(-kappa*vecsum)
            elif args_train.gmm_distribution=='efam':
                vecsum=np.sum(np.abs(x_reshape-mu)**output['aglm_mix/pow'],axis=0)
                loglikelihood_per_component = np.exp(-kappa*vecsum)
            else:
                loglikelihood_per_component=np.where(
                        np.greater(kappa,1.0),
                        -kappa,
                        -kappa*kappa
                        )+(kappa * np.sum(x_reshape*mu,axis=0))
            likelihood_mixed = np.sum(np.exp(loglikelihood_per_component)*mixture,axis=1)
            return likelihood_mixed
            #return np.log(likelihood_mixed+1e-10)

        # set resolution
        if args.plot_res=='hi':
            steps=500
            steps=500
        elif args.plot_res=='good':
            steps=200
            steps=200
        elif args.plot_res=='med':
            steps=100
            steps=100
        elif args.plot_res=='lo':
            steps=20
            steps=20
        #lons,lats=np.meshgrid(np.linspace(px0,px2,(steps+1)*zoom),np.linspace(px3,px1,(steps+1)*zoom))
        lons,lats=np.meshgrid(np.linspace(box[0],box[2],steps),np.linspace(box[3],box[1],steps))

        # generate plot data
        z=np.zeros([steps,steps])
        for x in range(0,steps):
            for y in range(0,steps):
                z[x][y]=log_likelihood(lats[y,x],lons[y,x])

        totalweights=np.sum(z)
        #z=z/totalweights
        z=np.log(z/totalweights+1e-10)/math.log(10)
        levels=[-4,-3,-2,-1,-0]
        #levels=[-5,-4,-3,-2,-1]
        #levels=[-6,-5,-4,-3,-2]

        # plot contours
        #cmap = colors.LinearSegmentedColormap.from_list("", ['#ffffff','#99ff99','#66cc66','#6699cc','#6666ff']) #,'yellow'])
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#ffffff','#ff9999','#cc6666','#9966cc','#9900ff'])
        norm = plt.Normalize(min(levels),max(levels))
        nx,ny=np.meshgrid(np.linspace(px0,px2,steps),np.linspace(px3,px1,steps))
        cs = ax.contourf(
            nx,
            ny,
            np.transpose(z),
            #levels=levels,
            #latlon=True,
            antialiased=True,
            cmap=cmap,
            norm=norm,
            alpha=0.5,
            )
        #cbar = m.colorbar(cs,location='right',pad="5%")
        #cbar.ax.get_yaxis().set_ticks([])
        #for j, lab in enumerate(['0.09%','0.9%','9%','90%']):
            #cbar.ax.text(0.6, (2 * j + 1) / 8.0, lab, ha='center', va='center', rotation='vertical')
        #cbar.ax.get_yaxis().labelpad = 10
        ##cbar.set_label('Probability that the tweet is in this region')
        #cbar.set_label('Probability Mass of Plotted Region')

        return
        # plot points
        if args.plot_true:
            x, y = m(lon_, lat_)
            m.scatter(x, y, marker=args.marker,color='m',zorder=4)

        if args.plot_mle:
            x, y = m(output['gps'][0,1], output['gps'][0,0])
            m.scatter(x, y, marker=args.marker,color='blue',zorder=5)

        if args.plot_mu:
            x, y = m(pre_mu_gps[1,:], pre_mu_gps[0,:])
            m.scatter(x, y, marker='.',color='blue',zorder=3)

        # draw oceans
        # https://stackoverflow.com/questions/13796315/plot-only-on-continent-in-matplotlib
        from matplotlib.patches import Path, PathPatch
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
        map_edges = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])
        polys = [p.boundary for p in m.landpolygons]
        polys = [map_edges]+polys[:]
        codes = [
            [Path.MOVETO] + [Path.LINETO for p in p[1:]]
                for p in polys
            ]
        polys_lin = [v for p in polys for v in p]
        codes_lin = [c for cs in codes for c in cs]
        path = Path(polys_lin, codes_lin)
        patch = PathPatch(path,facecolor='white', lw=0)
        ax.add_patch(patch)

    m1=smopy.Map(
        box,
        tileserver="http://tile.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@8x.png",
        tilesize=2048,
        maxtiles=16,
        )
    ax = m1.show_mpl(figsize=(16, 16))
    px0,px1 = m1.to_pixels((box[0],box[1]))
    px2,px3 = m1.to_pixels((box[2],box[3]))
    if True:
        ratio=3.0/4.0
        #ratio=1.0
        yrange=(px2-px0)*ratio
        yave=(px1+px3)/2.0
        px1=yave+yrange/2.0
        px3=yave-yrange/2.0
        box0,box1=m1.from_pixels(px0,px1) #yave-yrange/2.0)
        box2,box3=m1.from_pixels(px2,px3) #yave+yrange/2.0)
        box=[box0,box1,box2,box3]
    ax.set_xlim((px0,px2))
    ax.set_ylim((px1,px3))
    plot_density(ax)

    # save
    #plt.tight_layout()
    plt.savefig(outputdir+'/'+basename+'_smopy.'+args.format, bbox_inches='tight',pad_inches=0)


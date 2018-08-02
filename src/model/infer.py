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

parser.add_argument('--modeldir',type=str)
parser.add_argument('--outputdir',type=str,default='img-infer')
#parser.add_argument('--tweets',type=str,default='data/infer/riverside')
parser.add_argument('--tweets',type=str,default='data/infer/infer')
#parser.add_argument('--tweets',type=str,default='data/twitter-early/geoTwitter17-10-21/geoTwitter17-10-21_01:05.gz')
#parser.add_argument('--tweets',type=str,default='data/MX/train.gz')
parser.add_argument('--maxtweets',type=int,default=100)
parser.add_argument('--plot_res',choices=['hi','med','lo'],default='med')
parser.add_argument('--plot_mu',type=bool,default=False)
parser.add_argument('--plot_numbars',type=int,default=5)

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
        #chkpt_file=tf.train.latest_checkpoint(args.modeldir)
        chkpt_file=args.modeldir+'/model.ckpt-130000'
        saver.restore(ret.sess,chkpt_file)
        return ret

bs1 = restore_session(1)

########################################
print('preparing fonts')
from fontTools.ttLib import TTFont
import matplotlib.font_manager as fm

font_paths=[
    'fonts/NotoSansMonoCJKjp-Regular.otf',
    'fonts/NotoSansThai-Regular.ttf',
    'fonts/NotoKufiArabic-Regular.ttf',
    'fonts/NotoEmoji-Regular.ttf',
    'fonts/NotoMono-Regular.ttf',
    ]

fonts=map(TTFont,font_paths)
font_properties=map(lambda x: fm.FontProperties(fname=x),font_paths)

font_notomono=fm.FontProperties(fname='fonts/NotoMono-Regular.ttf')

def get_font_property(unicode_char):
    for (font,prop) in zip(fonts,font_properties):
        for cmap in font['cmap'].tables:
            if cmap.isUnicode():
                if ord(unicode_char) in cmap.cmap:
                    return prop
    return font_properties[0]

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
from mpl_toolkits.basemap import Basemap
import numpy as np

#try:
    #file=gzip.open(args.tweets,'r')
#except:
file=open(args.tweets,'r')
#file.readline()

for tweetnum in range(0,args.maxtweets):

    print(datetime.datetime.now(),'tweet',tweetnum)
    line=file.readline()

    # get tweet data
    tweet=json.loads(line)
    tweet['text']=model.preprocess_text(args_train,tweet['text'])
    input=model.json2dict(args_train,line)
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
        lat_rad=lat/360*2*math.pi
        lon_rad=lon/360*2*math.pi
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
    #pre_kappa=0.001
    #kappa=math.exp(pre_kappa)

    mixture=output['aglm_mix/mixture']
    #print('mixture=',mixture)
    #mixture=np.ones(mixture.shape)/args_train.gmm_components
    #mixture_hard=(mixture == mixture.max(axis=1)[:,None]).astype(float)
    #mixture=mixture_hard

    # tweet mangling
    import copy
    all_losses_mangled=[]
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
    #sys.exit(1)

    # tweet fig
    def mk_tweet_ax(lossname,ax,maxwidth=35):
        #z=np.array([d[lossname]-losses[lossname] for d in all_losses_mangled])
        z=np.array([d[lossname] for d in all_losses_mangled])
        z=np.reshape(z,[model.tweetlen/maxwidth,maxwidth])
        #zabsmax=np.amax(np.abs(z))
        zabsmax=np.amax(np.abs(z-losses[lossname]))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","white","green"])
        norm=plt.Normalize(losses[lossname]-zabsmax,losses[lossname]+zabsmax)
        plt.imshow(z,cmap=cmap,norm=norm)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        for i in range(0,len(tweet['text'])):
            prop=get_font_property(tweet['text'][i])
            ax.text(i%maxwidth-0.25,i/maxwidth+0.25,tweet['text'][i],fontproperties=prop,fontsize=14)
        ax.set_ylim(len(tweet['text'])/maxwidth+0.5,-0.5)
        #plt.colorbar(orientation='horizontal')

    def mk_tweet_fig(lossname,maxwidth=35):
        fig,ax = plt.subplots(figsize=(12,10))
        mk_tweet_ax(lossname,ax,maxwidth)
        lossname_modified=lossname.split('/')[-1]
        plt.tight_layout()
        plt.savefig(args.outputdir+'/'+str(tweetnum)+'-'+lossname_modified+'.png', bbox_inches='tight')
        plt.close()

    mk_tweet_fig('optimization/op_loss_regularized')
    mk_tweet_fig('lang_xentropy')
    mk_tweet_fig('country_xentropy')
    mk_tweet_fig('pos_loss_mix')

    # setup plot
    fig = plt.figure(figsize=(10,7.5))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(3, 2,height_ratios=[2,4,1])
    #_,((ax10,ax11),(ax20,ax21))=plt.subplots(2,2, gridspec_kw = {'height_ratios':[4, 1],'top':0.8},figsize=(12,6))

    ax10=plt.subplot(gs[1,0])
    ax11=plt.subplot(gs[1,1])
    ax20=plt.subplot(gs[2,0])
    ax21=plt.subplot(gs[2,1])

    ax10.get_xaxis().set_visible(False)
    ax10.get_yaxis().set_visible(False)
    ax11.get_xaxis().set_visible(False)
    ax11.get_yaxis().set_visible(False)

    # add tweet title
    ax0 = plt.subplot(gs[0,:])
    #mk_tweet_ax('optimization/op_loss_regularized',ax0)
    mk_tweet_ax('country_xentropy',ax0)
    #wrappedtweet='\n'.join(textwrap.wrap(tweet['text'],60))
    #plt.suptitle(wrappedtweet)
    #plt.subplots_adjust(bottom=0.1)

    # plot densities
    def plot_density(m):

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
            nx=500
            ny=250
        elif args.plot_res=='med':
            nx=100
            ny=50
        elif args.plot_res=='lo':
            nx=20
            ny=10
        lons, lats = m.makegrid(nx, ny)

        # generate plot data
        z=np.zeros([nx,ny])
        for x in range(0,nx):
            for y in range(0,ny):
                z[x][y]=log_likelihood(lats[y,x],lons[y,x])

        totalweights=np.sum(z)
        #z=z/totalweights
        z=np.log(z/totalweights+1e-10)/math.log(10)
        levels=[-5,-4,-3,-2,-1]

        zz=np.zeros([nx,ny])
        numlevels=100
        for x in range(0,nx):
            for y in range(0,ny):

                for i in range(0,numlevels):
                    if z[x][y] >= totalweights*10**(-i):
                        zz[x][y]=float(i)
        #z=zz
        #z=np.log(z/totalweights)

        # plot contours
        cmap = colors.LinearSegmentedColormap.from_list("", ['white','green','blue']) #,'yellow'])
        norm = plt.Normalize(min(levels),max(levels))
        cs = m.contourf(
            lons,
            lats,
            np.transpose(z),
            levels=levels,
            latlon=True,
            antialiased=True,
            cmap=cmap,
            norm=norm
            )
        cbar = m.colorbar(cs,location='right',pad="5%")
        cbar.set_label('log error rate')

        # plot points
        x, y = m(lon_, lat_)
        m.scatter(x, y, marker='o',color='m',zorder=4)

        x, y = m(output['gps'][0,1], output['gps'][0,0])
        m.scatter(x, y, marker='o',color='red',zorder=5)

        if args.plot_mu:
            x, y = m(pre_mu_gps[1,:], pre_mu_gps[0,:])
            m.scatter(x, y, marker='.',color='blue',zorder=3)

        # draw oceans
        # https://stackoverflow.com/questions/13796315/plot-only-on-continent-in-matplotlib
        from matplotlib.patches import Path, PathPatch
        ax=m.ax
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

    m1=Basemap(
        projection='merc',
        llcrnrlat=-80,
        urcrnrlat=80,
        llcrnrlon=-180,
        urcrnrlon=180,
        #lat_ts=20,
        resolution='c',
        fix_aspect=False,
        ax=ax10,
        )
    plot_density(m1)
    m1.drawcoastlines(linewidth=0.25)#,zorder=-1)
    m1.drawcountries(linewidth=0.1)#,zorder=-1)
    #m.drawcoastlines()
    #m.fillcontinents(color='gray',lake_color='white')

    m2=Basemap(
        projection='merc',
        llcrnrlat=input['gps_'][0]-16,
        urcrnrlat=input['gps_'][0]+16,
        llcrnrlon=input['gps_'][1]-16,
        urcrnrlon=input['gps_'][1]+16,
        fix_aspect=False,
        #lat_0=input['gps_'][0],
        #lon_0=input['gps_'][1],
        #lat_ts=20,
        resolution='l',
        ax=ax11
        )
    plot_density(m2)
    m2.drawcoastlines(linewidth=0.5)#,zorder=-1)
    m2.drawcountries(linewidth=0.5)#,zorder=-1)
    m2.drawstates(linewidth=0.1)

    # add country info
    countries=output['country_softmax'][0,:]
    sorted_countries=np.flip(np.argsort(countries),axis=0)
    labels=[]
    vals=[]
    for i in range(0,args.plot_numbars):
        label=''
        if sorted_countries[i]==input['country_'][0]:
            label+='*'
        label+=myhash.int2country(sorted_countries[i])
        labels.append(label)
        val=output['country_softmax'][0,:][sorted_countries[i]]
        vals.append(val)
    y_pos = np.arange(args.plot_numbars)
    ax20.barh(y_pos, vals, align='center', color='green', ecolor='black')
    ax20.set_yticks(y_pos)
    ax20.set_yticklabels(labels)
    ax20.invert_yaxis()
    ax20.spines['right'].set_visible(False)
    ax20.spines['top'].set_visible(False)
    ax20.set_xlim([0,1])

    for label in ax20.get_yticklabels():
        label.set_fontproperties(font_notomono)

    # add language info
    langs=output['lang_softmax'][0,:]
    sorted_langs=np.flip(np.argsort(langs),axis=0)
    labels=[]
    vals=[]
    for i in range(0,args.plot_numbars):
        label=''
        if sorted_langs[i]==input['lang_']:
            label+='*'
        label+=myhash.int2lang(sorted_langs[i])
        labels.append(label)
        val=output['lang_softmax'][0,:][sorted_langs[i]]
        vals.append(val)
    y_pos = np.arange(args.plot_numbars)
    ax21.barh(y_pos, vals, align='center', color='green', ecolor='black')
    ax21.set_yticks(y_pos)
    ax21.set_yticklabels(labels)
    ax21.invert_yaxis()
    ax21.spines['right'].set_visible(False)
    ax21.spines['top'].set_visible(False)
    ax21.set_xlim([0,1])

    for label in ax20.get_yticklabels():
        label.set_fontproperties(font_notomono)

    # save
    #plt.tight_layout()
    plt.savefig(args.outputdir+'/'+str(tweetnum)+'.png', bbox_inches='tight')
    plt.close()

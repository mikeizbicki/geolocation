#s!/usr/bin/env python
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
parser.add_argument('--tweets',type=str,default='data/infer/infer')
#parser.add_argument('--tweets',type=str,default='data/twitter-early/geoTwitter17-10-21/geoTwitter17-10-21_01:05.gz')
#parser.add_argument('--tweets',type=str,default='data/MX/train.gz')
parser.add_argument('--plot_res',choices=['hi','good','med','lo'],default='med')
parser.add_argument('--plot_true',type=bool,default=True)
parser.add_argument('--plot_mle',type=bool,default=False)
parser.add_argument('--plot_mu',type=bool,default=False)
parser.add_argument('--plot_numbars',type=int,default=5)
parser.add_argument('--overwrite',action='store_true')
parser.add_argument('--notext',action='store_true')
parser.add_argument('--noplots',action='store_true')
parser.add_argument('--format',type=str,choices=['png','eps','pdf'],default='png')
parser.add_argument('--marker',type=str,choices=['.','o'],default='o')
parser.add_argument('--language',type=str,default=None)

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
    with tf.Graph().as_default():
        ret=MyNamespace({})
        ret.args=copy.deepcopy(args_train)
        ret.args.batchsize=None #batchsize
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
        ret.args.batchsize=batchsize

        ret.op_metrics,ret.op_loss_regularized,ret.op_losses,ret.op_losses_unreduced,ret.op_outputs = model.inference(ret.args,ret.input_tensors,disable_summaries=True)

        config = tf.ConfigProto(allow_soft_placement = True)
        ret.sess = tf.Session(config=config)
        saver=tf.train.Saver()
        chkpt_file=tf.train.latest_checkpoint(args.modeldir)
        #chkpt_file=args.modeldir+'/model.ckpt-130000'
        saver.restore(ret.sess,chkpt_file)
        return ret

bs1 = restore_session(1)
bs280 = restore_session(None)

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

# load the tweets file
try:
    file=gzip.open(args.tweets,'r')
    file.readline()
    file.close()
    file=gzip.open(args.tweets,'r')
except:
    file=open(args.tweets,'r')

# generate outputdir directory if needed
dataname=os.path.basename(os.path.normpath(args.tweets))
outputdir=args.outputdir+'/'+dataname
try:
    os.makedirs(outputdir)
except:
    pass

# main loop
import itertools
for tweetnum in itertools.count():

    if tweetnum%100==0:
        print(datetime.datetime.now(),'tweet',tweetnum)
    line=file.readline()
    basename=str(tweetnum)+'-'+args.plot_res

    # quit on EOF
    if line=='':
        break

    # skip if already computed
    if not args.overwrite:
        mainfilepath=outputdir+'/'+basename+'.'+args.format
        #print('mainfilepath=',mainfilepath)
        if os.path.exists(mainfilepath):
            continue

    # get tweet data
    try:
        tweet=json.loads(line)
        tweet['text']=model.preprocess_text(args_train,tweet['text'])
        input=model.json2dict(args_train,line)
        input['newuser_']=np.array([1])
        lat_=input['gps_'][0]
        lon_=input['gps_'][1]

        if args.language is not None:
            if not args.language==tweet['lang']:
                continue

        # FIXME:
        #line_data=json.loads(line)
        #tweet=line_data['tweet']
        #tweet['text']=model.preprocess_text(args_train,tweet['text'])
        #line2=json.dumps(tweet)
        #input=model.json2dict(args_train,line2)
        #input['newuser_']=np.array([1])
        #lat_=input['gps_'][0]
        #lon_=input['gps_'][1]
        #all_losses_mangled=line_data['all_losses_mangled']
        #losses=line_data['losses']
        #loss=line_data['loss']
    except Exception as e:
        #import traceback
        #traceback.print_exc()
        #asd
        continue

    # get model data
    feed_dict=model.mk_feed_dict(bs1.args,[input])
    try:
        losses
        output,=bs1.sess.run(
            [bs1.op_outputs],
            feed_dict=feed_dict
            )
    except:
        loss,losses,output=bs1.sess.run(
            [bs1.op_loss_regularized, bs1.op_losses, bs1.op_outputs],
            feed_dict=feed_dict
            )

    if not args.noplots:
        print(('  tweet[text]='+tweet['text']).encode('utf-8').strip())
        print('  input[gps_]=',input['gps_'])
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
    mu=np.reshape(output['aglm_mix/mu'],[3,args_train.gmm_components])

    pre_kappa=output['aglm_mix/pre_kappa']
    kappa=output['aglm_mix/kappa']
    pre_kappa=10.0
    kappa=math.exp(pre_kappa)

    mixture=output['aglm_mix/mixture']

    # tweet mangling
    import copy
    if not args.notext:
        batch=[]
        try:
            all_losses_mangled
        except:
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
                batch.append(input_new)

            if batch==[]:
                tweet_empty=copy.deepcopy(input)
                tweet_empty['text_']=np.zeros(input_new['text_'].shape)
                batch=[tweet_empty]

            feed_dict=model.mk_feed_dict(bs1.args,batch)
            all_losses_mangled,output_mangled=bs280.sess.run(
                [bs280.op_losses_unreduced, bs280.op_outputs],
                feed_dict=feed_dict
                )
            all_losses_mangled=[ {k:v[i] for k,v in all_losses_mangled.items() } for i in range(0,len(batch))]

    # data file
    if args.noplots:
        try:
            output_data_file
        except:
            output_data_file=gzip.open(outputdir+'/datafile.'+str(datetime.datetime.now())+'.data.gz','w')

        outputdata={
            'tweet' : tweet,
            'input[gps_]' : input['gps_'].tolist(),
            'output[gps]' : np.reshape(output['gps'],[2]).tolist(),
            'loss' : float(loss),
            'losses' : { k:float(v) for k,v in losses.items() },
            'all_losses_mangled' : [ { k:float(v) for k,v in losses.items() } for losses in all_losses_mangled ],
        }
        output_data_file.write(json.dumps(outputdata)+'\n')
        continue

    # tweet fig
    def mk_tweet_ax(lossname,ax,maxwidth=35):
        z=np.array([d[lossname]-losses[lossname] for d in all_losses_mangled])
        #z=np.array([d[lossname] for d in all_losses_mangled])
        #zabsmax=np.amax(np.abs(z))
        zabsmax=np.amax(np.abs(z-losses[lossname]))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","white","green"])
        norm=plt.Normalize(-zabsmax,zabsmax)
        #norm=plt.Normalize(losses[lossname]-zabsmax,losses[lossname]+zabsmax)
        z=np.pad(z,[0,model.tweetlen-len(all_losses_mangled)],'constant',constant_values=0.0)
        z=np.reshape(z,[model.tweetlen/maxwidth,maxwidth])
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
        plt.savefig(outputdir+'/'+basename+'-'+lossname_modified+'.'+args.format, bbox_inches='tight')
        plt.close()

    if not args.notext:
        mk_tweet_fig('optimization/op_loss_regularized')
        #mk_tweet_fig('lang_xentropy')
        #mk_tweet_fig('country_xentropy')
        #mk_tweet_fig('pos_loss_mix')

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
    if not args.notext:
        #mk_tweet_ax('optimization/op_loss_regularized',ax0)
        mk_tweet_ax('country_xentropy',ax0)
        #mk_tweet_ax('pos_loss_mix',ax0)
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
        elif args.plot_res=='good':
            nx=200
            ny=100
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
        levels=[-6,-5,-4,-3,-2]

        zz=np.zeros([nx,ny])
        numlevels=100
        for x in range(0,nx):
            for y in range(0,ny):

                for i in range(0,numlevels):
                    if z[x][y] >= totalweights*10**(-i):
                        zz[x][y]=float(i)

        # helper function
        # see: http://scipy-cookbook.readthedocs.io/items/Matplotlib_ColormapTransformations.html
        def cmap_map(function, cmap):
            """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
            This routine will break any discontinuous points in a colormap.
            """
            cdict = cmap._segmentdata
            step_dict = {}
            # Firt get the list of points where the segments start or end
            for key in ('red', 'green', 'blue'):
                step_dict[key] = list(map(lambda x: x[0], cdict[key]))
            step_list = sum(step_dict.values(), [])
            step_list = np.array(list(set(step_list)))
            # Then compute the LUT, and apply the function to the LUT
            reduced_cmap = lambda step : np.array(cmap(step)[0:3])
            old_LUT = np.array(list(map(reduced_cmap, step_list)))
            new_LUT = np.array(list(map(function, old_LUT)))
            # Now try to make a minimal segment definition of the new LUT
            cdict = {}
            for i, key in enumerate(['red','green','blue']):
                this_cdict = {}
                for j, step in enumerate(step_list):
                    if step in step_dict[key]:
                        this_cdict[step] = new_LUT[j, i]
                    elif new_LUT[j,i] != old_LUT[j, i]:
                        this_cdict[step] = new_LUT[j, i]
                colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
                colorvector.sort()
                cdict[key] = colorvector

            return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

        # plot contours
        #cmap = colors.LinearSegmentedColormap.from_list("", ['white','darkgreen','blue']) #,'yellow'])
        cmap = colors.LinearSegmentedColormap.from_list("", ['#ffffff','#99ff99','#66cc66','#6699cc','#6666ff']) #,'yellow'])
        #cdict={
            #'red':((0.0,0.0,0.0)),
            #'green':((0.5
        #}
        #cmap = colors.LinearSegmentedColormap('colormap',cdict,1024)

        #cmap = cmap_map(lambda x: 4*x/2, cmap)
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
        cbar.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate(['0.09%','0.9%','9%','90%']):
            cbar.ax.text(0.6, (2 * j + 1) / 8.0, lab, ha='center', va='center', rotation='vertical')
        cbar.ax.get_yaxis().labelpad = 10
        #cbar.set_label('Probability that the tweet is in this region')
        cbar.set_label('Probability Mass of Plotted Region')

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
        #llcrnrlat=input['gps_'][0]-7,
        #urcrnrlat=input['gps_'][0]+13,
        #llcrnrlon=input['gps_'][1]-10,
        #urcrnrlon=input['gps_'][1]+10,
        llcrnrlat=input['gps_'][0]-3,
        urcrnrlat=input['gps_'][0]+6,
        llcrnrlon=input['gps_'][1]-5,
        urcrnrlon=input['gps_'][1]+5,
        fix_aspect=False,
        #lat_0=input['gps_'][0],
        #lon_0=input['gps_'][1],
        #lat_ts=20,
        resolution='i',
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
        print('  label=',label,'; val=',val)
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
    plt.savefig(outputdir+'/'+basename+'.'+args.format, bbox_inches='tight')

    extent = ax10.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    extent.x1+=0.9
    extent.y0-=0.1
    plt.savefig(outputdir+'/'+basename+'-world.'+args.format, bbox_inches=extent)

    extent = ax11.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    extent.x1+=0.6
    extent.y0-=0.1
    plt.savefig(outputdir+'/'+basename+'-zoom.'+args.format, bbox_inches=extent)
    #plt.savefig(outputdir+'/'+basename+'-zoom.png', bbox_inches=extent)
    #plt.savefig(outputdir+'/'+basename+'-zoom.eps', bbox_inches=extent)
    #plt.savefig(outputdir+'/'+basename+'-zoom.pgf', bbox_inches=extent)

    extent = ax20.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    extent.x0-=0.1
    extent.y0-=0.1
    plt.savefig(outputdir+'/'+basename+'-bar-country.'+args.format, bbox_inches=extent)

    extent = ax21.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    extent.x0-=0.3
    extent.y0-=0.2
    plt.savefig(outputdir+'/'+basename+'-bar-lang.'+args.format, bbox_inches=extent)

    plt.close()

    # specialized country plot
    plt.rcParams.update({'pgf.rcfonts':False})
    fig,ax = plt.subplots(figsize=(3.5,0.7))
    countries=output['country_softmax'][0,:]
    sorted_countries=np.flip(np.argsort(countries),axis=0)
    labels=[]
    vals=[]
    plt.xticks(fontsize=8)#,fontname = "FreeSerif")
    plt.yticks(fontsize=8)#,fontname = "FreeSerif")
    for i in range(0,args.plot_numbars):
        label=''
        #if sorted_countries[i]==input['country_'][0]:
            #label+='*'
        label+=myhash.country_code2name(myhash.int2country(sorted_countries[i]))
        labels.append(''+label)
        val=output['country_softmax'][0,:][sorted_countries[i]]
        vals.append(val)
        print('  label=',label,'; val=',val)
    y_pos = np.arange(args.plot_numbars)
    ax.barh(y_pos, vals, align='center', color='green', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlim([0,1])
    #ax.set_ylabel('Country')
    #plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.35,left=0.3)
    plt.savefig(outputdir+'/'+basename+'-bar-country2.'+args.format)


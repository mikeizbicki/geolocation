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

parser=argparse.ArgumentParser('plot a map')

parser.add_argument('--input_format',type=str,choices=['jpg','tfrecord'],default='jpg')
parser.add_argument('--img',type=str,default=None)
parser.add_argument('--tweet',type=str,default=None)
parser.add_argument('--s2file',type=str,default=None)
parser.add_argument('--initial_weights',type=str,default=None)
parser.add_argument('--outputdir',type=str,default='img/maps')
parser.add_argument('--outputname',type=str,default='')
parser.add_argument('--marks',type=float,nargs='*',default=[])
parser.add_argument('--ratio',choices=['wide2','wide','full'],default='full')
parser.add_argument('--custom',choices=['gaussian','vmf'],default=None)
parser.add_argument('--voronoi',action='store_true')
parser.add_argument('--voronoi_random',action='store_true')
parser.add_argument('--output_plain',action='store_true')

parser.add_argument('--plot_mu',action='store_true')
parser.add_argument('--plot_gps',action='store_true')
parser.add_argument('--plot_res',choices=['lo','med','med2','good','hi'],default='med')

parser.add_argument('--plot_sample_points',action='store_true')
parser.add_argument('--plot_grid',action='store_true')
parser.add_argument('--force_center',type=float,nargs=2,default=None)
parser.add_argument('--lat_km',type=int,default=300)
parser.add_argument('--lon_km',type=int,default=400)
parser.add_argument('--autoscale_y',default=True)
parser.add_argument('--lat_offset',type=float,default=0)
parser.add_argument('--lon_offset',type=float,default=0)

parser.add_argument('--num_crops',type=int,default=1)
parser.add_argument('--batchsize',type=int,default=256)

parser.add_argument('--s2warmstart_mu',action='store_true')
parser.add_argument('--s2warmstart_kappa',action='store_true')
parser.add_argument('--s2warmstart_kappa_s',type=float,default=0.95)
parser.add_argument('--lores_gmm_prekappa0',type=float,default=10.0)
parser.add_argument('--force_even_mixture',action='store_true')

args = parser.parse_args()

try:
    args.outputs
except:
    args.outputs=[]

# output filename prefix
if args.img:
    import os
    basename=os.path.splitext(os.path.basename(args.img))[0]
    prefixname=args.outputdir+'/'+basename+'_'+args.outputname

    # load gps of file
    import data
    gps_file=data.imgpath2gps(args.img)
    gps=gps_file
    print('  gps_file=',gps_file)

else:
    prefixname=args.outputdir+'/'+'empty'+'_'+args.outputname
    if not args.force_center:
        args.force_center=[7,0]
        args.lat_km=15000
        args.lon_km=36000
    args.outputs=[]
    gps=None

########################################
if args.initial_weights:
    print('loading model')

    ####################
    print('  libraries')
    import tensorflow as tf
    import tensornets as nets
    import datetime

    ####################
    print('  loading arguments from %s/args.json'%args.initial_weights)
    import simplejson as json
    with open(args.initial_weights+'/args.json','r') as f:
        args_str=f.readline()

    args=argparse.Namespace(**json.loads(args_str))
    args=parser.parse_args(namespace=args)
    args.gmm_minimizedist=False
    args.gmm_gradient_method='all'
    args.image_size=224

    ####################
    print('  model first pass')
    import model

    # load jpg model
    if args.input_format=='jpg':

        # load img
        if args.img:
            gps=gps_file
            files=[args.img]
            iter=model.mkDataset_jpg(args,files,is_training=False)
            file_,(gps_,country_),image_=iter.get_next()
            image_=tf.reshape(image_,[-1,10,args.image_size,args.image_size,3])
            if args.num_crops==1:
                image_=image_[:,0,:,:,:]
                image_=tf.reshape(image_,[-1,1,args.image_size,args.image_size,3])

            with tf.Session() as sess:
                sess.run(iter.initializer)
                image=sess.run(image_)
                image=tf.reshape(image,[1,1,args.image_size,args.image_size,3])

            features=[]
            s2_=None

        # load tweet
        elif args.tweet:
            import sys
            sys.path.insert(0, '/rhome/mizbicki/bigdata/geolocating/src/model')
            import model
            import simplejson as json
            with open(args.tweet) as f:
                line=f.readline()
                tweet=json.loads(line)
                tweet['text']=model.preprocess_text(args_train,tweet['text'])
                input=model.json2dict(args_train,line)
                print('tweet=',tweet)
                asd

        # raise error
        else:
            raise ValueError('either --img or --tweet must be specified')

    # load tfrecord model
    elif args.input_format=='tfrecord':
        files=['/rhome/mizbicki/bigdata/geolocating/data/im2gps_tfrecord/out_tfrecord_1_0']
        iter=model.mkDataset_tfrecord_map(args,files,is_training=False)
        gps_,country_,features_,s2_=iter.get_next()

        with tf.Session() as sess:
            sess.run(iter.initializer)

            while True:
                try:
                    gps,features=sess.run([gps_,features_])
                except:
                    print('gps_file=',gps_file,'gpses=',sorted(map(lambda x: x[0],gpses)))
                    crash_me
                if abs(gps[0]-gps_file[0])+abs(gps[1]-gps_file[1]) < 1e-3:
                    break
            gps=list(gps)
            image=features
            s2_=None

########################################
print('prepare plotting')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import smopy
import math

# create map
range_lat=math.degrees(min(args.lon_km,15000)/6371.0)/2
range_lon=math.degrees(args.lon_km/6371.0)/2

if args.force_center:
    box=(args.force_center[0]-range_lat+args.lat_offset,
         args.force_center[1]-range_lon+args.lon_offset,
         args.force_center[0]+range_lat+args.lat_offset,
         args.force_center[1]+range_lon+args.lon_offset,
         )
else:
    box=(gps[0]-range_lat+args.lat_offset,
         gps[1]-range_lon+args.lon_offset,
         gps[0]+range_lat+args.lat_offset,
         gps[1]+range_lon+args.lon_offset,
         )

print('box=',box)

map = smopy.Map(
    box,
    tileserver="http://tile.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@8x.png",
    #tileserver="https://tiles.wmflabs.org/bw-mapnik/{z}/{x}/{y}.png",
    #tileserver="http://a.tile.stamen.com/toner/{z}/{x}/{y}.png",
    #tileserver="http://c.tile.stamen.com/watercolor/{z}/{x}/{y}.png",
    #margin=None,
    #tilesize=512,
    # tilesize=1024,
    tilesize=2048,
    maxtiles=16,
    )
ax = map.show_mpl(figsize=(16, 16))

# plot gps location
if gps and args.plot_gps:
    x, y = map.to_pixels(gps)
    ax.plot(
        x,
        y,
        color='black',
        marker='o',
        ms=15
        )
    ax.annotate(
        'image location',
        xy=(x,y),
        xytext=(x+200, y-150),
        fontsize=30,
        fontweight='bold',
        color='black',
        arrowprops=dict(facecolor='black', shrink=0.15),
        )
    print('x,y=',(x,y))

# plot marks
for lat,lon in zip(args.marks[0::2], args.marks[1::2]):
    print('lat,lon=',(lat,lon))
    x, y = map.to_pixels([lat,lon])
    print('x,y=',(x,y))
    ax.plot(x,y, 'og', ms=15)

# crop bounding box
px0,px1 = map.to_pixels((box[0],box[1]))
px2,px3 = map.to_pixels((box[2],box[3]))
if args.autoscale_y:
    if args.ratio=='full':
        ratio=3.0/4.0
    elif args.ratio=='wide':
        ratio=9.0/16.0
    elif args.ratio=='wide2':
        ratio=4.0/3.0
    yrange=(px2-px0)*ratio
    yave=(px1+px3)/2.0
    px1=yave+yrange/2.0
    px3=yave-yrange/2.0
    box0,box1=map.from_pixels(px0,px1) #yave-yrange/2.0)
    box2,box3=map.from_pixels(px2,px3) #yave+yrange/2.0)
    box=[box0,box1,box2,box3]

ax.set_xlim((px0,px2))
ax.set_ylim((px1,px3))

if args.output_plain:
    plt.savefig(
        prefixname+'_plain.pdf',
        bbox_inches = 'tight',
        pad_inches = 0
        )

# set resolution
if args.plot_res=='hi':
    steps=1500
elif args.plot_res=='good':
    steps=200
elif args.plot_res=='med2':
    steps=100
elif args.plot_res=='med':
    steps=50
elif args.plot_res=='lo':
    steps=20

steps=100
# create grid of gps coordinates
#grid_px=[ (px0+stepx*(px2-px0)/float(steps), px3+stepy*(px1-px3)/float(steps))
    #for stepx in range(0,steps+1)
    #for stepy in range(0,steps+1) ]
#grid_gps_flat=[ map.from_pixels(px,py) for px,py in grid_px ]
#steps=20
grid_gps_hi=[[ map.from_pixels(px0+stepx*(px2-px0)/float(steps), min(px1,px3)+stepy*abs(px1-px3)/float(steps))
    for stepy in range(0,steps+1) ]
    for stepx in range(0,steps+1) ]
grid_gps_flat = [ a for b in grid_gps_hi for a in b ]

#steps_lo=50
#grid_gps_lo=[[ map.from_pixels(px0+stepx*(px2-px0)/float(steps_lo), min(px1,px3)+stepy*abs(px1-px3)/float(steps_lo))
    #for stepy in range(0,steps_lo+1) ]
    #for stepx in range(0,steps_lo+1) ]
#grid_gps_flat_lo = [ a for b in grid_gps_lo for a in b ]
#grid_gps_flat_lo = []
grid_gps_flat_lo = grid_gps_flat

####################
if args.s2file:
    print('  getting mesh gps coords')
    with open(args.s2file,'rb') as f:
        import pickle
        import s2sphere
        s2cells=pickle.load(f)
        gps_coords=[]
        for cellid in s2cells:
            cell=s2sphere.Cell(cellid)
            latlng=cellid.to_lat_lng()
            lat=latlng.lat().degrees
            lng=latlng.lng().degrees
            x,y = map.to_pixels(lat,lng)
            if (((x>=px0 and x<=px2) or (x>=px2 and x<= px0)) and
                ((y>=px1 and y<=px3) or (y>=px3 and y<= px1))):
                gps_coords.append([lat,lng])
        print('    len=',len(gps_coords))
else:
    gps_coords=[]

########################################
#plot gps grid
if args.plot_sample_points:
    for coord in grid_gps_flat_lo:
        x,y=map.to_pixels(coord)
        ax.plot(x,y,'or', ms=1)

    for coord in gps_coords:
        x,y=map.to_pixels(coord)
        ax.plot(x,y,'or', ms=3)

########################################
if args.initial_weights:
    print('model second pass')

    if args.plot_grid:
        gps_list=grid_gps_flat+gps_coords
    else:
        gps_list=grid_gps_flat_lo+gps_coords

    iter2 = tf.data.Dataset.zip((
        tf.data.Dataset.zip((
            tf.data.Dataset.from_tensor_slices(image),
            tf.data.Dataset.from_tensors(features),
            )).repeat(len(gps_list)),
        tf.data.Dataset.from_tensor_slices(gps_list),
        )).batch(args.batchsize if 'gps' in args.outputs else 1).make_initializable_iterator()
    (image_,features_),gps_=iter2.get_next()

    net,features,loss,loss_regularized,op_metrics,op_endpoints=model.mkModel(
        args,
        image_ if args.input_format=='jpg' else None,
        country_,
        gps_,
        is_training=False,
        features=None if args.input_format=='jpg' else features_,
        )

    ####################
    print('  creating session')
    config = tf.ConfigProto(allow_soft_placement = True)
    sess = tf.Session(config=config)

    reset_global_vars=tf.global_variables_initializer()
    reset_local_vars=tf.local_variables_initializer()
    sess.run([reset_global_vars,reset_local_vars])

    # see https://stackoverflow.com/questions/41621071/restore-subset-of-variables-in-tensorflow
    print('  restoring model')
    chkpt_file=tf.train.latest_checkpoint(args.initial_weights)
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
                    if var.get_shape() == var_to_shape_map[k] and 'kappa' not in k:
                        var_dict[k]=var
                        print('    restoring',knew)
                    else:
                        print('    not restoring',k,'; old=',var_to_shape_map[k],'; new=',var.get_shape())
                except Exception as e:
                    if 'Adam' in k or 'beta1_power' in k or 'beta2_power' in k:
                        pass
                    else:
                        print('    variable not found in graph: ',k)
                        print('      e=',e)
        loader = tf.train.Saver(var_dict)
        loader.restore(sess, chkpt_file)

    restore('')
    print('  model restored from ',args.initial_weights)

    print('  initializing variables')
    sess.run(reset_local_vars)
    sess.graph.finalize()
    sess.run(iter.initializer)
    sess.run(iter2.initializer)


########################################
print('gps output')
if 'gps' in args.outputs:

    ####################
    print('  eval loop')
    local_step=0
    losses=[]
    gpses=[]
    while True:
        try:
            local_step+=1
            gps_run,res_loss,endpoints=sess.run([gps_,loss_regularized,op_endpoints])
            print('    %s  step=%d  loss=%g' %
                ( datetime.datetime.now()
                , local_step
                , res_loss
                ))
            losses+=endpoints['gps_loss/log_loss'].tolist()
            gpses+=gps_run.tolist()
            gps_predicted=endpoints['gps'][0,:]

        except tf.errors.OutOfRangeError:
            break


    # create grid of results
    print('  interpolating')
    import scipy
    import numpy as np
    grid_losses=scipy.interpolate.griddata(
        gpses,
        losses,
        grid_gps_hi,
        'linear',
        fill_value=max(1e3,max(losses))
        )

    # FIXME: ensure that peaks get included in the grid
    grid_losses_peaks=np.zeros([steps+1,steps+1])+1000.0
    for pt,loss in zip(gps_list,losses):
        x,y=map.to_pixels(pt)
        i=int(round((x-px0)*steps/(px2-px0)))
        j=int(round((y-min(px1,px3))*steps/abs(px1-px3)))
        grid_losses_peaks[i,j]=min(grid_losses[i,j],loss)
    #grid_losses_peaks=scipy.ndimage.minimum_filter(grid_losses_peaks,3)
    grid_losses=np.minimum(grid_losses,grid_losses_peaks)

    # trim oceans
    #from mpl_toolkits.basemap import Basemap
    #bm = Basemap()
    #for i in range(0,steps+1):
        #for j in range(0,steps+1):
            #coord=grid_gps_hi[i][j]
            #if not bm.is_land(coord[0],coord[1]):
                #grid_losses[i][j]=1e3

    ####################
    print('  plotting results')

    # convert loss into probabilities
    z=np.zeros([steps+1,steps+1])
    for x in range(0,steps+1):
        for y in range(0,steps+1):
            z[x][y]=math.exp(-grid_losses[x][y]) #log_likelihood(lats[y,x],lons[y,x])

    # test
    #z = np.zeros([steps+1,steps+1])
    #for x in range(0,steps+1):
        #for y in range(0,steps+1):
            #mux=steps*1.3/5.0
            #muy=steps*3.0/5.0
            #sigma=20.0
            #z[x][y]+=math.exp(-((mux-x)**2+(muy-y)**2)/sigma)
            #mux=steps*2.4/5.0
            #muy=steps*3.35/5.0
            ##z[x][y]+=math.exp(-((mux-x)**2+(muy-y)**2)/sigma)
            #mux=steps*1.97/5.0
            #muy=steps*3.38/5.0
            ##z[x][y]+=math.exp(-((mux-x)**2+(muy-y)**2)/sigma)

    # scale probabilities into exponential ranges
    totalweights=np.sum(z)
    print('    totalweights=',totalweights)
    z=np.log(z/totalweights+1e-10)/math.log(10)
    levels=[-4,-3,-2,-1,0]
    #levels=[-3,-2,-1.5,-1,-0.5,1]
    #levels=[-5,-4,-3,-2,-1]

    # generate color map
    #cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#ffffff','#ff9999','#cc6666','#9966cc','#9900ff'])
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#ffffff','#ff9999','#ff6666','#ff3333','#ff0000'])
    norm = plt.Normalize(min(levels),max(levels))

    # generate plot
    zoom=1000/steps
    nx,ny=np.meshgrid(np.linspace(px0,px2,(steps+1)*zoom),np.linspace(px3,px1,(steps+1)*zoom))
    z=scipy.ndimage.zoom(z,zoom)
    cs = ax.contourf(
        nx,
        ny,
        #grid_losses,
        #z,
        np.transpose(z),
        levels=levels,
        antialiased=True,
        cmap=cmap,
        norm=norm,
        alpha=0.8,
        )
    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #cbar = plt.colorbar(cs,cax=cax)
    #cbar.ax.get_yaxis().set_ticks([])
    #for j, lab in enumerate(['0.09%','0.9%','9%','90%']):
    #       #cbar.ax.text(0.6, (2 * j + 1) / 8.0, lab, ha='center', va='center', rotation='vertical', fontsize=30)
    #cbar.ax.get_yaxis().labelpad = 10
    #cbar.set_label('Probability Mass of Plotted Region',fontsize=30)

    print('  plot predicted gps')
    if False and args.plot_gps:
        x,y = map.to_pixels([gps_predicted[0],gps_predicted[1]])
        if x>=min(px0,px2) and x<=max(px0,px2) and y>=min(px1,px3) and y<=max(px1,px3):
            ax.plot(
                x,
                y,
                color='black',
                marker='o',
                ms=15
                )
            ax.annotate(
                'predicted location',
                xy=(x,y),
                xytext=(x-1300, y+250),
                fontsize=30,
                fontweight='bold',
                color='black',
                arrowprops=dict(facecolor='black', shrink=0.15),
                )
        #ax.plot(x, y, 'oc', ms=10);
    print('    gps_predicted=',gps_predicted)

    print('    saving')
    plt.savefig(
        prefixname+'_prob.pdf',
        bbox_inches = 'tight',
        pad_inches = 0
        )

########################################
print('s2 output')
s2grid_fill=False
if 's2' in args.outputs:
    endpoints=sess.run(op_endpoints)
    s2grid_fill=True
    gps_predicted=endpoints['gps'][0,:]
    mixture=endpoints['mixture'][0,:]

    ########################################
    print('  plot predicted gps')
    x,y = map.to_pixels([gps_predicted[0],gps_predicted[1]])
    ax.plot(x, y, 'oc', ms=10);

##############################
if args.s2file:
    print('plotting s2file')
    import s2sphere
    import pickle
    with open(args.s2file,'rb') as f:
        s2cells=pickle.load(f)
        polys=[]
        mixture_indices=[]
        s2cells_onscreen=[]
        cellnum=-1
        for cellid in s2cells:
            cellnum+=1
            zoomlevel=6
            if cellid.level()<zoomlevel:
                children=list(cellid.children(zoomlevel))
            else:
                children=[cellid]
            pts_children=[]
            onscreen=False
            for child in children:
                cell=s2sphere.Cell(child)
                pts=[]
                lats=[]
                lngs=[]
                for i in [0,1,2,3]:
                    latlng=s2sphere.LatLng.from_point(cell.get_vertex(i))
                    lat=latlng.lat().degrees
                    lng=latlng.lng().degrees
                    lats.append(lat)
                    lngs.append(lng)
                    x,y = map.to_pixels(lat,lng)
                    if math.isinf(y):
                        if y<0:
                            y=-1e6
                        else:
                            y=1e6

                    # prevent cells from wrapping "in front" of map
                    if i>0:
                        lng_add=0
                        x1,y_ = map.to_pixels(lat,lng-360)
                        if abs(pts[0][0]-x1)<abs(pts[0][0]-x):
                            x=x1
                        x1,y_ = map.to_pixels(lat,lng+360)
                        if abs(pts[0][0]-x1)<abs(pts[0][0]-x):
                            x=x1
                    pts.append([x,y])

                    # check if pt appears on screen
                    fudge=1000
                    if (((x>=px0-fudge and x<=px2+fudge) or (x>=px2-fudge and x<= px0+fudge)) and
                        ((y>=px1-fudge and y<=px3+fudge) or (y>=px3-fudge and y<= px1+fudge))):
                        onscreen=True


                #if ((pts[0][1]==pts[1][1] and pts[1][1]==pts[2][1] and pts[2][1]==pts[3][1]) or
                    #(pts[0][0]==pts[1][0] and pts[1][0]==pts[2][0] and pts[2][0]==pts[3][0])):
                    #onscreen=False

                # draw each child
                if args.plot_mu:
                    latlng=cellid.to_lat_lng()
                    lat=latlng.lat().degrees
                    lng=latlng.lng().degrees
                    x,y = map.to_pixels(lat,lng)
                    if (((x>=px0 and x<=px2) or (x>=px2 and x<= px0)) and
                        ((y>=px1 and y<=px3) or (y>=px3 and y<= px1))):
                        ax.plot(x, y, 'or', ms=5);

                pts_children.append(pts)

            # remove interior points
            from collections import Counter
            cnt=Counter()
            for poly in pts_children:
                for pt in poly:
                    cnt[(pt[0],pt[1])]+=1
            cnt[max(cnt.keys(),key=lambda (x,y): x)]=1
            cnt[max(cnt.keys(),key=lambda (x,y): y)]=1
            cnt[min(cnt.keys(),key=lambda (x,y): x)]=1
            cnt[min(cnt.keys(),key=lambda (x,y): y)]=1
            pts_hull=[]
            for pt in cnt.keys():
                if cnt[pt]==1 or cnt[pt]==2:
                    pts_hull.append(list(pt))

            # sort points clockwise
            import numpy as np
            origin=np.mean(np.asarray(pts_hull),axis=0)
            if abs(origin[1]-1235.14857787)<1e-6:
                origin[1]-=2000
            if abs(origin[1]-5827.06378127)<1e-6:
                origin[1]+=2000
            pts_hull.sort(key=lambda c:math.atan2(c[0]-origin[0], c[1]-origin[1]))

            # disable drawing in edge cases at the poles
            lngs=[pt[0] for pt in pts_hull]
            if abs(origin[0]-2048.0)<=1e-6 and abs(origin[1]-2048.0)<=1e-6:
                onscreen=False

            # create the polygon
            if pts_hull!=[] and onscreen:
                s2cells_onscreen.append(cellid)
                poly=matplotlib.patches.Polygon(
                    pts_hull,
                    closed=True,
                    edgecolor='b',
                    facecolor='#ff0000',
                    linewidth=1.0,
                    fill=False,
                    alpha=0.5,
                    )
                if 's2' in args.outputs:
                    weight=mixture[cellnum]
                else:
                    weight=0.0
                polys.append((poly,weight))
                mixture_indices.append(cellnum)
                    #ax.add_patch(poly)

        import math
        min_weight=min([m for (p,m) in polys])
        totalweights=sum([1e-10+m-min_weight for (p,m) in polys])
        #totalweights=sum([1e-10+m for (p,m) in polys])
        print('   totalweights=',totalweights)
        if s2grid_fill:
            facecolor=[(1.0,0.0,0.0,(1.0-max(-6,min(0,math.log(1e-10+(m-min_weight)/totalweights)))/-6.0)*0.6) for (p,m) in polys]
        else:
            facecolor=(0,0,0,0)
        patches=matplotlib.collections.PatchCollection(
            [p for (p,m) in polys],
            edgecolor='b',
            facecolor=facecolor,
            )

        if not args.voronoi:
            ax.add_collection(patches)
        else:
            xy_onscreen=[]
            for cellid in s2cells_onscreen:
                latlng=cellid.to_lat_lng()
                lat=latlng.lat().degrees
                lng=latlng.lng().degrees
                x, y = map.to_pixels([lat,lng])
                xy_onscreen.append([x,y])
                #ax.plot(x, y, 'og', ms=15);
                #print('x,y=',(x,y))
            r=50.0
            if args.voronoi_random:
                xy_onscreen=[ [x+np.random.randn()*r,y+np.random.randn()*r] for [x,y] in xy_onscreen ]
            for x,y in xy_onscreen:
                ax.plot(x, y, 'or', ms=5);
            from scipy.spatial import Voronoi, voronoi_plot_2d
            vor = Voronoi(xy_onscreen)
            #voronoi_plot_2d(vor,ax=ax,line_colors='b',show_vertices=False,show_points=False)
            line_segments = []
            for simplex in vor.ridge_vertices:
                simplex = np.asarray(simplex)
                if np.all(simplex >= 0):
                    line_segments.append([(x, y) for x, y in vor.vertices[simplex]])

            from matplotlib.collections import LineCollection
            lc = LineCollection(line_segments,
                                colors='b',
                                #lw=line_width,
                                linestyle='solid')
            ax.add_collection(lc)

            pass

    plt.savefig(
        prefixname+'_s2grid.pdf',
        bbox_inches = 'tight',
        pad_inches = 0
        )

########################################
print('custom output')
if args.custom:
    import numpy as np
    import math
    steps_x=160
    steps_y=120
    nx,ny=np.meshgrid(np.linspace(px0,px2,(steps_x)),np.linspace(px3,px1,(steps_y)))

    gps=args.force_center
    #gps=[47.11,-101.3]
    gpses_vmf=[[50,-100],[-0,-70],[46,6]]

    #ax.plot(px0, px1, 'or', ms=50);
    #ax.plot(px0, px3, 'or', ms=50);
    #ax.plot(px2, px1, 'or', ms=50);
    #ax.plot(px2, px3, 'or', ms=50);

    #x, y = map.to_pixels(gps_gaussian)
    #ax.plot(x, y, 'og', ms=5);
    #x, y = map.to_pixels(gps_vmf)
    #ax.plot(x, y, 'og', ms=5);

    #print(

    nz_gaussian=np.zeros([steps_x,steps_y])
    nz_vmf=np.zeros([steps_x,steps_y])
    for i in range(0,steps_x):
        for j in range(0,steps_y):
            x = min(px0,px2)+i*abs(px0-px2)/float(steps_x)
            y = min(px1,px3)+j*abs(px3-px1)/float(steps_y)

            if args.custom=='both':
                gps_gaussian=[args.force_center[0],(box[1]/3.0+box[3]*2.0/3.0)]
                gps_vmf=[args.force_center[0],(box[1]*3.5/5.0+box[3]/5.0)]
                sigma=0.34
                #mu=map.to_pixels(gps)
                #gps2_lat = min(box[0],box[2])+i*abs(box[0]-box[2])/float(steps_x)
                #gps2_lon = min(box[1],box[3])+j*abs(box[3]-box[1])/float(steps_y)
                gps2=map.from_pixels(x,y)
                nz_gaussian[i][j]=math.exp(- ((gps_gaussian[0]-gps2[0])**2 + (gps_gaussian[1]-gps2[1])**2)/sigma**2)

                gps2=map.from_pixels(x,y)
                mu=map.to_pixels(gps_vmf)
                sigma=200
                nz_vmf[i][j]=math.exp(- ((mu[0]-x)**2 + (mu[1]-y)**2)/sigma**2)
            if args.custom=='gaussian':
                sigma=0.34
                #mu=map.to_pixels(gps)
                #gps2_lat = min(box[0],box[2])+i*abs(box[0]-box[2])/float(steps_x)
                #gps2_lon = min(box[1],box[3])+j*abs(box[3]-box[1])/float(steps_y)
                gps2=map.from_pixels(x,y)
                nz_gaussian[i][j]=math.exp(- ((gps_gaussian[0]-gps2[0])**2 + (gps_gaussian[1]-gps2[1])**2)/sigma**2)

            elif args.custom=='vmf':
                for gps_vmf in gpses_vmf:
                    gps_rad=[math.radians(gps_vmf[0]),math.radians(gps_vmf[1])/2]
                    mu = [ math.sin(gps_rad[0])
                         , math.cos(gps_rad[0]) * math.sin(gps_rad[1]*2)
                         , math.cos(gps_rad[0]) * math.cos(gps_rad[1]*2)
                         ]
                    gps2=map.from_pixels(x,y)
                    gps2_rad=[math.radians(gps2[0]),math.radians(gps2[1])/2]
                    p  = [ math.sin(gps2_rad[0])
                         , math.cos(gps2_rad[0]) * math.sin(gps2_rad[1]*2)
                         , math.cos(gps2_rad[0]) * math.cos(gps2_rad[1]*2)
                         ]
                    kappa=4.0e1
                    nz_vmf[i][j]+=math.exp(kappa*(p[0]*mu[0]+p[1]*mu[1]+p[2]*mu[2]))
                    nz_vmf[i][j]+=kappa/(4*math.pi*math.sinh(kappa))*math.exp(kappa*(p[0]*mu[0]+p[1]*mu[1]+p[2]*mu[2]))


    totalweights=np.sum(nz_gaussian)
    nz_gaussian=np.log(nz_gaussian/totalweights+1e-10)/math.log(10)

    totalweights=np.sum(nz_vmf)
    nz_vmf=np.log(nz_vmf/totalweights+1e-10)/math.log(10)
    levels=[-4,-3,-2,-1,0]
    levels_gaussian=[-4,-3,-2,-1,0]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#ffffff','#ff9999','#cc6666','#9966cc','#9900ff'])
    norm = plt.Normalize(min(levels),max(levels))

    cs = ax.contourf(
        nx,
        ny,
        np.transpose(nz_vmf),
        #levels=levels,
        #antialiased=True,
        cmap=cmap,
        #norm=norm,
        alpha=0.5,
        )
    cs = ax.contourf(
        nx,
        ny,
        np.transpose(nz_gaussian),
        levels=levels_gaussian,
        antialiased=True,
        cmap=cmap,
        norm=norm,
        alpha=0.5,
        )

    plt.savefig(
        prefixname+'_'+args.custom+'.pdf',
        bbox_inches = 'tight',
        pad_inches = 0
        )

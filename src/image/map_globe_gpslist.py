from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import s2sphere
import smopy
import sys

########################################
print('loading data points')

gps_coords=[]
#for i in range(0,1000):
    #gps_coords.append([random.uniform(-90,90),random.uniform(-180,180)])

#with open('s2.old/gps.list','rb') as input_file:
    #gps_coords=pickle.load(input_file)

with open('tweets.gpslist') as f:
    i=0
    while True:
        i+=1
        if i%10000==0:
            print('i=',i)
        try:
            gps=pickle.load(f)
            if i>100000:
                gps_coords.append(gps)
        except:
            break
        if (i+1)%200000==0:
            break

########################################
print('create map')
lat_offset=7
lon_offset=0
lon_km=36000
import math
range_lat=math.degrees(min(lon_km,15000)/6371.0)/2
range_lon=math.degrees(lon_km/6371.0)/2

box=(lat_offset-range_lat,
     lon_offset-range_lon,
     lat_offset+range_lat,
     lon_offset+range_lon,
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
ax = map.show_mpl(figsize=(16, 16),alpha=0)
#ax = map.show_mpl(hide_image=True,figsize=(16, 16))

px0,px1 = map.to_pixels((box[0],box[1]))
px2,px3 = map.to_pixels((box[2],box[3]))
if True:
    ratio=3.0/4.0
    yrange=(px2-px0)*ratio
    yave=(px1+px3)/2.0
    px1=yave+yrange/2.0
    px3=yave-yrange/2.0
    box0,box1=map.from_pixels(px0,px1) #yave-yrange/2.0)
    box2,box3=map.from_pixels(px2,px3) #yave+yrange/2.0)
    box=[box0,box1,box2,box3]
ax.set_xlim((px0,px2))
ax.set_ylim((px1,px3))

########################################
print('gps2xy')
xy_coords=[]
i=0
for gps in gps_coords:
    x, y = map.to_pixels(gps)
    xy_coords.append([x,y])
    if i%10000==0:
        print('i=',i)
    i+=1
    #if i>1e5:
        #break

########################################
print('plotting')
#xbins=360
#ybins=180
#hist=np.zeros([lat_bins,lon_bins])
#
#i=0
#for gps in gps_coords:
    #i+=1
    #gps_cell=s2sphere.CellId.from_lat_lng(s2.LatLng.from_degrees(gps[0],gps[1]))
#
    #if i>=10000:
        #break
#for x,y in xy_coords:
    #ax.plot(x, y, 'oc', ms=10);

ax.scatter([x for x,y in xy_coords],[y for x,y in xy_coords],s=0.1,color='r',alpha=float(sys.argv[1]))
plt.savefig(
    'img/maps/test_twitter_'+sys.argv[1]+'.png',
    bbox_inches = 'tight',
    pad_inches = 0,
    transparent = True,
    )

#steps=1000
#nx,ny=np.meshgrid(np.linspace(px0,px2,steps),np.linspace(px3,px1,steps))
#zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
#cs = ax.contourf(
    #nx,
    #ny,
    ##grid_losses,
    ##z,
    #np.transpose(z),
    #levels=levels,
    #antialiased=True,
    #cmap=cmap,
    #norm=norm,
    #alpha=0.5,
    #)

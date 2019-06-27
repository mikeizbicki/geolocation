from mpl_toolkits.basemap import Basemap
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# set up orthographic map projection with
# perspective of satellite looking down at 50N, 100W.
# use low resolution coastlines.
map = Basemap(projection='ortho',lat_0=45,lon_0=-100,resolution='l')
# draw coastlines, country boundaries, fill continents.
map.drawcoastlines(linewidth=0.25)
map.drawcountries(linewidth=0.25)
map.fillcontinents(color='white',lake_color='lightgray',zorder=1)

# draw the edge of the map projection region (the projection limb)
map.drawmapboundary(fill_color='lightgray')
# draw lat/lon grid lines every 30 degrees.
map.drawmeridians(np.arange(0,360,30))
map.drawparallels(np.arange(-90,90,30))
# make up some data on a regular lat/lon grid.
nlats = 73; nlons = 145; delta = 2.*np.pi/(nlons-1)
lats = (0.5*np.pi-delta*np.indices((nlats,nlons))[0,:,:])
lons = (delta*np.indices((nlats,nlons))[1,:,:])


#print('lats=',lats.shape)
def mk3d(lat,lon):
    lats_rad=lat#/180*math.pi
    lons_rad=lon#/180*math.pi
    x=np.sin(lats_rad)
    y=np.cos(lats_rad)*np.sin(lons_rad)
    z=np.cos(lats_rad)*np.cos(lons_rad)
    return [x,y,z]

#gpses=[[50,-100],[20,-100],[0,-80]]
gpses=[[50,-100],[-10,-60],[46,6]]
kappas=[math.exp(2.0),math.exp(2.0),math.exp(2.0)]
#gpses=[[50,-100]]
#kappas=[math.exp(1.0)]

mus=[mk3d(math.radians(gps[0]),math.radians(gps[1])) for gps in gpses]
x=mk3d(lats,lons)

density=[kappa/math.sinh(kappa)*np.exp(kappa*(x[0]*mu[0]+x[1]*mu[1]+x[2]*mu[2]))
    for (kappa,mu) in zip(kappas,mus)
]
##density=[ np.asarray([dd/sum(d) for dd in d]) for d in density]
density=[ d/np.max(d) for d in density]
#density[0]=density[0]/np.max(density[0])
#density[1]=density[1]/np.max(density[1])
print('max[0]=',np.max(density[0]))
print('max[1]=',np.max(density[1]))
#density=density[0]
density=sum(density)
#density=(kappa*x[0]*mu[0]+x[1]*mu[1]+x[2]*mu[2])

#print('density=',density)

totalweights=np.sum(density)
#density=np.log(density/totalweights+1e-10)/math.log(10)
levels=[-4,-3,-2,-1,0]
levels_gaussian=[-4,-3,-2,-1,0]

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['#ffffff','#ff9999','#cc6666','#9966cc','#9900ff'])
norm = plt.Normalize(min(levels),max(levels))

#wave = 0.75*(np.sin(2.*lats)**8*np.cos(4.*lons))
#mean = 0.5*np.cos(2.*lats)*((np.sin(2.*lats))**2 + 2.)
# compute native map projection coordinates of lat/lon grid.
x, y = map(lons*180./np.pi, lats*180./np.pi)
# contour data over the map.
cs = map.contourf(x,y,density,15,linewidths=1.5,alpha=0.5,zorder=2,
        #levels=levels,
        #antialiased=True,
        cmap=cmap,
        #norm=norm,
)
#plt.title('contour lines over filled continent background')
#plt.show()
plt.savefig(
    'img/maps/globe.pdf',
    bbox_inches = 'tight',
    pad_inches = 0
    )

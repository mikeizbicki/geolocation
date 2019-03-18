from __future__ import print_function

import os
import pickle

data='data/BillionTwitter'
files_all=[]
for path_date in os.listdir(data):
    path_date_full=os.path.join(data,path_date)
    if os.path.isdir(path_date_full):
        for path_hour in os.listdir(path_date_full):
            files_all.append(os.path.join(path_date_full,path_hour))

import gzip
import simplejson as json
totalcoords=0
totalfiles=0
with open('tweets.gpslist','w') as out:
    for filename in files_all:
        totalfiles+=1
        with gzip.open(filename,'rt') as f:
            while True:
                nextline=f.readline()
                if nextline=='':
                    break
                try:
                    data=json.loads(nextline)
                    lat=data['geo']['coordinates'][0]
                    lon=data['geo']['coordinates'][1]
                    pickle.dump([lat,lon],out)
                    totalcoords+=1
                    if totalcoords%1000==0:
                        print('totalcoords=',totalcoords,'totalfiles=',totalfiles)
                except TypeError as e:
                    pass
                except KeyError as e:
                    pass



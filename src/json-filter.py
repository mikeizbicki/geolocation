#!/usr/bin/env python

from __future__ import print_function

import pprint
import simplejson as json
import sys

while True:
    nextline=sys.stdin.readline()
    if nextline=='':
        break

    #data=json.loads(nextline)
    #pprint.pprint(data)
    #sys.exit(0)

    try:
        data=json.loads(nextline)
        data_new={
            'text' : data['text'],
            'lang' : data['lang'],
            'timestamp_ms' : data['timestamp_ms'],
        }

        try:
            data_new['geo']=data['geo']
        except:
            pass

        try:
            keys=['full_name','country_code','place_type','bounding_box']
            data_new['place']={}
            for key in keys:
                data_new['place'][key] = data['place'][key]
        except:
            pass

        #try:
            #keys=['url','screen_name','name','location','description']
            #data_new['user']={}
            #for key in keys:
                #data_new['user'][key] = data['user'][key]
        #except:
            #pass

        print(json.dumps(data_new))

    except:
        pass

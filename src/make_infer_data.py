import geocoder
import simplejson as json

patterns={
    'en':'I am at $place.',
    'es':'Estoy en $place.',
    'und':'$place',
    }

places=['Riverside, California, United States']

def mk_json(lang,pattern,place):
    g=geocoder.osm(place)
    #print('g=',g.json)
    tweet={
        'text':pattern.replace('$place',place),
        'lang':lang,
        'geo':{
            'type':'Point',
            'coordinates':[g.latlng[0],g.latlng[1]],
        },
        'place':{
            'place_type':'city',
            'country_code':'US',
            'full_name':place,
        },
        'timestamp_ms':1508573111095
    }
    return json.dumps(tweet)

print(mk_json('en','I am at $place','Riverside, California'))

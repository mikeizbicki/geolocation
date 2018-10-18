#!/usr/bin/env python

from __future__ import print_function

# extract gps info from images
def imgpath2gps(imgpath):
    from libxmp.utils import file_to_dict
    def lat_lon_str_to_deg(str):
        parts=str.split(',')
        deg=float(parts[0])
        min=float(parts[1][:-1])
        dir=parts[1][-1]
        if dir=='N' or dir=='E':
            sign=1
        else:
            sign=-1
        return sign*(deg+min/60.0)
    try:
        xmp=file_to_dict(imgpath)
        exif=xmp[u'http://ns.adobe.com/exif/1.0/']
        lat_str=[x[1] for x in exif if x[0]==u'exif:GPSLatitude'][0]
        lon_str=[x[1] for x in exif if x[0]==u'exif:GPSLongitude'][0]
        lat=lat_lon_str_to_deg(lat_str)
        lon=lat_lon_str_to_deg(lon_str)
    except KeyError as e:
        lat=float('NaN')
        lon=float('NaN')
    #import numpy as np
    #return np.array([lat,lon],np.float32)
    return [lat,lon]

def gps2country(gps):
    import reverse_geocode
    try:
        info=reverse_geocode.search([gps])
        return info[0]['country_code']
    except IndexError:
        return 'UNDEFINED'

def imgpath2labels(imgpath):
    import numpy as np
    gps=imgpath2gps(imgpath)
    gps_np=np.array(gps,dtype=np.float32)

    country=gps2country(gps)
    country_int=country2int(country)

    return (gps_np,country_int)

def tf_imgpath2labels(imgpath):
    import tensorflow as tf
    gps,country=tf.py_func(imgpath2labels,[imgpath],(tf.float32,tf.int64))
    gps.set_shape([2])
    country.set_shape([])
    return (gps,country)


# FIXME: this is shared from twitter geolocation
country_codes_iso=['AF','AL','DZ','AS','AD','AO','AI','AQ','AG','AR','AM','AW','AU','AT','AZ','BS','BH','BD','BB','BY','BE','BZ','BJ','BM','BT','BO','BA','BW','BR','IO','VG','BN','BG','BF','BI','KH','CM','CA','CV','KY','CF','TD','CL','CN','CX','CC','CO','KM','CK','CR','HR','CU','CW','CY','CZ','CD','DK','DJ','DM','DO','TL','EC','EG','SV','GQ','ER','EE','ET','FK','FO','FJ','FI','FR','PF','GA','GM','GE','DE','GH','GI','GR','GL','GD','GU','GT','GG','GN','GW','GY','HT','HN','HK','HU','IS','IN','ID','IR','IQ','IE','IM','IL','IT','CI','JM','JP','JE','JO','KZ','KE','KI','XK','KW','KG','LA','LV','LB','LS','LR','LY','LI','LT','LU','MO','MK','MG','MW','MY','MV','ML','MT','MH','MR','MU','YT','MX','FM','MD','MC','MN','ME','MS','MA','MZ','MM','NA','NR','NP','NL','AN','NC','NZ','NI','NE','NG','NU','KP','MP','NO','OM','PK','PW','PS','PA','PG','PY','PE','PH','PN','PL','PT','PR','QA','CG','RE','RO','RU','RW','BL','SH','KN','LC','MF','PM','VC','WS','SM','ST','SA','SN','RS','SC','SL','SG','SX','SK','SI','SB','SO','ZA','KR','SS','ES','LK','SD','SR','SJ','SZ','SE','CH','SY','TW','TJ','TZ','TH','TG','TK','TO','TT','TN','TR','TM','TC','TV','VI','UG','UA','AE','GB','US','UY','UZ','VU','VA','VE','VN','WF','EH','YE','ZM','ZW']
country_codes_empirical=['GP','MQ','GF','BQ','AX','BV','TF','NF','UM','GS','HM','UNDEFINED']
country_codes=['']+country_codes_iso+country_codes_empirical
country_codes_uniq=list(set(country_codes))
#country_codes=country_codes_uniq
num_countries=len(country_codes)

def country2int(str):
    try:
        return country_codes.index(str)
    except:
        #print('unknown country code = [',str,']')
        return 0


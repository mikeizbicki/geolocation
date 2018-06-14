langs_emperical=[u'am',u'ar',u'bg',u'bn',u'bo',u'ckb',u'cs',u'cy',u'da',u'de',u'dv',u'el',u'en',u'es',u'et',u'eu',u'fa',u'fi',u'fr',u'gu',u'hi',u'ht',u'hu',u'hy',u'in',u'is',u'it',u'iw',u'ja',u'ka',u'km',u'kn',u'ko',u'lo',u'lt',u'lv',u'ml',u'mr',u'my',u'ne',u'nl',u'no',u'or',u'pa',u'pl',u'ps',u'pt',u'ro',u'ru',u'sd',u'si',u'sl',u'sr',u'sv',u'ta',u'te',u'th',u'tl',u'tr',u'uk',u'und',u'ur',u'vi',u'zh']
langs_iso_639_1=[u'ab',u'aa',u'af',u'ak',u'sq',u'am',u'ar',u'an',u'hy',u'as',u'av',u'ae',u'ay',u'az',u'bm',u'ba',u'eu',u'be',u'bn',u'bh',u'bi',u'nb',u'bs',u'br',u'bg',u'my',u'es',u'ca',u'km',u'ch',u'ce',u'ny',u'ny',u'zh',u'za',u'cu',u'cu',u'cv',u'kw',u'co',u'cr',u'hr',u'cs',u'da',u'dv',u'dv',u'nl',u'dz',u'en',u'eo',u'et',u'ee',u'fo',u'fj',u'fi',u'nl',u'fr',u'ff',u'gd',u'gl',u'lg',u'ka',u'de',u'ki',u'el',u'kl',u'gn',u'gu',u'ht',u'ht',u'ha',u'he',u'hz',u'hi',u'ho',u'hu',u'is',u'io',u'ig',u'id',u'ia',u'ie',u'iu',u'ik',u'ga',u'it',u'ja',u'jv',u'kl',u'kn',u'kr',u'ks',u'kk',u'ki',u'rw',u'ky',u'kv',u'kg',u'ko',u'kj',u'ku',u'kj',u'ky',u'lo',u'la',u'lv',u'lb',u'li',u'li',u'li',u'ln',u'lt',u'lu',u'lb',u'mk',u'mg',u'ms',u'ml',u'dv',u'mt',u'gv',u'mi',u'mr',u'mh',u'ro',u'ro',u'mn',u'na',u'nv',u'nv',u'nd',u'nr',u'ng',u'ne',u'nd',u'se',u'no',u'nb',u'nn',u'ii',u'ny',u'nn',u'ie',u'oc',u'oj',u'cu',u'cu',u'cu',u'or',u'om',u'os',u'os',u'pi',u'pa',u'ps',u'fa',u'pl',u'pt',u'pa',u'ps',u'qu',u'ro',u'rm',u'rn',u'ru',u'sm',u'sg',u'sa',u'sc',u'gd',u'sr',u'sn',u'ii',u'sd',u'si',u'si',u'sk',u'sl',u'so',u'st',u'nr',u'es',u'su',u'sw',u'ss',u'sv',u'tl',u'ty',u'tg',u'ta',u'tt',u'te',u'th',u'bo',u'ti',u'to',u'ts',u'tn',u'tr',u'tk',u'tw',u'ug',u'uk',u'ur',u'ug',u'uz',u'ca',u've',u'vi',u'vo',u'wa',u'cy',u'fy',u'wo',u'xh',u'yi',u'yo',u'za',u'zu']
langs=[u'unknown']+langs_emperical+langs_iso_639_1
langs_uniq=list(set(langs))
langs=langs_uniq

def lang2int(lang):
    try:
        return langs.index(lang)
    except:
        print('unk=',lang)
        return 0

def int2lang(i):
    return langs[i]

def softmax2lang(x):
    import numpy as np
    return int2lang(np.argmax(x))

################################################################################

country_codes_iso=['AF','AL','DZ','AS','AD','AO','AI','AQ','AG','AR','AM','AW','AU','AT','AZ','BS','BH','BD','BB','BY','BE','BZ','BJ','BM','BT','BO','BA','BW','BR','IO','VG','BN','BG','BF','BI','KH','CM','CA','CV','KY','CF','TD','CL','CN','CX','CC','CO','KM','CK','CR','HR','CU','CW','CY','CZ','CD','DK','DJ','DM','DO','TL','EC','EG','SV','GQ','ER','EE','ET','FK','FO','FJ','FI','FR','PF','GA','GM','GE','DE','GH','GI','GR','GL','GD','GU','GT','GG','GN','GW','GY','HT','HN','HK','HU','IS','IN','ID','IR','IQ','IE','IM','IL','IT','CI','JM','JP','JE','JO','KZ','KE','KI','XK','KW','KG','LA','LV','LB','LS','LR','LY','LI','LT','LU','MO','MK','MG','MW','MY','MV','ML','MT','MH','MR','MU','YT','MX','FM','MD','MC','MN','ME','MS','MA','MZ','MM','NA','NR','NP','NL','AN','NC','NZ','NI','NE','NG','NU','KP','MP','NO','OM','PK','PW','PS','PA','PG','PY','PE','PH','PN','PL','PT','PR','QA','CG','RE','RO','RU','RW','BL','SH','KN','LC','MF','PM','VC','WS','SM','ST','SA','SN','RS','SC','SL','SG','SX','SK','SI','SB','SO','ZA','KR','SS','ES','LK','SD','SR','SJ','SZ','SE','CH','SY','TW','TJ','TZ','TH','TG','TK','TO','TT','TN','TR','TM','TC','TV','VI','UG','UA','AE','GB','US','UY','UZ','VU','VA','VE','VN','WF','EH','YE','ZM','ZW']
country_codes_empirical=['GP','MQ','GF','BQ','AX','BV','TF','NF','UM','GS','HM']
country_codes=['']+country_codes_iso+country_codes_empirical
country_codes_uniq=list(set(country_codes))
country_codes=country_codes_uniq

def country2int(str):
    try:
        return country_codes.index(str)
    except:
        print('unknown country code = [',str,']')
        return 0

def int2country(i):
    return country_codes[i]

def softmax2country(x):
    import numpy as np
    return int2country(np.argmax(x))

################################################################################

def init_loc_hash(args):
    global loc_dict
    global loc_count
    global loc_max
    loc_dict = {}
    loc_count = 0
    loc_max=2**args.loc_hashsize

def loc2int_diff(str):
    global loc_dict
    global loc_count
    global loc_max
    if str in loc_dict:
        return loc_dict[str]
    else:
        if loc_count>=loc_max:
            return loc_max-1
        else:
            #if loc_count%1000==0:
                #print('loc_count=',loc_count,'/',loc_max)
            loc_dict[str]=loc_count
            loc_count+=1
            return loc_dict[str]

def loc2int(str):
    return hash(str)%loc_max

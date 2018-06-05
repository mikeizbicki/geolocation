langs_emperical=[u'am',u'ar',u'bg',u'bn',u'bo',u'ckb',u'cs',u'cy',u'da',u'de',u'dv',u'el',u'en',u'es',u'et',u'eu',u'fa',u'fi',u'fr',u'gu',u'hi',u'ht',u'hu',u'hy',u'in',u'is',u'it',u'iw',u'ja',u'ka',u'km',u'kn',u'ko',u'lo',u'lt',u'lv',u'ml',u'mr',u'my',u'ne',u'nl',u'no',u'or',u'pa',u'pl',u'ps',u'pt',u'ro',u'ru',u'sd',u'si',u'sl',u'sr',u'sv',u'ta',u'te',u'th',u'tl',u'tr',u'uk',u'und',u'ur',u'vi',u'zh']
langs_iso_639_1=['ab','aa','af','ak','sq','am','ar','an','hy','as','av','ae','ay','az','bm','ba','eu','be','bn','bh','bi','nb','bs','br','bg','my','es','ca','km','ch','ce','ny','ny','zh','za','cu','cu','cv','kw','co','cr','hr','cs','da','dv','dv','nl','dz','en','eo','et','ee','fo','fj','fi','nl','fr','ff','gd','gl','lg','ka','de','ki','el','kl','gn','gu','ht','ht','ha','he','hz','hi','ho','hu','is','io','ig','id','ia','ie','iu','ik','ga','it','ja','jv','kl','kn','kr','ks','kk','ki','rw','ky','kv','kg','ko','kj','ku','kj','ky','lo','la','lv','lb','li','li','li','ln','lt','lu','lb','mk','mg','ms','ml','dv','mt','gv','mi','mr','mh','ro','ro','mn','na','nv','nv','nd','nr','ng','ne','nd','se','no','nb','nn','ii','ny','nn','ie','oc','oj','cu','cu','cu','or','om','os','os','pi','pa','ps','fa','pl','pt','pa','ps','qu','ro','rm','rn','ru','sm','sg','sa','sc','gd','sr','sn','ii','sd','si','si','sk','sl','so','st','nr','es','su','sw','ss','sv','tl','ty','tg','ta','tt','te','th','bo','ti','to','ts','tn','tr','tk','tw','ug','uk','ur','ug','uz','ca','ve','vi','vo','wa','cy','fy','wo','xh','yi','yo','za','zu']
langs=['unknown']+langs_emperical+langs_iso_639_1

def lang2int(lang):
    try:
        return langs.index(lang)
    except:
        print('unk=',lang)
        return 0

################################################################################

# country
country_codes_iso=['AF','AL','DZ','AS','AD','AO','AI','AQ','AG','AR','AM','AW','AU','AT','AZ','BS','BH','BD','BB','BY','BE','BZ','BJ','BM','BT','BO','BA','BW','BR','IO','VG','BN','BG','BF','BI','KH','CM','CA','CV','KY','CF','TD','CL','CN','CX','CC','CO','KM','CK','CR','HR','CU','CW','CY','CZ','CD','DK','DJ','DM','DO','TL','EC','EG','SV','GQ','ER','EE','ET','FK','FO','FJ','FI','FR','PF','GA','GM','GE','DE','GH','GI','GR','GL','GD','GU','GT','GG','GN','GW','GY','HT','HN','HK','HU','IS','IN','ID','IR','IQ','IE','IM','IL','IT','CI','JM','JP','JE','JO','KZ','KE','KI','XK','KW','KG','LA','LV','LB','LS','LR','LY','LI','LT','LU','MO','MK','MG','MW','MY','MV','ML','MT','MH','MR','MU','YT','MX','FM','MD','MC','MN','ME','MS','MA','MZ','MM','NA','NR','NP','NL','AN','NC','NZ','NI','NE','NG','NU','KP','MP','NO','OM','PK','PW','PS','PA','PG','PY','PE','PH','PN','PL','PT','PR','QA','CG','RE','RO','RU','RW','BL','SH','KN','LC','MF','PM','VC','WS','SM','ST','SA','SN','RS','SC','SL','SG','SX','SK','SI','SB','SO','ZA','KR','SS','ES','LK','SD','SR','SJ','SZ','SE','CH','SY','TW','TJ','TZ','TH','TG','TK','TO','TT','TN','TR','TM','TC','TV','VI','UG','UA','AE','GB','US','UY','UZ','VU','VA','VE','VN','WF','EH','YE','ZM','ZW']
country_codes_empirical=['GP','MQ','GF','BQ','AX','BV','TF','NF','UM','GS','HM']
country_codes=['']+country_codes_iso+country_codes_empirical
def country2int(str):
    try:
        return country_codes.index(str)
    except:
        print('unknown country code = [',str,']')
        return 0

from __future__ import print_function

# set unbuffered output
import os
import sys
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

import googletrans
import pycountry

translator=googletrans.Translator()

country_codes=['AD','AE','AF','AG','AI','AL','AM','AO','AQ','AR','AS','AT','AU','AW','AX','AZ','BA','BB','BD','BE','BF','BG','BH','BI','BJ','BL','BM','BN','BO','BQ','BR','BS','BT','BV','BW','BY','BZ','CA','CC','CD','CF','CG','CH','CI','CK','CL','CM','CN','CO','CR','CU','CV','CW','CX','CY','CZ','DE','DJ','DK','DM','DO','DZ','EC','EE','EG','EH','ER','ES','ET','FI','FJ','FK','FM','FO','FR','GA','GB','GD','GE','GF','GG','GH','GI','GL','GM','GN','GP','GQ','GR','GS','GT','GU','GW','GY','HK','HN','HR','HT','HU','ID','IE','IL','IM','IN','IO','IQ','IR','IS','IT','JE','JM','JO','JP','KE','KG','KH','KI','KM','KN','KP','KR','KW','KY','KZ','LA','LB','LC','LI','LK','LR','LS','LT','LU','LV','LY','MA','MC','MD','ME','MF','MG','MH','MK','ML','MM','MN','MO','MP','MQ','MR','MS','MT','MU','MV','MW','MX','MY','MZ','NA','NC','NE','NF','NG','NI','NL','NO','NP','NR','NU','NZ','OM','PA','PE','PF','PG','PH','PK','PL','PM',]
country_codes=['PN','PR','PT','PW','PY','QA','RE','RO','RS','RU','RW','SA','SB','SC','SD','SE','SG','SH','SI','SK','SL','SM','SN','SO','SR','ST','SV','SX','SY','SZ','TC','TD','TF','TG','TH','TJ','TK','TL','TM','TN','TO','TR','TT','TV','TW','TZ','UA','UG','UM','US','UY','UZ','VA','VC','VE','VG','VI','VN','VU','WF','WS','XK','YE','YT','ZA','ZM','ZW']

outfile_full=open('data/infer/custom_full2','w')
outfile_country=open('data/infer/custom_country2','w')
outfile_misspellings=open('data/infer/custom_misspellings2','w')

for country_code in country_codes:
    print('country_code=',country_code)
    print('    ',end='')
    for lang in googletrans.LANGUAGES:
        print(lang,end=' ')
        country=pycountry.countries.get(alpha_2=country_code)
        while True:
            try:
                res=translator.translate([
                    unicode(country.name),
                    u"I'm at a restaurant in "+unicode(country.name),
                    u"I'm having fun in "+unicode(country.name),
                    unicode(country.name)+u' is great!',
                    u"I'm in "+unicode(country.name),
                    ],dest=lang)
                break
            except:
                print('  waiting')
                import time
                time.sleep(20)
                continue

        def mktweet(text):
            return (u'{"lang": "'+unicode(lang)+u'", "text": "'+unicode(text)+u'", "geo": null, "place": {"place_type": "poi", "bounding_box": {"type": "Polygon", "coordinates": [0,0]}, "country_code": "'+unicode(country_code)+u'", "full_name": ""}, "timestamp_ms": "0"}').encode('unicode-escape')+'\n'

        outfile_country.write(mktweet(u"I'm at a restaurant in "+res[0].text))

        n=len(country.name)
        for misspelling in [country.name[0:i]+country.name[i+1:n] for i in range(0,n)]:
            outfile_misspellings.write(mktweet(u"I'm at a restaurant in "+misspelling))

        for full in res[1:]:
            outfile_full.write(mktweet(full.text))

#kuwait={
    #'Afrikaans' : (u'restaurant straat',u'Ek is by restaurantstraat in Koeweit',u'Koeweit'),
    #'Albanian' : (u'rruga e restorantit',u'Jam n\u00EB nj\u00EB restorant n\u00EB Kuvajt',u'Kuvajti'),
    #'Amharic' : (u'\u12E8\u121D\u130D\u1265 \u1264\u1275 \u1218\u1295\u1308\u12F5',u'\u1260\u12A9\u12CC\u1275 \u1260\u121A\u1308\u1298\u12CD \u12E8\u121D\u130D\u1265 \u1264\u1275 \u1218\u1295\u1308\u12F5 \u120B\u12ED \u1290\u129D',u'\u12B5\u12CC\u1275'),
    #'Arabic' : (u'\u0634\u0627\u0631\u0639 \u0627\u0644\u0645\u0637\u0639\u0645',u'\u0623\u0646\u0627 \u0641\u064A \u0634\u0627\u0631\u0639 \u0627\u0644\u0645\u0637\u0627\u0639\u0645 \u0641\u064A \u0627\u0644\u0643\u0648\u064A\u062A',u'\u0627\u0644\u0643\u0648\u064A\u062A'),
    #'Armenian' : (u'\u057C\u0565\u057D\u057F\u0578\u0580\u0561\u0576\u0561\u0575\u056B\u0576 \u0583\u0578\u0572\u0578\u0581',u'\u0535\u057D \u0554\u0578\u0582\u057E\u0565\u0575\u0569\u056B \u057C\u0565\u057D\u057F\u0578\u0580\u0561\u0576\u0561\u0575\u056B\u0576 \u0583\u0578\u0572\u0578\u0581\u0578\u0582\u0574 \u0565\u0574',u'\u0554\u0578\u0582\u057E\u0565\u0575\u0569\u0568'),
    #'Azerbaijani' : (u'restoran caddesi',u'K\u00FCveytd\u0259 restoran k\u00FC\u00E7\u0259sind\u0259 oldum',u'K\u00FCveyt'),
    #'Basque' : (u'jatetxe kale',u'Kuwait jatetxeko kalean nago',u'Kuwaiten'),
    #'Belarusian' : (u'\u0440\u044D\u0441\u0442\u0430\u0440\u0430\u043D\u043D\u0430\u044F \u0432\u0443\u043B\u0456\u0446\u0430',u'\u042F \u045E \u0440\u044D\u0441\u0442\u0430\u0440\u0430\u043D\u0435 \u0432\u0443\u043B\u0456\u0446\u044B \u045E \u041A\u0443\u0432\u0435\u0439\u0446\u0435',u'\u041A\u0443\u0432\u0435\u0439\u0442'),
    #'Bengali' : (u'\u09B0\u09C7\u09B8\u09CD\u099F\u09C1\u09B0\u09C7\u09A8\u09CD\u099F \u09B0\u09BE\u09B8\u09CD\u09A4\u09BE\u09B0',u'\u0986\u09AE\u09BF \u0995\u09C1\u09AF\u09BC\u09C7\u09A4\u09C7 \u09B0\u09C7\u09B8\u09CD\u09A4\u09CB\u09B0\u09BE\u0981 \u09B0\u09BE\u09B8\u09CD\u09A4\u09BE\u09AF\u09BC \u0986\u099B\u09BF',u'\u0995\u09C1\u09AF\u09BC\u09C7\u09A4'),
    #'Bosnian' : (u'restoran ulice',u'Ja sam u restoranskoj ulici u Kuvajtu',u'Kuvajt'),
    #'Bulgarian' : (u'\u0440\u0435\u0441\u0442\u043E\u0440\u0430\u043D\u0442 \u0443\u043B\u0438\u0446\u0430',u'\u0410\u0437 \u0441\u044A\u043C \u043D\u0430 \u0443\u043B\u0438\u0446\u0430\u0442\u0430 \u043D\u0430 \u0440\u0435\u0441\u0442\u043E\u0440\u0430\u043D\u0442\u0430 \u0432 \u041A\u0443\u0432\u0435\u0439\u0442',u'\u041A\u0443\u0432\u0435\u0439\u0442'),
    #'Catalan' : (u'carrer del restaurant',u'Estic al carrer del restaurant a Kuwait',u'Kuwait'),
    #'Cebuano' : (u'dalan sa restawran',u'Sa dalan sa restaurant sa Kuwait',u'Kuwait'),
    #'Chichewa' : (u'malo ogulitsira',u'Ndiri ku malo odyera ku Kuwait',u'Kuwait'),
    #'Chinese' : (u'\u9910\u5385\u8857',u'\u6211\u5728\u79D1\u5A01\u7279\u7684\u9910\u5385\u8857',u'\u79D1\u5A01\u7279'),
    #'Corsican' : (u'restaurant street',u'S\u00F2 \u00E0 a strada di ristorante in Kuwait',u'Kuwait'),
    #'Croatian' : (u'restoran ulica',u'Ja sam u restorantskoj ulici u Kuvajtu',u'Kuvajt'),
    #'Czech' : (u'restaurace ulice',u'Jsem na ulici restaurace v Kuvajtu',u'Kuvajt'),
    #'Danish' : (u'restaurant street',u'Jeg er p\u00E5 restaurant street i Kuwait',u'Kuwait'),
    #'Dutch' : (u'restaurant straat',u'Ik ben in restaurant street in Koeweit',u'Koeweit'),
    #'English' : (u'restaurant street',u"I'm at restaurant street",u'Kuwait'),
    #'Esperanto' : (u'restoracio strato',u'Mi estas \u0109e restoracio en Kuvajto',u'Kuvajto'),
    #'Estonian' : (u'restorani t\u00E4nav',u'Ma olen Kuveidi restoranis',u'Kuveit'),
    #a
    #'Burmese' : (u'',u'',u'\u1000\u1030\u101D\u102D\u1010\u103A'),
    #'Chinese' : (u'',u'',u'\u79D1\u5A01\u7279'),
    #'Czech' : (u'',u'',u'Kuvajt'),
    #'English' : (u'',u'',u'Kuwait'),
    #'Esperanto' : (u'',u'',u'kuwaito'),
    #'Greek' : (u'',u'',u'\u03BA\u03BF\u03C5\u03B2\u03AD\u03B9\u03C4'),
    #'Gujarathi' : (u'',u'',u'\u0A95\u0AC1\u0AB5\u0AC8\u0AA4'),
    #'Hawaiian' : (u'',u'',u'Kuete'),
    #'Hebrew' : (u'',u'',u'\u05DB\u05D5\u05D5\u05D9\u05EA'),
    #'Hindi' : (u'',u'',u'\u0915\u0941\u0935\u0948\u091F'),
    #'Hmong' : (u'',u'',u'cov neeg'),
    #'Igbo' : (u'',u'',u'ekweisi'),
    #'Kannada' : (u'',u'',u'\u0C95\u0CC1\u0CB5\u0CC8\u0C9F\u0CCD'),
    #'Korean' : (u'',u'',u'\uCFE0\uC6E8\uC774\uD2B8'),
    #'Kurdish' : (u'',u'',u'Kuw\u00EAt'),
    #'Malagasay' : (u'',u'',u'Koety'),
    #'Malagasy' : (u'',u'',u'Koety'),
    #'Marathi' : (u'',u'',u'\u0915\u0941\u0935\u0948\u0924'),
    #'Pashto' : (u'',u'',u'\u06A9\u0648\u06CC\u067C'),
    #'Persian' : (u'',u'',u'\u06A9\u0648\u06CC\u062A'),
    #'Punjabi' : (u'',u'',u'\u0A15\u0A41\u0A35\u0A48\u0A24'),
    #'Russian' : (u'',u'',u'\u041A\u0443\u0432\u0435\u0439\u0442'),
    #'Samoan' : (u'',u'',u'Kuaki'),
    #'Serbian' : (u'',u'',u'\u041A\u0443\u0432\u0430\u0458\u0442'),
    #'Shona' : (u'',u'',u'Kuweti'),
    #'Sindhi' : (u'',u'',u'\u06AA\u0648\u064A\u062A'),
    #'Sinhala' : (u'',u'',u'\u0D9A\u0DD4\u0DC0\u0DDA\u0DA7\u0DBA'),
    #'Somali' : (u'',u'',u'caan ah'),
    #'Tajik' : (u'',u'',u'\u0445\u0438\u0442\u043E\u0431'),
    #'Tamil' : (u'',u'',u'\u0B95\u0BC1\u0BB5\u0BC8\u0BA4\u0BCD'),
    #'Telugu' : (u'',u'',u'\u0C15\u0C41\u0C35\u0C46\u0C56\u0C1F\u0C4D'),
    #'Thai' : (u'',u'',u'\u0E04\u0E39\u0E40\u0E27\u0E15'),
    #'Turkish' : (u'',u'',u'Kuveyt'),
    #'Urdu' : (u'',u'',u'\u06A9\u0648\u06CC\u0648'),
    #'Uzbek' : (u'',u'',u'Quvayt'),
    #'Xhosa' : (u'',u'',u'Khuweyithi'),
#}

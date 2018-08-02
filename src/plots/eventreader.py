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

parser=argparse.ArgumentParser('infer using a model')

parser.add_argument('--models',type=str,nargs='*',required=True)
parser.add_argument('--outdir',type=str,default='img/tables')

args = parser.parse_args()

########################################
print('importing libraries')
import tensorflow as tf
import datetime
import pprint

########################################
print('looping through event files')

results_sum={}
results_tot={}
def update_results(xs,v):
    def local_func(xs,v,retdict):
        if xs[0] in retdict:
            if len(xs)==1:
                if xs[0]=='newuser':
                    retdict['newuser_raw']+=v
                else:
                    retdict[xs[0]]+=v
            else:
                local_func(xs[1:],v,retdict=retdict[xs[0]])
        else:
            if len(xs)==1:
                if xs[0]=='newuser':
                    retdict['newuser_raw']=v
                else:
                    retdict[xs[0]]=v
            else:
                retdict[xs[0]]={}
                local_func(xs[1:],v,retdict=retdict[xs[0]])
    local_func(xs,v,results_sum)
    local_func(xs,1,results_tot)

def get_results(xs):
    def local_func(xs,retdict):
        if len(xs)==1:
            return retdict[xs[0]]
        else:
            return local_func(xs[1:],retdict[xs[0]])
    return local_func(xs,results_sum)/local_func(xs,results_tot)

for modeldir in args.models:
    print('modeldir=',modeldir)
    events_files=os.listdir(modeldir+'/train')
    for events_file in events_files:
        events_file=modeldir+'/train/'+events_file
        i=0
        for event in tf.train.summary_iterator(events_file):
            i+=1
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    #print('tag=',value.tag.split('/'),'; simple_value=',value.simple_value)
                    tags=value.tag.split('/')
                    update_results([tags[0]]+[modeldir]+tags[1:],value.simple_value)

            if i%100==0:
                print(datetime.datetime.now(),'i=',i)

########################################
print('writing tables')

for tag in results_sum.keys():
    #if 'all' in tag or 'filter' in tag:
        #pprint.pprint(results_sum[tag])
        #sys.exit(0)
    try:
        with open(args.outdir+'/'+tag+'.tex','w') as f:
            f.write('''
\\begin{tabular}{l|c|cccccccc}
& average &\multicolumn{8}{c}{accuracy} \\\\
model & distance (km) & @country & @10km & @50km & @100km & @500km & @1000km & @2000km & @3000km \\\\
\\hline
\\hline
''')
            for model in args.models:
                modelname=model.split('/')[-1]
                if modelname=='':
                    modelname=model.split('/')[-2]
                f.write(modelname)
                f.write(' & ')
                results=[]
                for key in ['dist','country_acc']:
                    results.append(get_results([tag,model,key]))
                for key in ['k10','k50','k100','k500','k1000','k2000','k3000']:
                    results.append(1.0-get_results([tag,model,key]))
                f.write(' & '.join(map(lambda x: '%.2f'%x,results)))
                f.write(' \\\\\n ')
            f.write('\\end{tabular}')
    except Exception as e:
        print('failed for tag=',tag, 'error=',e)



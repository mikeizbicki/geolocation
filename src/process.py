#!/usr/bin/env python

from __future__ import print_function
import simplejson as json
from pprint import pprint
import sys
import datetime
import time
import gzip
import geopy.distance
import nltk.tokenize
import numpy as np
import scipy as sp
import math
import tensorflow as tf

import sklearn.feature_extraction.text
hv=sklearn.feature_extraction.text.HashingVectorizer(n_features=2**18,norm=None)

# get better pretty printing of defaultdict
import collections
class defaultdict(collections.defaultdict):
        __repr__ = dict.__repr__

import treetaggerwrapper
tagger=defaultdict(lambda: treetaggerwrapper.TreeTagger(TAGLANG='en'))
tagger['es']=treetaggerwrapper.TreeTagger(TAGLANG='es')

################################################################################
# load the data

files=(sys.argv[1:])
#files=['/data/twitter/geoTwitter17-11-06/geoTwitter17-11-06_14:05']
maxtweets=sys.maxint
maxtweets=1000

datas=[]
numlines=0
for filename in files:
    print(datetime.datetime.now(),filename)
    f=gzip.open(filename,'rt')

    text='start'
    while text != '' and numlines < maxtweets:
        text=f.readline()
        numlines+=1
        try:
            data=json.loads(text)
            format='%a %b %d %H:%M:%S +0000 %Y'
            newdata={}
            newdata['tweet']=data

            newdata['datetime']=datetime.datetime.strptime(data['created_at'],format)
            newdata['lang']=data['lang']
            newdata['userid']=data['user']['id']
            newdata['user_location']=data['user']['location']
            newdata['user_timezone']=data['user']['time_zone']

            newdata['words']=set(data['text'].lower().split())
            newdata['words_hash']=hv.transform([data['text']])

            isHashTag = lambda x: x[0]=='#'
            hashtags=' '.join(filter(isHashTag,newdata['words']))
            newdata['words_hashtags']=hv.transform([hashtags])

            ## add treetaggerwrapper features
            #text=data['text']
            #if isinstance(text,str):
                #text=unicode(text,'utf-8')
            #alltags = treetaggerwrapper.make_tags(tagger[data['lang']].tag_text(text))
            #nottags = [t.what for t in filter(lambda x:isinstance(x,treetaggerwrapper.NotTag),alltags)]
            #tags = filter(lambda x: isinstance(x,treetaggerwrapper.Tag),alltags)
            #lemmas=[tag.lemma for tag in tags]
            #nps=[tag.lemma for tag in filter(lambda x: x.pos==u'NP' and x.lemma[0]!='#',tags)]
            ##print(nps)
            #newdata['words_ttw_lemmas']=hv.transform([' '.join(lemmas)])
            #newdata['words_ttw_nottags']=hv.transform([' '.join(nottags)])

            # add geometric features
            if data['geo']:
                lat=data['geo']['coordinates'][0]
                lon=data['geo']['coordinates'][1]
                newdata['gps']=(lat,lon)
            else:
                list=data['place']['bounding_box']['coordinates']
                coords=[item for sublist in list for item in sublist]
                lats=[coord[0] for coord in coords]
                lons=[coord[1] for coord in coords]
                lat=sum(lats)/float(len(lats))
                lon=sum(lons)/float(len(lons))
                coord=(lat,lon)
                newdata['gps']=coord
            datas.append(newdata)

        except Exception as e:
            #e = sys.exc_info()[0]
            print('error on line', numlines,': ',e)
            #pprint(data)

    f.close()

################################################################################
# some distance functions

def dist_time(t1,t2):
    diff1=t1['datetime']-t2['datetime']
    diff2=t2['datetime']-t1['datetime']
    #print(t1['datetime'])
    #print(t2['datetime'])
    seconds1=diff1.seconds+diff1.days*24*3600
    seconds2=diff2.seconds+diff2.days*24*3600
    rel_hour_of_day1=seconds1%(24*3600)/float(3600)
    rel_hour_of_day2=seconds2%(24*3600)/float(3600)
    rel_day_of_week1=seconds1%(24*3600*7)/float(3600*24)
    rel_day_of_week2=seconds2%(24*3600*7)/float(3600*24)
    seconds=max(seconds1,seconds2)
    rel_hour_of_day=min(rel_hour_of_day1,rel_hour_of_day2)
    rel_day_of_week=min(rel_day_of_week1,rel_day_of_week2)
    #print(seconds1,seconds2,rel_hour_of_day,rel_day_of_week)
    return (seconds,rel_hour_of_day,rel_day_of_week)

def dist_discrete(a,b):
    if a==b:
        return 0
    else:
        return 1

def dist_samelang(t1,t2):
    return dist_discrete(t1['lang'],t2['lang'])

def dist_sameuser(t1,t2):
    return dist_discrete(t1['userid'],t2['userid'])

def dist_bow(t1,t2):
    bow=math.exp(-len(t1['words']&t2['words']))
    hash=sp.sparse.linalg.norm(t1['words_hash']-t2['words_hash'])
    vec=t1['words_hash']-t2['words_hash']
    return (bow,hash,vec)

def dist_gps(t1,t2):
    d1=geopy.distance.vincenty(t1['gps'],t2['gps'])
    d2=geopy.distance.great_circle(t1['gps'],t2['gps'])
    return (d1.miles,d2.miles)

def dist_all(t1,t2):
    return (dist_time(t1,t2),dist_samelang(t1,t2),dist_sameuser(t1,t2),dist_bow(t1,t2),dist_gps(t1,t2))

################################################################################
# tensorflow

def mkSparseTensorValue(m):
    m2=sp.sparse.coo_matrix(m)
    #print('shape=',m2.shape)
    #print('indices=',m.indices)
    #print('indptr=',m.indptr)
    #print('index=',zip(m2.row,m2.col))
    #print('data=',m2.data)
    #sys.exit(0)
    #return tf.SparseTensorValue([[1,0],[2,0],[3,0]],[4,5,6],m.shape)
    return tf.SparseTensorValue(zip(m2.row,m2.col),m2.data,m2.shape)

batch_size=100
cur_data=0
learning_rate=0.005

xdim=int(2**18)
#x_ = tf.sparse_placeholder(tf.float32, np.array([batch_size, xdim],dtype=np.int64))
x_ = tf.sparse_placeholder(tf.float32)
x_.get_shape()
y_ = tf.placeholder(tf.float32, [batch_size, 2])

w = tf.Variable(tf.zeros([xdim, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.sparse_tensor_dense_matmul(x_,w) + b
#y = tf.matmul(tf.sparse_to_dense(x_),w) + b

loss = tf.reduce_sum((y - y_) * (y - y_))
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, name='global_step', trainable=False)
optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.minimize(loss, global_step=global_step)

# prepare logging
#local_log_dir=os.path.join(FLAGS.log_dir_out, '%s-%s.%d-%1.2f-%s.%d-%d'%(FLAGS.dataset,FLAGS.model,FLAGS.seed,FLAGS.induced_bias,FLAGS.same_seed,FLAGS.numproc,FLAGS.procid))
#if tf.gfile.Exists(local_log_dir):
    #tf.gfile.DeleteRecursively(local_log_dir)
#tf.gfile.MakeDirs(local_log_dir)

# create session
sess = tf.Session()
summary = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=1)
summary_writer = tf.summary.FileWriter('log', sess.graph)
sess.run(tf.global_variables_initializer())

print('training')
for step in xrange(1000000):
    start_time = time.time()

    # Fill a feed dictionary with the actual set of images and labels
    # for this particular training step.
    lastdp=len(datas)-len(datas)%batch_size
    start=step*batch_size%lastdp
    stop=start+batch_size
    dps=datas[start:stop]
    #dps=datas[step*batch_size:(step+1)*batch_size]
    feed_dict = {
        x_ : mkSparseTensorValue(sp.sparse.vstack(map(lambda dp: dp['words_hash'],dps))),
        y_ : np.vstack(map(lambda dp: np.array(dp['gps']),dps))
    }

    # Run one step of the model.  The return values are the activations
    # from the `train_op` (which is discarded) and the `loss` Op.  To
    # inspect the values of your Ops or variables, you may include them
    # in the list passed to sess.run() and the value tensors will be
    # returned in the tuple from the call.
    _, metric_value = sess.run([train_op, loss],feed_dict=feed_dict)
    duration = time.time() - start_time

    # Write the summaries and print an overview fairly often.
    if step % 100 == 0:
        print('  step %d: metric = %.2f (%.3f sec)' % (step, metric_value, duration))
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()


    # Save a checkpoint and evaluate the model periodically.
    #if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        #checkpoint_file = os.path.join(local_log_dir, 'model.ckpt')
        #saver.save(sess, checkpoint_file, global_step=step)
#
        #evals.append(do_eval(sess,metric,test_set,FLAGS))


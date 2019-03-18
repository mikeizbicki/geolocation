
########################################
print('processing cmd line args')
import argparse

parser=argparse.ArgumentParser('manually check if s2 cells are good')
parser.add_argument('--tfrecord_input',type=str,default='out_tfrecord_32_2')
args=parser.parse_args()

########################################
print('processing data points')

import tensorflow as tf
record_iterator = tf.python_io.tf_record_iterator(path=args.tfrecord_input)

num_entries=0
for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    gps=list(example.features.feature['train/gps'].float_list.value)
    print('num_entries=',num_entries)
    for i in range(4,17):
        s2file='s2/class_cells-'+str(2**i)
        with open(s2file,'rb') as f:
            import pickle
            s2cells=pickle.load(f)
        s2str='train/s2/'+str(2**i)
        s2cell_id=example.features.feature[s2str].int64_list.value[0]
        s2cell=s2cells[s2cell_id]
        s2cell_gps=[s2cell.to_lat_lng().lat().degrees,s2cell.to_lat_lng().lng().degrees]

        print('  i:',i,'gps=',gps,'s2cell_gps=',s2cell_gps)
    asd

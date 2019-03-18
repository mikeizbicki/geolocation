#!/bin/python

from __future__ import print_function

########################################
print('processing cmd line args')
import argparse

parser=argparse.ArgumentParser('convert gps coordinates to s2 class labels')
parser.add_argument('--input_file',type=str,default='gps.list')
parser.add_argument('--output_prefix',type=str,default='class_cells')
args=parser.parse_args()

########################################
print('importing')
import pickle
import datetime
import heapq
import queue
import s2sphere as s2

########################################
print('loading data points')
with open(args.input_file,'rb') as input_file:
    gps_coords=pickle.load(input_file)

########################################
print('create initial s2 cells')

initial_cells={
    s2.CellId.from_face_pos_level(0,0,0) : [],
    s2.CellId.from_face_pos_level(1,0,0) : [],
    s2.CellId.from_face_pos_level(2,0,0) : [],
    s2.CellId.from_face_pos_level(3,0,0) : [],
    s2.CellId.from_face_pos_level(4,0,0) : [],
    s2.CellId.from_face_pos_level(5,0,0) : [],
    }

num_entries=0
for gps in gps_coords:
    gps_cell=s2.CellId.from_lat_lng(s2.LatLng.from_degrees(gps[0],gps[1]))
    for cell in initial_cells:
        if cell.intersects(gps_cell):
            initial_cells[cell].append(gps_cell)

    if num_entries%1000==0:
        print('%s  num_entries=%d' %
            ( datetime.datetime.now()
            , num_entries
            ))
    num_entries+=1

def print_celldict(celldict):
    total=0
    for cell in celldict:
        length=len(celldict[cell])
        print('  ',cell,' : ',length)
        total+=length
    print('  ','total=',total)

print('initial_cells:')
print_celldict(initial_cells)

########################################
print('splitting s2 cells')

priority_list=[(-len(initial_cells[cell]),cell,initial_cells[cell]) for cell in initial_cells]
#s2cells=heapq.heapify(priority_list)
s2cells=queue.PriorityQueue()
for item in priority_list:
    s2cells.put(item)

sizes=range(6,32)
# sizes=[2**i for i in range(4,22)]
#sizes=[2**i for i in range(4,12)]
#print('sizes=',sizes)
#sizes.pop(0)
#print('sizes=',sizes)

iteration=0
while True:

    # save cell list
    if s2cells.qsize() > sizes[0]:
        print('  saving cell list for size',sizes[0])
        with open(args.output_prefix+'-'+str(sizes[0]),'wb') as output_file:
            cellids = [ cell for (priority,cell,gps_cells) in list(s2cells.queue) ]
            #print('cellids=',cellids)
            pickle.dump(cellids,output_file)
        sizes.pop(0)

    # end when no more sizes
    if sizes == []:
        break

    # split next cell
    (neglen,cell,gps_cells)=s2cells.get()
    try:
        children=dict([(c,[]) for c in cell.children()])
        if cell.level() >= 22:
            s2cells.put((0,cell,gps_cells))
            continue
        for gps_cell in gps_cells:
            for child in children:
                if child.intersects(gps_cell):
                    children[child].append(gps_cell)
        #print_celldict(children)
        for child in children:
            s2cells.put((-len(children[child]),child,children[child]))
    except AssertionError:
        print('cell=',cell,' is a leaf; len(gps_cells)=',len(gps_cells))
        s2cells.put((1,cell,gps_cells))

    # print debug messages
    if iteration%100==0:
        print('%s  iteration=%d  s2cells.qsize=%d' %
            ( datetime.datetime.now()
            , iteration
            , s2cells.qsize()
            ))
    iteration+=1

#print('iteration=',iteration)
#print_celldict(dict([(cell,gps_cells) for (priority,cell,gps_cells) in list(s2cells.queue)]))

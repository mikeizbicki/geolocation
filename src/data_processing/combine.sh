#!/bin/bash

. /rhome/mizbicki/tf-cpu-1.4/bin/activate
python src/data_processing/combine.py --files data/BillionTwitter/geoTwitter1*.pkl --outfile=data/BillionTwitter/summary.pkl

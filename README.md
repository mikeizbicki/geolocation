# Geolocation library for TensorFlow

This repository contains source code for the paper [Exploiting the Earth's Spherical Geometry to Geolocate Images](https://izbicki.me/public/papers/XXX) and a working paper [Geolocating Tweets Sent in Any Language form Any Location ](https://izbicki.me/public/papers/XXX).

The file `src/image/gps_loss.py` contains a function `gps_loss`,
which contains the code for the MvMF layer
(and has options for many variations on the layer described in the paper).
This function can be added to your TensorFlow project in order to predict GPS coordinates.
All code was developed using TensorFlow v1.4.

For information on training a model, run the command
```
python ./src/image/train.py --help
```

For information on using a model to predict GPS locations, run the command
```
python ./src/image/infer.py --help
```

Documentation for this code is currently fairly sparse.
If you have any questions, or want help performing a particular task, please email me at [mike@izbicki.me](mailto:mike@izbicki.me).

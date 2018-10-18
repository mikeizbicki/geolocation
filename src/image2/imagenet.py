
import tensorflow as tf
import tensornets as nets

inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
model = nets.ResNet50(inputs)

assert isinstance(model, tf.Tensor)

file='/rhome/mizbicki/bigdata/geolocating/data/flickr/img2/aaa/'
file='/rhome/mizbicki/bigdata/geolocating/data/flickr/img2/aaa/6984331957_aaaa5a3503.jpg'
#file='/rhome/mizbicki/bigdata/geolocating/data/flickr/img2/aaa/82374795_aaabbd3fae.jpg'
#file='/rhome/mizbicki/bigdata/geolocating/data/flickr/img2/aaa/86653640_aaa9b79a4b.jpg'
#file='/rhome/mizbicki/bigdata/geolocating/data/flickr/img2/aaa/88958607_aaa7de7c89.jpg'

img = nets.utils.load_img(file, target_size=256, crop_size=224)
assert img.shape == (1, 224, 224, 3)

with tf.Session() as sess:
    img = model.preprocess(img)  # equivalent to img = nets.preprocess(model, img)
    sess.run(model.pretrained())  # equivalent to nets.pretrained(model)
    preds = sess.run(model, {inputs: img})
    #print('preds=',preds)
    #print('preds=',nets.utils.decode_predictions(preds, top=2)[0])
    print('model=',model)
    print('')

    model = nets.YOLOv3(inputs, nets.Darknet19)
    sess.run(model.pretrained())
    preds = sess.run(model, {inputs: model.preprocess(img)})
    boxes = model.get_boxes(preds, img.shape[1:3])
    print('model=',model)
    #print('preds=',preds)
    #print('preds=',nets.utils.decode_predictions(preds, top=2)[0])
    #print('boxes=',boxes)
    #print('preds=',nets.utils.decode_predictions(preds, top=2)[0])


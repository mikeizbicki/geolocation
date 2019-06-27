from __future__ import print_function

def image10(args,image):
    import tensorflow as tf

    with tf.name_scope('image10'):
        if image.dtype != tf.float32:
          image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        new_image_size=int(args.image_size*1.2)
        image = tf.expand_dims(image, 0)
        image=tf.image.resize_bilinear(image, [new_image_size,new_image_size])
        image=tf.reshape(image,[new_image_size,new_image_size,3])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        image_flip=tf.image.flip_left_right(image)

        images=[]
        for image in [image,image_flip]:
            offset=new_image_size-args.image_size
            images.append(tf.image.crop_to_bounding_box(image,0,0,args.image_size,args.image_size))
            images.append(tf.image.crop_to_bounding_box(image,0,offset,args.image_size,args.image_size))
            images.append(tf.image.crop_to_bounding_box(image,offset,0,args.image_size,args.image_size))
            images.append(tf.image.crop_to_bounding_box(image,offset,offset,args.image_size,args.image_size))
            images.append(tf.image.crop_to_bounding_box(image,offset/2,offset/2,args.image_size,args.image_size))

        images=tf.stack(images,axis=0)
        print('images=',images)
        return images


    ## Crop the central region of the image with an area containing 87.5% of
    ## the original image.
    #if central_fraction:
      #image = tf.image.central_crop(image, central_fraction=central_fraction)
    #if height and width:
      ## Resize the image to the specified height and width.
      #image = tf.expand_dims(image, 0)
      #image = tf.image.resize_bilinear(image, [height, width],
                                       #align_corners=False)
      #image = tf.squeeze(image, [0])

################################################################################

def mkDataset_tfrecord_map(args,files,is_training=True):
    import tensorflow as tf

    s2str='train/s2/'+str(2**args.s2size)
    def _parse_function(example_proto):
        features = {
            #'filename': tf.FixedLenFeature([],tf.string),
            'train/gps': tf.FixedLenFeature([2],tf.float32),
            'train/country': tf.FixedLenFeature([],tf.int64),
            'train/features': tf.FixedLenFeature([2049],tf.float32),
            s2str : tf.FixedLenFeature([],tf.int64),
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        return (
            #parsed_features['filename'],
            parsed_features['train/gps'],
            parsed_features['train/country'],
            #tf.cast(parsed_features['train/country'],tf.int64),
            parsed_features['train/features'],
            parsed_features[s2str],
            )

    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(_parse_function)

    # generate data tensors
    #if is_training:
        #dataset=dataset.shuffle(args.batchsize*args.shufflemul,seed=args.seed)
    #dataset=dataset.batch(args.batchsize)
    #dataset=dataset.prefetch(1)
    iter = dataset.make_initializable_iterator()
    return iter

def mkDataset_tfrecord_infer(args,files,is_training=True):
    import tensorflow as tf

    s2str='train/s2/'+str(2**args.s2size)
    def _parse_function(example_proto):
        features = {
            'train/gps': tf.FixedLenFeature([2],tf.float32),
            'train/country': tf.FixedLenFeature([],tf.int64),
            'train/features': tf.FixedLenFeature([2049],tf.float32),
            s2str : tf.FixedLenFeature([],tf.int64),
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        return (
            parsed_features['train/gps'],
            parsed_features['train/country'],
            #tf.cast(parsed_features['train/country'],tf.int64),
            parsed_features['train/features'],
            parsed_features[s2str],
            )

    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(_parse_function)

    # generate data tensors
    if is_training:
        dataset=dataset.shuffle(args.batchsize*args.shufflemul,seed=args.seed)
    dataset=dataset.take(10000)
    dataset=dataset.batch(args.batchsize)
    dataset=dataset.prefetch(1)
    iter = dataset.make_initializable_iterator()
    return iter

def mkDataset_tfrecord_features(args,files,is_training=True):
    import tensorflow as tf

    s2str='train/s2/'+str(2**args.s2size)
    def _parse_function(example_proto):
        features = {
            'train/gps': tf.FixedLenFeature([2],tf.float32),
            'train/country': tf.FixedLenFeature([],tf.int64),
            'train/features': tf.FixedLenFeature([2049],tf.float32),
            s2str : tf.FixedLenFeature([],tf.int64),
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        return (
            parsed_features['train/gps'],
            parsed_features['train/country'],
            #tf.cast(parsed_features['train/country'],tf.int64),
            parsed_features['train/features'],
            parsed_features[s2str],
            )

    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(_parse_function)

    # generate data tensors
    if is_training:
        dataset=dataset.shuffle(args.batchsize*args.shufflemul,seed=args.seed)
    dataset=dataset.batch(args.batchsize)
    dataset=dataset.prefetch(1)
    iter = dataset.make_initializable_iterator()
    return iter


def mkDataset_jpg(args,files,is_training):
    import tensorflow as tf

    with tf.variable_scope('input'):
        # select files
        dataset_files=tf.data.Dataset.from_tensor_slices(files)
        if is_training:
            dataset_files=dataset_files.interleave(tf.data.Dataset.list_files,cycle_length=args.cyclelength)
            dataset_files=dataset_files.shuffle(args.cyclelength*3,seed=args.seed)
        else:
            dataset_files=dataset_files.interleave(tf.data.Dataset.list_files,cycle_length=1)

        dataset_files=dataset_files.prefetch(args.cyclelength)

        # extract gps info from images
        import data
        dataset_labels=dataset_files.map(data.tf_imgpath2labels)

        # extract images
        dataset_images=dataset_files.map(tf.read_file)
        dataset_images=dataset_images.map(tf.image.decode_image)

        # filter out files with bad gps/image data
        dataset=tf.data.Dataset.zip((dataset_files,dataset_labels,dataset_images))
        filter_fn=lambda f,(g,c),i: tf.logical_and(
            tf.logical_not(tf.reduce_any(tf.is_nan(g))),
            tf.equal(tf.shape(i)[2],3),
            )
        dataset=dataset.filter(filter_fn)

        # preprocess images
        from preprocessing import preprocessing_factory
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            'inception',
            is_training=is_training)

        #dataset=dataset.map(lambda f,l,i: (f,l,image_preprocessing_fn(i,args.image_size,args.image_size)),num_parallel_calls=8)
        if is_training:
            dataset=dataset.map(lambda f,l,i: (f,l,image_preprocessing_fn(i,args.image_size,args.image_size)),num_parallel_calls=8)
            #pass
        else:
            if args.num_crops==10:
                dataset=dataset.map(lambda f,l,i: (f,l,image10(args,i)),num_parallel_calls=8)
            else:
                dataset=dataset.map(lambda f,l,i: (f,l,image10(args,i)),num_parallel_calls=8)
                #dataset=dataset.map(lambda f,l,i: (f,l,image_preprocessing_fn(i,args.image_size,args.image_size)),num_parallel_calls=8)

        # generate data tensors
        if is_training:
            dataset=dataset.shuffle(args.batchsize*args.shufflemul,seed=args.seed)
        dataset=dataset.batch(args.batchsize)
        dataset=dataset.prefetch(1)
        iter=dataset.make_initializable_iterator()

    return iter

################################################################################

def mkModel(args,image_,country_,gps_,is_training,gmm_log2=False,features=None,s2_=None):
    import tensorflow as tf
    import tensornets as nets
    import data

    # 1-hot encode country
    with tf.variable_scope('country_1hot_'):
        country_1hot_=tf.one_hot(country_,data.num_countries)
        country_1hot_=tf.reshape(country_1hot_,[-1,data.num_countries])

    # create features
    if features is None:
        with tf.variable_scope('features'): #,reuse=tf.AUTO_REUSE):
            features_list=[]
            #for i in [0]: #range(0,5): #int(image_.get_shape()[1])):
            for i in range(0,int(image_.get_shape()[1])):
            #for i in [0,1,2,3,4]:
                with tf.variable_scope('empty_features'):
                    features=tf.reduce_sum(gps_,axis=1)*0+1
                    features=tf.reshape(features,[-1,1])
                    #batchsize=args.batchsize
                    #print('features=',features)
                    #features=tf.reshape(tf.tile(tf.constant([1.0]),[batchsize]),[batchsize,1])

                if (not args.model is None) and (not image_ is None):
                    net_fn = eval('nets.'+args.model)
                    net = net_fn(
                        image_[:,i,:,:,:],
                        #tf.Print(image_[:,i,:,:,:],[image_[:,i,:,:,:]],'image_'+str(i)),
                        is_training=is_training,
                        scope='wideresnet50'+('_'+str(i) if not is_training else ''),
                        reuse=False,
                        #reuse=tf.AUTO_REUSE,
                        )

                    if 'middles_all' in args.inputs:
                        with tf.variable_scope('middles'):
                            middles=map(tf.contrib.layers.flatten,net.get_middles())
                            features=tf.concat([features]+middles,axis=1)

                    if 'middles_last' in args.inputs:
                        with tf.variable_scope('middles_last'):
                            middles_last=net.get_middles()[-1]
                            print('middles_last=',middles_last)
                            #features=tf.Print(features,[middles_last],'features_'+str(i))
                            pooled = tf.reduce_mean(middles_last, [1, 2], name='avgpool')
                            #pooled = tf.reduce_mean(middles_last, [2, 3], name='avgpool')
                            print('features=',features)
                            print('pooled=',pooled)
                            features=tf.concat([features,pooled],axis=1)
                            print('features=',features)

                    if 'model' in args.inputs:
                        with tf.variable_scope('net'):
                            features=tf.concat([features,net],axis=1)

                    features_list.append(features)
                    features=None
            #features=features_list[4]
            features=tf.reduce_mean(tf.stack(features_list,axis=0),axis=0)
            #features=tf.Print(features,features_list)

    else:
        net=None

    # create losses
    loss=0.0
    op_metrics={}

    l2gps=0
    if 'gps' in args.outputs:
        import gps_loss
        print('features=',features)
        loss_gps,op_regularizers,op_metrics,op_endpoints = gps_loss.gps_loss(
            features,
            gps_,
            pos_type='aglm_mix',
            trainable_mu=args.trainable_mu,
            trainable_kappa=args.trainable_kappa,
            trainable_weights=args.trainable_weights,
            gmm_distribution=args.gmm_distribution,
            gmm_xentropy=args.gmm_xentropy,
            gmm_minimizedist=args.gmm_minimizedist,
            gmm_logloss=not args.gmm_no_logloss,
            gmm_log2=gmm_log2,
            gmm_components=args.gmm_components,
            gmm_prekappa0=args.lores_gmm_prekappa0,
            s2warmstart_mu=args.s2warmstart_mu,
            s2warmstart_kappa=args.s2warmstart_kappa,
            s2warmstart_kappa_s=args.s2warmstart_kappa_s,
            gradient_method=args.gmm_gradient_method,
            )
        op_metrics['optimization/loss_gps']=loss_gps
        loss+=loss_gps
        l2gps=args.l2gps*sum(op_regularizers.values())

    if 'gps2' in args.outputs:
        import gps_loss
        loss_gps,op_regularizers,op_metrics,op_endpoints = gps_loss.gps_loss_layered(
            features,
            gps_,
            gmm_type=args.gmm_type,
            trainable_mu=args.trainable_mu,
            trainable_kappa=args.trainable_kappa,
            trainable_weights=args.trainable_weights,
            lores_gmm_prekappa0=args.lores_gmm_prekappa0,
            gmm_minimizedist=args.gmm_minimizedist,
            gmm_xentropy=args.gmm_xentropy,
            gmm_logloss=not args.gmm_no_logloss,
            gmm_log2=gmm_log2,
            hires_concat_endpoints=True,
            hires_gmm_prekappa0=args.hires_gmm_prekappa0,
            )
        op_metrics['optimization/loss_gps']=loss_gps
        loss+=loss_gps
        for k,v in op_regularizers.iteritems():
            op_metrics['regularizer/'+k]=v
            op_metrics['regularizer/'+k+'_weighted']=args.l2gps*v
        l2gps=args.l2gps*sum(op_regularizers.values())

    if 'gps2b' in args.outputs:
        import gps_loss
        loss_gps,op_regularizers,op_metrics,op_endpoints = gps_loss.gps_loss_layered(
            features,
            gps_,
            gmm_type=args.gmm_type,
            trainable_mu=args.trainable_mu,
            trainable_kappa=args.trainable_kappa,
            trainable_weights=args.trainable_weights,
            lores_gmm_prekappa0=args.lores_gmm_prekappa0,
            gmm_minimizedist=args.gmm_minimizedist,
            gmm_xentropy=args.gmm_xentropy,
            gmm_logloss=not args.gmm_no_logloss,
            gmm_log2=gmm_log2,
            #lores_device='GPU:1',
            #hires_device='GPU:1',
            hires_gmm_components=2**17,
            hires_gmm_sparsity=None, #2**16,
            hires_gmm_decomposed=[],
            hires_gmm_prekappa0=args.hires_gmm_prekappa0,
            )
        op_metrics['optimization/loss_gps']=loss_gps
        loss+=loss_gps
        for k,v in op_regularizers.iteritems():
            op_metrics['regularizer/'+k]=v
            op_metrics['regularizer/'+k+'_weighted']=args.l2gps*v
        l2gps=args.l2gps*sum(op_regularizers.values())

    if 'country' in args.outputs:
        with tf.variable_scope('loss_country'):
            logits = tf.contrib.layers.fully_connected(features,data.num_countries)
            xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=country_1hot_,logits=logits)
            loss_country = tf.reduce_mean(xentropy)
            country = tf.argmax(logits,axis=1)
            country_acc = tf.reduce_mean(tf.cast(tf.equal(country,country_),tf.float32))
            op_metrics['optimization/loss_country']=loss_country
            op_metrics['optimization/acc_country']=country_acc
            loss+=loss_country

    if 's2' in args.outputs:
        import s2sphere
        import pickle
        import numpy as np
        import math

        s2file='s2/class_cells-'+str(2**args.s2size)
        with open(s2file,'rb') as f:
            s2cells=pickle.load(f)
            max_cells=len(s2cells)
            s2cells=zip(s2cells,range(0,max_cells))

        # extract true s2 cell id
        with tf.variable_scope('s2_'):
            if s2_ is None:
                def gps2cellid(gps):
                    gps=np.float64(gps)
                    gps_cell=s2sphere.CellId.from_lat_lng(s2sphere.LatLng.from_degrees(gps[0],gps[1]))
                    def heap_sort(unsorted):
                        import heapq
                        unsorted = unsorted[:]
                        def heap_sort_destructive(items):
                            heapq.heapify(items)
                            while items:
                                yield heapq.heappop(items)
                        return heap_sort_destructive(unsorted)
                    s2cells_ordered=heap_sort([(abs(cell.id()-gps_cell.id()),cell,id) for (cell,id) in s2cells])
                    for (dist,cell,id) in s2cells_ordered:
                        if cell.intersects(gps_cell):
                            return id
                    return max_cells+1
                s2_=tf.map_fn(
                    lambda x: tf.py_func(gps2cellid,[x],tf.int64,stateful=False),
                    gps_,
                    dtype=tf.int64
                    )
            s2_1hot_=tf.one_hot(s2_,max_cells)
            s2_1hot_=tf.reshape(s2_1hot_,[-1,max_cells])

        # xentropy loss for s2 cell id
        with tf.variable_scope('loss_s2'):
            logits = tf.contrib.layers.fully_connected(features,max_cells) #,weights_initializer=tf.zeros_initializer())
            xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=s2_1hot_,logits=logits)
            loss_s2 = tf.reduce_mean(xentropy)
            s2 = tf.argmax(logits,axis=1)
            s2_acc = tf.reduce_mean(tf.cast(tf.equal(s2,s2_),tf.float32))
            op_metrics['optimization/s2_loss']=loss_s2
            op_metrics['optimization/s2_acc']=s2_acc
            loss+=loss_s2

            with tf.variable_scope('distance'):
                s2cells_gps=map(lambda (x,y): [x.to_lat_lng().lat().degrees,x.to_lat_lng().lng().degrees],s2cells)
                s2cells_gps=np.array(s2cells_gps,dtype=np.float32)

                gps=tf.gather(s2cells_gps,s2)
                op_lat_rad_ = gps_[:,0]/360*2*math.pi
                op_lon_rad_ = gps_[:,1]/360*2*math.pi
                op_lat_rad = gps[:,0]/360*2*math.pi
                op_lon_rad = gps[:,1]/360*2*math.pi
                hav = lambda x: tf.sin(x/2)**2
                squared_angular_dist = ( hav(op_lat_rad-op_lat_rad_)
                    +tf.cos(op_lat_rad)*tf.cos(op_lat_rad_)*hav(op_lon_rad-op_lon_rad_)
                    )
                s2_dist = 2*6371*tf.asin(tf.sqrt(tf.maximum(0.0,squared_angular_dist)))
                #s2_dist = 6371*tf.sqrt(tf.maximum(0.0,squared_angular_dist))
                s2_dist_ave = tf.reduce_mean(s2_dist)
                op_metrics['gps_loss/dist']=s2_dist_ave

                op_endpoints={
                    'gps':gps,
                    'dist':s2_dist,
                    'mixture':tf.nn.softmax(logits),
                }

                def mk_threshold(threshold):
                    op_threshold = tf.sign(s2_dist-threshold)/2+0.5
                    op_threshold_ave = tf.reduce_mean(op_threshold)
                    name='gps_loss/k'+str(threshold)
                    op_metrics[name]=op_threshold_ave

                if not gmm_log2:
                    mk_threshold(10)
                    mk_threshold(50)
                    mk_threshold(100)
                    mk_threshold(500)
                    mk_threshold(1000)
                    mk_threshold(2000)
                    mk_threshold(3000)
                else:
                    mk_threshold(1)
                    mk_threshold(25)
                    mk_threshold(200)
                    mk_threshold(750)
                    mk_threshold(2500)


    # add regularizers
    with tf.variable_scope('loss_regularized'):
        regularizers=[]
        with tf.variable_scope('l2_regularization'):
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                regularizers.append(args.l2*tf.nn.l2_loss(var))
        loss_l2=loss+tf.reduce_sum(regularizers)
        op_metrics['optimization/loss_l2']=loss_l2

        loss_regularized=loss_l2+l2gps
        op_metrics['optimization/loss_regularized']=loss_regularized

    return net,features,loss,loss_regularized,op_metrics,op_endpoints

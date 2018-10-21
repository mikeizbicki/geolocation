image_size = 224

def image10(image):
    import tensorflow as tf

    with tf.name_scope('image10'):
        if image.dtype != tf.float32:
          image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        new_image_size=int(image_size*1.2)
        image = tf.expand_dims(image, 0)
        image=tf.image.resize_bilinear(image, [new_image_size,new_image_size])
        image=tf.reshape(image,[new_image_size,new_image_size,3])
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        image_flip=tf.image.flip_left_right(image)

        images=[]
        for image in [image,image_flip]:
            offset=new_image_size-image_size
            images.append(tf.image.crop_to_bounding_box(image,0,0,image_size,image_size))
            images.append(tf.image.crop_to_bounding_box(image,0,offset,image_size,image_size))
            images.append(tf.image.crop_to_bounding_box(image,offset,0,image_size,image_size))
            images.append(tf.image.crop_to_bounding_box(image,offset,offset,image_size,image_size))
            images.append(tf.image.crop_to_bounding_box(image,offset/2,offset/2,image_size,image_size))

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

def mkDataset(args,files,is_training):
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
        dataset=dataset.map(lambda f,l,i: (f,l,image_preprocessing_fn(i,image_size,image_size)),num_parallel_calls=8)
        #dataset=dataset.map(lambda f,l,i: (f,l,image10(i)),num_parallel_calls=8)

        # generate data tensors
        if is_training:
            dataset=dataset.shuffle(args.batchsize*args.shufflemul,seed=args.seed)
        dataset=dataset.batch(args.batchsize)
        dataset=dataset.prefetch(1)
        iter=dataset.make_initializable_iterator()

    return iter

################################################################################

def mkModel(args,image_,country_,gps_,is_training,gmm_log2=False):
    import tensorflow as tf
    import tensornets as nets
    import data

    # create features
    with tf.variable_scope('features'):

        with tf.variable_scope('country_1hot_'):
            country_1hot_=tf.one_hot(country_,data.num_countries)
            country_1hot_=tf.reshape(country_1hot_,[-1,data.num_countries])

        with tf.variable_scope('empty_features'):
            batchsize=tf.shape(image_)[0]
            features=tf.reshape(tf.tile(tf.constant([1.0]),[batchsize]),[batchsize,1])

        if not args.model is None:
            net_fn = eval('nets.'+args.model)
            net = net_fn(
                image_,
                is_training=True,
                )

        if 'middles_all' in args.inputs:
            with tf.variable_scope('middles'):
                middles=map(tf.contrib.layers.flatten,net.get_middles())
                features=tf.concat([features]+middles,axis=1)

        if 'middles_last' in args.inputs:
            with tf.variable_scope('middles_last'):
                middles_last=net.get_middles()[-1]
                pooled = tf.reduce_mean(middles_last, [1, 2], name='avgpool')
                features=tf.concat([features,pooled],axis=1)

        if 'model' in args.inputs:
            with tf.variable_scope('net'):
                features=tf.concat([features,net],axis=1)

        #if 'yolo' in args.inputs:
            #with tf.variable_scope('yolo'):


    #if 'cnn' in args.inputs:
        #pooled = tf.reduce_mean(net, [1, 2], name='avgpool')
        #features = tf.concat([pooled,features],axis=1)

    # create losses
    loss=0.0
    op_metrics={}

    if 'gps' in args.outputs:
        import gps_loss
        loss_gps,op_regularizers,op_metrics,op_endpoints = gps_loss.gps_loss(
            features,
            gps_,
            pos_type='aglm_mix',
            trainable_mu=args.trainable_mu,
            trainable_kappa=args.trainable_kappa,
            trainable_weights=args.trainable_weights,
            gmm_minimizedist=args.gmm_minimizedist,
            gmm_log2=gmm_log2,
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

    return net,loss,loss_regularized,op_metrics

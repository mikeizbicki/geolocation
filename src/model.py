import math
import city_loc
import hash

tweetlen=280

################################################################################

def update_parser(parser):
    parser.add_argument('--batchsize',type=int,default=100)
    parser.add_argument('--learningrate',type=float,default=0.005)
    parser.add_argument('--optimizer',choices=['adam','sgd'],default='adam')
    parser.add_argument('--momentum',type=float,default=0.9)
    parser.add_argument('--decay',type=float,default=None)
    parser.add_argument('--dropout',type=float,default=0.5)
    parser.add_argument('--l1',type=float,default=0.0)
    parser.add_argument('--l2',type=float,default=1e-5)

    parser.add_argument('--input',choices=['cnn','bow','lang','time','const'],nargs='+',required=True)
    parser.add_argument('--bow_hashsize',type=int,default=20)
    parser.add_argument('--bow_layersize',type=int,default=2)
    parser.add_argument('--bow_dense',action='store_true')
    parser.add_argument('--cnn_type',choices=['vdcnn','cltcc'],default='cltcc')
    parser.add_argument('--cnn_vocabsize',type=int,default=128)
    parser.add_argument('--cnn_khot',type=int,default=1)
    parser.add_argument('--vdcnn_numfilters',type=int,default=64)
    parser.add_argument('--vdcnn_size',type=int,default=0)
    parser.add_argument('--vdcnn_resnet',action='store_true')
    parser.add_argument('--vdcnn_no_bn',action='store_true')
    parser.add_argument('--cltcc_numfilters',type=int,default=1024)
    parser.add_argument('--cltcc_variance',type=float,default=0.02)

    parser.add_argument('--full',type=int,nargs='*',default=[])

    parser.add_argument('--output',choices=['pos','country','loc'],default=['pos','country','loc'],nargs='+')
    parser.add_argument('--loss_weights',choices=['auto','ave','manual','manual2','manual3','prod'],default='manual')
    parser.add_argument('--loss_staircase',action='store_true')
    parser.add_argument('--pos_type',choices=['naive','aglm','aglm2','aglm_mix','proj3d'],default='aglm')
    parser.add_argument('--pos_loss',choices=['l2','chord','dist','dist_sqrt','dist2','angular'],default='dist')
    parser.add_argument('--pos_shortcut',choices=['loc','country'],default=[],nargs='*')
    parser.add_argument('--pos_warmstart',type=bool,default=True)
    parser.add_argument('--gmm_type',choices=['simple','complex'],default='complex')
    parser.add_argument('--gmm_notrain',action='store_true')
    parser.add_argument('--gmm_components',type=int,default=1)
    parser.add_argument('--country_shortcut',choices=['bow','lang'],default=[],nargs='*')
    parser.add_argument('--loc_type',choices=['popular','hash'],default='hash')
    parser.add_argument('--loc_filter',action='store_true')
    parser.add_argument('--loc_hashsize',type=int,default=14)
    parser.add_argument('--loc_bottleneck',type=int,default=256)
    parser.add_argument('--loc_shortcut',choices=['bow','lang','country'],default=[],nargs='*')

################################################################################

def inference(args,input_tensors):
    import tensorflow as tf

    # helper for initializing variables consistently
    def var_init(shape,stddev=None):
        import tensorflow as tf
        var_init.count+=1
        size=float(sum(shape))
        if stddev is None:
            stddev=math.sqrt(2.0/size)
        return tf.truncated_normal(shape,stddev=stddev,seed=args.seed)
        return tf.truncated_normal(shape,stddev=stddev,seed=args.seed+var_init.count)
    var_init.count=0

    # tf inputs
    with tf.name_scope('inputs'):

        regularizers=[]
        inputs=[]

        # hash bow inputs
        if 'bow' in args.input:
            with tf.name_scope('bow'):
                bow_size=2**args.bow_hashsize
                if args.bow_dense:
                    hash_ = tf.placeholder(tf.float32,[args.batchsize,bow_size],'hash_')
                    matmul = tf.matmul
                    hash_reg=args.l1*tf.reduce_sum(tf.abs(hash_))
                else:
                    hash_ = tf.sparse_placeholder(tf.float32,name='hash_')
                    matmul = tf.sparse_tensor_dense_matmul
                    hash_reg=args.l1*tf.sparse_reduce_sum(tf.abs(hash_))
                regularizers.append(hash_reg)
                w = tf.Variable(var_init([bow_size,args.bow_layersize],1.0),name='w')
                b = tf.constant(0.1,shape=[args.bow_layersize])
                bow = matmul(hash_,w)+b
                inputs.append(bow)

        # cnn inputs
        if 'cnn' in args.input:
            text_reshaped = tf.reshape(input_tensors['text_'],[args.batchsize,tweetlen,args.cnn_vocabsize,1])

            # Very Deep Convolutional Neural Network
            # follows paper "very deep convolutional networks for text classification"
            if 'vdcnn' == args.cnn_type:
                s=tweetlen
                with tf.name_scope('vdcnn'):
                    def mk_conv(prev,numin,numout,swapdim=False):
                        mk_conv.count+=1
                        with tf.name_scope('conv'+str(mk_conv.count)):
                            if swapdim:
                                shape=[3,1,numin,numout]
                                padding='SAME'
                            else:
                                shape=[3,numin,1,numout]
                                padding='VALID'
                            w = tf.Variable(var_init(shape,0.1),name='w')
                            b = tf.Variable(tf.constant(0.1,shape=[numout]),name='b')
                            conv = tf.nn.conv2d(prev, w, strides=[1,1,1,1], padding=padding)
                            return tf.nn.bias_add(conv,b)
                    mk_conv.count=0

                    def mk_conv_block(input,numin,numout,size=2):
                        net=input
                        with tf.name_scope('conv_block'):
                            for i in range(0,size):
                                net = mk_conv(net,numin,numout,swapdim=True)
                                if not args.vdcnn_no_bn:
                                    net = tf.layers.batch_normalization(net,axis=1,training=True)
                                net = tf.nn.relu(net)
                                numin=numout
                            if args.vdcnn_resnet:
                                paddims=np.zeros([4,2])
                                for i in range(0,4):
                                    paddims[i][0]=0
                                    diff=abs(int(net.get_shape()[i])-int(input.get_shape()[i]))
                                    paddims[i][1]=diff
                                input2=tf.pad(input,paddims)
                                return net+input2
                            else:
                                return net

                    def pool2(prev):
                        return tf.nn.max_pool(
                            prev,
                            ksize=[1, 3, 1, 1],
                            strides=[1, 2, 1, 1],
                            padding='SAME')

                    net = mk_conv(text_reshaped,args.cnn_vocabsize,args.vdcnn_numfilters)
                    net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                    if args.vdcnn_size>=1:
                        net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                        net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                    if args.vdcnn_size>=2:
                        net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                        net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                        net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                        net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                        net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                        net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                    if args.vdcnn_size>=3:
                        net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                        net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                        net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                        net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                        net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                        net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters)
                    net = pool2(net)

                    net = mk_conv_block(net,args.vdcnn_numfilters,args.vdcnn_numfilters*2)
                    net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                    if args.vdcnn_size>=1:
                        net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                        net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                    if args.vdcnn_size>=2:
                        net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                        net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                        net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                        net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                        net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                        net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                    if args.vdcnn_size>=3:
                        net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                        net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                        net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                        net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                        net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                        net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*2)
                    net = pool2(net)

                    net = mk_conv_block(net,args.vdcnn_numfilters*2,args.vdcnn_numfilters*4)
                    net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                    if args.vdcnn_size>=1:
                        net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                        net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                    if args.vdcnn_size>=3:
                        net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                        net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                        net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                        net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                        net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                        net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*4)
                    net = pool2(net)

                    net = mk_conv_block(net,args.vdcnn_numfilters*4,args.vdcnn_numfilters*8)
                    net = mk_conv_block(net,args.vdcnn_numfilters*8,args.vdcnn_numfilters*8)
                    if args.vdcnn_size>=1:
                        net = mk_conv_block(net,args.vdcnn_numfilters*8,args.vdcnn_numfilters*8)
                        net = mk_conv_block(net,args.vdcnn_numfilters*8,args.vdcnn_numfilters*8)
                    if args.vdcnn_size>=3:
                        net = mk_conv_block(net,args.vdcnn_numfilters*8,args.vdcnn_numfilters*8)
                        net = mk_conv_block(net,args.vdcnn_numfilters*8,args.vdcnn_numfilters*8)
                    net = pool2(net)

                    input_size=int(net.get_shape()[1]*net.get_shape()[2]*net.get_shape()[3])
                    input_=tf.reshape(net,[args.batchsize,input_size])
                    inputs.append(input_)

            # follows paper "character level convnets for text classification"
            # see also: Language-Independent Twitter Classification Using Character-Based Convolutional Networks
            if 'cltcc' == args.cnn_type:
                activation=tf.nn.relu
                with tf.name_scope('cltcc'):
                    filterlen=7
                    with tf.name_scope('conv1'):
                        w = tf.Variable(var_init([filterlen,args.cnn_vocabsize,1,args.cltcc_numfilters],args.cltcc_variance),name='w')
                        b = tf.Variable(tf.constant(0.1,shape=[args.cltcc_numfilters]),name='b')
                        conv = tf.nn.conv2d(text_reshaped, w, strides=[1,1,1,1], padding='VALID')
                        h = activation(tf.nn.bias_add(conv,b))
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, 3, 1, 1],
                            strides=[1, 3, 1, 1],
                            padding='VALID')

                    with tf.name_scope('conv2'):
                        w = tf.Variable(var_init([filterlen,1,args.cltcc_numfilters,args.cltcc_numfilters],args.cltcc_variance),name='w')
                        b = tf.Variable(tf.constant(0.1,shape=[args.cltcc_numfilters]),name='b')
                        conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
                        h = activation(tf.nn.bias_add(conv,b))
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, 3, 1, 1],
                            strides=[1, 3, 1, 1],
                            padding='VALID')

                    filterlen=3
                    with tf.name_scope('conv3'):
                        w = tf.Variable(var_init([filterlen,1,args.cltcc_numfilters,args.cltcc_numfilters],args.cltcc_variance),name='w')
                        b = tf.Variable(tf.constant(0.1,shape=[args.cltcc_numfilters]),name='b')
                        conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
                        h = activation(tf.nn.bias_add(conv,b))
                        pooled = h

                    with tf.name_scope('conv4'):
                        w = tf.Variable(var_init([filterlen,1,args.cltcc_numfilters,args.cltcc_numfilters],args.cltcc_variance),name='w')
                        b = tf.Variable(tf.constant(0.1,shape=[args.cltcc_numfilters]),name='b')
                        conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
                        h = activation(tf.nn.bias_add(conv,b))
                        pooled = h

                    with tf.name_scope('conv5'):
                        w = tf.Variable(var_init([filterlen,1,args.cltcc_numfilters,args.cltcc_numfilters],args.cltcc_variance),name='w')
                        b = tf.Variable(tf.constant(0.1,shape=[args.cltcc_numfilters]),name='b')
                        conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
                        h = activation(tf.nn.bias_add(conv,b))
                        pooled = h

                    with tf.name_scope('conv6'):
                        w = tf.Variable(var_init([filterlen,1,args.cltcc_numfilters,args.cltcc_numfilters],args.cltcc_variance),name='w')
                        b = tf.Variable(tf.constant(0.1,shape=[args.cltcc_numfilters]),name='b')
                        conv = tf.nn.conv2d(pooled, w, strides=[1,1,1,1], padding='VALID')
                        h = activation(tf.nn.bias_add(conv,b))
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, 3, 1, 1],
                            strides=[1, 3, 1, 1],
                            padding='VALID')

                    last=pooled
                    input_size=int(last.get_shape()[1]*last.get_shape()[2]*last.get_shape()[3])
                    input_=tf.reshape(last,[args.batchsize,input_size])
                    inputs.append(input_)

        # language inputs
        if 'lang' in args.input:
            with tf.name_scope('lang'):
                lang_one_hot = tf.reshape(tf.one_hot(input_tensors['lang_'],len(hash.langs),axis=1),shape=[args.batchsize,len(hash.langs)])
                inputs.append(lang_one_hot)

        # time inputs
        if 'time' in args.input:
            with tf.name_scope('time'):

                #timestamp_ms_ = tf.placeholder(tf.float32, [args.batchsize,1], name='timestamp_ms_')

                def wrapped(var,length):
                    scaled=var/length*2*math.pi
                    return [tf.sin(scaled),tf.cos(scaled)]

                time1 = wrapped(input_tensors['timestamp_ms_'],1000*60*60*24*7)
                time2 = wrapped(input_tensors['timestamp_ms_'],1000*60*60*24)
                time3 = wrapped(input_tensors['timestamp_ms_'],1000*60*60*8)

                inputs.append(tf.stack(time1+time2+time3,axis=1))

        # constant input, for debugging purposes
        if 'const' in args.input:
            with tf.name_scope('const'):
                const=tf.reshape(tf.tile(tf.constant([1.0]),[args.batchsize]),[args.batchsize,1])
                inputs.append(const)

    # fully connected hidden layers
    with tf.name_scope('full'):
        layerindex=0
        with tf.name_scope('flattened_inputs'):
            final_layer=tf.concat(map(tf.contrib.layers.flatten,inputs),axis=1)
            final_layer_size=int(final_layer.get_shape()[1])

        for layersize in args.full:
            with tf.name_scope('full%d'%layerindex):
                w = tf.Variable(var_init([final_layer_size,layersize],1.0/math.sqrt(float(layersize))),name='w')
                b = tf.Variable(tf.constant(0.1,shape=[layersize]),name='b')
                h = tf.nn.relu(tf.matmul(final_layer,w)+b)
                final_layer=tf.nn.dropout(h,args.dropout)
                final_layer_size=layersize
            layerindex+=1

    # summary utilities for outputs
    newuser_vec = tf.reshape(input_tensors['newuser_'],[args.batchsize])

    '''
    A utility function for summarizing the accuracy of our outputs
    with respect to different languages and countries.
    '''
    def make_summaries(mk_summary):
        make_summaries_without_newuser(mk_summary)
        def mk_summary2(basename,weights):
            return mk_summary(basename+'newuser/',weights*newuser_vec)
        make_summaries_without_newuser(mk_summary2)

    def make_summaries_without_newuser(mk_summary):
        summary_langs=['en','ja','es','ar','fr','ko','zh']
        for lang in summary_langs:
            weights = tf.cast(tf.equal(input_tensors['lang_'],hash.lang2int(lang)),tf.float32)
            weights = tf.reshape(weights,[args.batchsize])
            mk_summary('filter_'+lang+'/',weights)

        summary_countries=['US','MX','ES','FR','JP']
        for country in summary_countries:
            weights = tf.cast(tf.equal(input_tensors['country_'],hash.country2int(country)),tf.float32)
            weights = tf.reshape(weights,[args.batchsize])
            mk_summary('filter_'+country+'/',weights)

        weights = tf.ones(input_tensors['lang_'].get_shape())
        weights = tf.reshape(weights,[args.batchsize])
        mk_summary('all/',weights)

        weights = tf.cast(tf.not_equal(input_tensors['lang_'],hash.lang2int('en')),tf.float32)
        weights = tf.reshape(weights,[args.batchsize])
        mk_summary('all_minus_en/',weights)

        weights = tf.cast(tf.not_equal(input_tensors['country_'],hash.country2int('US')),tf.float32)
        weights = tf.reshape(weights,[args.batchsize])
        mk_summary('all_minus_us/',weights)

    def mk_newuser_summary(basename,weights):
        tf.summary.scalar(basename+'newuser', tf.reduce_mean(weights*input_tensors['newuser_']))
    make_summaries_without_newuser(mk_newuser_summary)

    # rf outputs
    with tf.name_scope('output'):
        op_losses={}
        op_metrics={}

        # country loss
        if 'country' in args.output:
            with tf.name_scope('country'):

                # shortcuts
                final_layer_country = final_layer
                final_layer_country_size = final_layer_size

                if 'lang' in args.country_shortcut:
                    try:
                        final_layer_country = tf.concat([final_layer_country,lang_one_hot],axis=1)
                        final_layer_country_size += len(langs)
                    except:
                        pass

                if 'bow' in args.pos_shortcut:
                    try:
                        final_layer_country = tf.concat([final_layer_country,bow],axis=1)
                        final_layer_country_size += args.bow_layersize
                    except:
                        pass

                # layer
                w = tf.Variable(tf.zeros([final_layer_country_size,len(hash.country_codes)]),name='w')
                b = tf.Variable(tf.zeros([len(hash.country_codes)]),name='b')
                logits = tf.matmul(final_layer_country,w)+b
                country_softmax=tf.nn.softmax(logits)

                xentropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.to_int64(tf.reshape(input_tensors['country_'],[args.batchsize])),
                    logits=logits,
                    name='country_xentropy'
                    ))
                op_losses['country_xentropy']=xentropy
                op_metrics['optimization/country_xentropy']=tf.contrib.metrics.streaming_mean(xentropy,name='country_xentropy')

                def mk_metric(basename,weights):
                    xentropy_sum = tf.reduce_sum(
                        weights*tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.to_int64(tf.reshape(input_tensors['country_'],[args.batchsize])),
                            logits=logits,
                            name='country_xentropy'
                            )
                        )
                    total_weights=tf.reduce_sum(weights)
                    xentropy = tf.cond(total_weights>0,lambda: xentropy_sum/total_weights,lambda: 0.0)
                    op_metrics[basename+'country_xentropy']=tf.contrib.metrics.streaming_mean(xentropy,name=basename+'country_xentropy',weights=total_weights)

                    country=tf.reshape(tf.argmax(logits,axis=1),shape=[args.batchsize,1])
                    op_metrics[basename+'country_acc']=tf.contrib.metrics.streaming_accuracy(country,input_tensors['country_'],weights=weights,name=basename+'country_acc')

                    #def topk(k):
                        #name=basename+'top'+str(k)
                        #with tf.name_scope(name):
                            #topk=tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits,tf.reshape(country_,shape=[args.batchsize]),k=k),tf.float32))
                            #op_metrics[name]=tf.contrib.metrics.streaming_mean(topk,name=name)
                    #topk(1)
                    #topk(3)
                    #topk(5)
                    #topk(10)

                make_summaries(mk_metric)

                #country_softmax=tf.nn.softmax(logits)
                #country=tf.reshape(tf.argmax(logits,axis=1),shape=[args.batchsize,1])
                #op_metrics['country_acc']=tf.contrib.metrics.streaming_accuracy(country,country_,name='country_acc')

        # loc buckets
        if 'loc' in args.output:
            #loc_ = tf.placeholder(tf.int64, [args.batchsize,1],name='loc_')
            loc_ = input_tensors['loc_']

            # hash
            import hashlib
            numloc=2**args.loc_hashsize
            def hash_loc(str):
                return hash(str)%numloc

            with tf.name_scope('loc'):
                # shortcuts
                final_layer_pos = final_layer
                final_layer_pos_size = final_layer_size

                if 'lang' in args.country_shortcut:
                    final_layer_pos = tf.concat([final_layer_pos,lang_one_hot],axis=1)
                    final_layer_pos_size += len(langs)

                if 'bow' in args.pos_shortcut:
                    final_layer_pos = tf.concat([final_layer_pos,bow],axis=1)
                    final_layer_pos_size += args.bow_layersize

                if 'country' in args.pos_shortcut:
                    final_layer_pos = tf.concat([final_layer_pos,country_softmax],axis=1)
                    final_layer_pos_size += len(country_codes)

                # layer
                w0 = tf.Variable(tf.zeros([final_layer_pos_size, args.loc_bottleneck]),name='w0')
                w1 = tf.Variable(tf.zeros([args.loc_bottleneck, numloc]),name='w1')
                b1 = tf.Variable(tf.zeros([numloc]),name='b1')
                logits = tf.matmul(final_layer_pos,tf.matmul(w0,w1))+b1

                xentropy = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tf.to_int64(tf.reshape(loc_,[args.batchsize])),
                        logits=logits,
                        name='xentropy'
                        ))
                op_losses['loc_xentropy']=xentropy
                op_metrics['optimization/loc_xentropy']=tf.contrib.metrics.streaming_mean(xentropy,name='loc_xentropy')

                loc_softmax=tf.nn.softmax(logits)
                loc=tf.reshape(tf.argmax(logits,axis=1),shape=[args.batchsize,1])

                def mk_metric(name,weights):
                    xentropy_sum = tf.reduce_sum(
                        weights*tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=tf.to_int64(tf.reshape(loc_,[args.batchsize])),
                            logits=logits,
                            name='loc_xentropy'
                            )
                        )
                    total_weights=tf.reduce_sum(weights)
                    xentropy = tf.cond(total_weights>0,lambda: xentropy_sum/total_weights,lambda: 0.0)
                    op_metrics[name+'loc_xentropy']=tf.contrib.metrics.streaming_mean(xentropy,name=name+'loc_xentropy',weights=total_weights)

                    loc_softmax=tf.nn.softmax(logits)
                    loc=tf.reshape(tf.argmax(logits,axis=1),shape=[args.batchsize,1])
                    op_metrics[name+'loc_acc']=tf.contrib.metrics.streaming_accuracy(loc,loc_,weights=weights,name=name+'loc_acc')
                make_summaries(mk_metric)

        # position based losses
        if 'pos' in args.output:
            #gps_ = tf.placeholder(tf.float32, [args.batchsize,2], name='gps_')
            gps_=input_tensors['gps_']

            with tf.name_scope('pos'):

                # shortcuts
                pos_final_layer = final_layer
                pos_final_layer_size = final_layer_size

                if 'loc' in args.pos_shortcut:
                    try:
                        pos_final_layer = tf.concat([pos_final_layer,loc_softmax],axis=1)
                        pos_final_layer_size += numloc
                    except:
                        pass

                if 'country' in args.pos_shortcut:
                    try:
                        pos_final_layer = tf.concat([pos_final_layer,country_softmax],axis=1)
                        pos_final_layer_size += len(country_codes)
                    except:
                        pass

                # decompose true labels
                with tf.name_scope('reshape'):
                    op_lat_ = gps_[:,0]
                    op_lon_ = gps_[:,1]
                    op_lat_rad_ = op_lat_/360*2*math.pi
                    op_lon_rad_ = op_lon_/360*2*math.pi
                    op_lat_rad_1_ = tf.reshape(op_lat_rad_,[args.batchsize,1])
                    op_lat_rad_tiled_ = tf.tile(op_lat_rad_1_,[1,args.gmm_components])
                    op_lon_rad_1_ = tf.reshape(op_lon_rad_,[args.batchsize,1])
                    op_lon_rad_tiled_ = tf.tile(op_lon_rad_1_,[1,args.gmm_components])
                    gps_1_ = tf.reshape(gps_,[args.batchsize,1,2])
                    gps_tiled_ = tf.tile(gps_1_,[1,args.gmm_components,1])

                # treat gps coords as R^2
                #if 'naive' == args.pos_type:
                    #w = tf.Variable(tf.zeros([pos_final_layer_size, 2]),name='w')
                    #if args.pos_warmstart:
                        #b = tf.Variable([34.052235,-118.243683],name='b')
                    #else:
                        #b = tf.Variable(tf.zeros([2]),name='b')
                    #gps = tf.matmul(pos_final_layer,w) + b
                    #op_lat = gps[:,0]
                    #op_lon = gps[:,1]

                # angular generalized linear model
                # See: "Regression Models for Angular Response" by Fisher and Lee
                if 'aglm' == args.pos_type:
                    if args.pos_warmstart:
                        b0 = [ [ math.tan(city['lat']/360*math.pi*2),
                                 math.tan(city['lon']/360*math.pi)
                               ]
                               for city in city_loc.city_locs[:args.gmm_components]
                             ]
                    else:
                        b0 = 0.1

                    if args.gmm_type=='simple':
                        b = tf.Variable(tf.constant(b0,shape=[args.gmm_components, 2]), trainable=not args.gmm_notrain,name='b')
                        response = tf.tile(tf.reshape(b,[1,args.gmm_components,2]),[args.batchsize,1,1])
                    elif args.gmm_type=='complex':
                        w = tf.Variable(var_init([pos_final_layer_size, args.gmm_components, 2]),name='w')
                        b = tf.Variable(tf.constant(b0,shape=[args.gmm_components, 2]),name='b')
                        response = tf.tensordot(pos_final_layer,w,axes=[[1],[0]]) + b

                    op_lat = tf.atan(response[:,:,0])*360/2/math.pi
                    op_lon = tf.atan(response[:,:,1])*360/math.pi
                    gps = tf.stack([op_lat,op_lon],2)

                w = tf.Variable(var_init([pos_final_layer_size,args.gmm_components],0.1),name='w')
                b = tf.Variable(tf.constant(0.1,shape=[args.gmm_components]),name='b')
                mixture_logits = tf.matmul(pos_final_layer,w)+b
                mixture = tf.nn.softmax(mixture_logits)
                print('mixture=',mixture)
                print('mixture_logits=',mixture_logits)

                with tf.name_scope('summaries'):
                    tf.summary.scalar('mixture_sum', tf.reduce_mean(tf.reduce_sum(mixture,axis=1),axis=0))
                    vals,indices=tf.nn.top_k(mixture,k=args.gmm_components)
                    tf.summary.scalar('mixture_top0', tf.reduce_mean(vals[:,0]))
                    try:
                        tf.summary.scalar('mixture_top1', tf.reduce_mean(vals[:,1]))
                    except:
                        pass
                    try:
                        tf.summary.scalar('mixture_top2', tf.reduce_mean(vals[:,2]))
                    except:
                        pass
                    tf.summary.scalar('mixture_75', tf.reduce_mean(vals[:,1*args.gmm_components/4]))
                    tf.summary.scalar('mixture_50', tf.reduce_mean(vals[:,2*args.gmm_components/4]))
                    tf.summary.scalar('mixture_25', tf.reduce_mean(vals[:,3*args.gmm_components/4]))
                    tf.summary.scalar('mixture_0', tf.reduce_mean(vals[:,args.gmm_components-1]))

                # common outputs

                epsilon = 1e-6

                op_lat_rad = op_lat/360*2*math.pi
                op_lon_rad = op_lon/360*2*math.pi

                hav = lambda x: tf.sin(x/2)**2
                squared_angular_dist = ( hav(op_lat_rad-op_lat_rad_tiled_)
                                +tf.cos(op_lat_rad)*tf.cos(op_lat_rad_tiled_)*hav(op_lon_rad-op_lon_rad_tiled_)
                               )

                # radius of earth = 3959 miles, 6371 kilometers
                op_dists = 2*6371*tf.asin(tf.sqrt(tf.maximum(epsilon,squared_angular_dist)))
                op_dist_mixed = tf.reduce_sum(mixture*op_dists,axis=1)
                op_dist_unmixed = tf.reduce_mean(op_dists,axis=1)

                # set loss function
                if args.pos_loss=='l2':
                    op_loss = tf.reduce_sum((gps - gps_tiled_) * (gps - gps_tiled_),axis=2)
                if args.pos_loss=='chord':
                    op_delta_x = tf.cos(op_lat_rad)*tf.cos(op_lon_rad)-tf.cos(op_lat_rad_tiled_)*tf.cos(op_lon_rad_tiled_)
                    op_delta_y = tf.cos(op_lat_rad)*tf.sin(op_lon_rad)-tf.cos(op_lat_rad_tiled_)*tf.sin(op_lon_rad_tiled_)
                    op_delta_z = tf.sin(op_lat_rad) - tf.sin(op_lat_rad_tiled_)
                    op_chord = tf.sqrt(epsilon + op_delta_x**2 + op_delta_y**2 + op_delta_z**2)
                    op_loss = op_chord
                if args.pos_loss=='dist_sqrt':
                    op_loss = tf.sqrt(op_dists)
                if args.pos_loss=='dist':
                    op_loss = op_dists
                if args.pos_loss=='dist2':
                    op_loss = op_dists*op_dists
                if args.pos_loss=='angular':
                    op_loss = squared_angular_dist

                op_losses['pos_loss']=tf.reduce_mean(tf.reduce_sum(mixture*op_loss,axis=1))
                op_metrics['optimization/pos_loss']=tf.contrib.metrics.streaming_mean(op_dist_mixed,name='dist')
                op_metrics['optimization/pos_loss_unmixed']=tf.contrib.metrics.streaming_mean(op_dist_unmixed,name='dist')

                def mk_metric(basename,weights):
                    total_weights = tf.reduce_sum(weights)
                    op_dist_ave = tf.cond(total_weights>0,
                        lambda: tf.reduce_sum(weights*op_dist_mixed)/total_weights,
                        lambda: 0.0
                        )
                    op_metrics[basename+'dist']=tf.contrib.metrics.streaming_mean(op_dist_ave,weights=total_weights,name='dist')
                    def mk_threshold(threshold):
                        op_threshold = tf.sign(op_dist_mixed-threshold)/2+0.5
                        op_threshold_ave = tf.cond(total_weights>0,
                            lambda: tf.reduce_sum(weights*op_threshold)/total_weights,
                            lambda: 0.0
                            )
                        name=basename+'k'+str(threshold)
                        op_metrics[name]=tf.contrib.metrics.streaming_mean(op_threshold_ave,weights=total_weights,name=name)
                    mk_threshold(10)
                    mk_threshold(50)
                    mk_threshold(100)
                    mk_threshold(500)
                    mk_threshold(1000)
                    mk_threshold(2000)
                    mk_threshold(3000)
                make_summaries(mk_metric)

            # position stddev
            #with tf.name_scope('stddev'):
                #layerindex=0
                #gps_layer=gps
                #gps_layer_size=2
                #for layersize in [4,4,4,4]:
                    #with tf.name_scope('full%d'%layerindex):
                        #w = tf.Variable(var_init([gps_layer_size,args.gmm_components,layersize],1.0/math.sqrt(float(layersize))),name='w')
                        #b = tf.Variable(tf.constant(0.1,shape=[args.gmm_components,layersize]),name='b')
                        #h = tf.nn.relu(tf.tensordot(gps_layer,w,axes=[[2],[0]])+b)
                        #gps_layer=tf.nn.dropout(h,args.dropout)
                        #gps_layer_size=layersize
                    #layerindex+=1
#
                #gps_layer=tf.reduce_sum(mixture*gps,axis=1)
                #stddev_input=tf.concat([final_layer,gps_layer],axis=1)
                #w = tf.Variable(var_init([final_layer_size+gps_layer_size,1],1.0/math.sqrt(float(final_layer_size+gps_layer_size))),name='w')
                #b = tf.Variable(tf.constant(3000.0,shape=[1]),name='b')
                #op_dist_guess = tf.matmul(stddev_input,w)+b
#
                ##guess_error = tf.reduce_mean((op_dist-op_dist_guess)**2)
                #guess_error = tf.reduce_mean(tf.abs(op_dist-op_dist_guess))
                #op_losses['dist_guess_error'] = guess_error
                #op_metrics['optimization/dist_guess']=tf.contrib.metrics.streaming_mean(op_dist_guess,name='dist_guess')
                #op_metrics['optimization/dist_guess_error']=tf.contrib.metrics.streaming_mean(guess_error,name='dist_guess_error')

    # set loss function
    with tf.name_scope('loss'):
        op_projections=[]

        if args.loss_weights == 'auto':
            epsilon=1e-3
            op_loss=0
            num_losses = len(op_losses)
            w0 = tf.constant(1.0/float(num_losses),shape=[num_losses])
            w = tf.Variable(w0,name='w')
            w_max = tf.maximum(w, tf.constant(epsilon,shape=[num_losses]))
            w_norm = tf.reduce_sum(w_max)+epsilon
            i=0
            tf.summary.scalar('weight_norm',w_norm)
            for k,v in op_losses.iteritems():
                op_loss += (w_max[i]/w_norm)*v
                tf.summary.scalar('optimization/weight_'+k, w[i])
                op_projections.append(w[i].assign(w_max[i]/w_norm))
                i+=1
            op_loss += tf.linalg.norm(w0-w_max)**2

        if args.loss_weights == 'manual':
            op_loss = 0
            if 'pos_loss' in op_losses:
                op_loss += op_losses['pos_loss']/1000
                #op_loss += op_losses['dist_guess_error']/10000
            if 'country_xentropy' in op_losses:
                op_loss += op_losses['country_xentropy']
            if 'loc_xentropy' in op_losses:
                op_loss += op_losses['loc_xentropy']/1000

        if args.loss_weights == 'manual2':
            op_loss = 0
            if 'pos_loss' in op_losses:
                op_loss += op_losses['pos_loss']/100
                op_loss += op_losses['dist_guess_error']/1000
            if 'country_xentropy' in op_losses:
                op_loss += op_losses['country_xentropy']
            if 'loc_xentropy' in op_losses:
                op_loss += op_losses['loc_xentropy']/10

        if args.loss_weights == 'ave':
            op_loss = tf.reduce_mean(op_losses.values())

        if args.loss_weights == 'manual3':
            op_loss = 1.0
            if 'pos_loss' in op_losses:
                op_loss *= op_losses['pos_loss']/1000
                #op_loss += op_losses['dist_guess_error']/10000
            if 'country_xentropy' in op_losses:
                op_loss *= op_losses['country_xentropy']
            if 'loc_xentropy' in op_losses:
                op_loss *= op_losses['loc_xentropy']/1000

        if args.loss_weights == 'prod':
            op_loss = tf.reduce_mean(map(tf.log,op_losses.values()))

        op_metrics['optimization/op_loss']=tf.contrib.metrics.streaming_mean(op_loss,name='op_loss')

        # add regularizers
        with tf.name_scope('l2_regularization'):
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            for var in vars:
                regularizers.append(args.l2*tf.nn.l2_loss(var))

        op_loss_regularized=op_loss+tf.reduce_sum(regularizers)
        op_losses['optimization/op_loss_regularized']=op_loss_regularized

        return op_metrics,op_loss_regularized

################################################################################

def json2dict(args,str):
    import simplejson as json
    import unicodedata
    import numpy as np
    data=json.loads(str)

    batch_dict={}

    # simplify the unicode representation
    data['text']=unicodedata.normalize('NFKC',unicode(data['text'].lower()))

    # FIXME: possibly skip locations
    #if args.loc_filter:
        #try:
            #full_name=data['place']['full_name']
            #if not (full_name in lochash):
                #continue
        #except:
            #continue

    # userids
    try:
        if data['user']['id'] in userids:
            batch_dict['newuser_']=0
        else:
            batch_dict['newuser_']=1
            userids.add(data['user']['id'])
    except:
        batch_dict['newuser_']=0

    # get hashes
    batch_dict['lang_']=hash.lang2int(data['lang'])
    if 'country' in args.output:
        try:
            country_code=hash.country2int(data['place']['country_code'])
        except:
            country_code=0
        batch_dict['country_']=np.array([country_code])

    # get inputs
    if 'bow' in args.input:
        batch_dict['hash_']=hv.transform([data['text']])

    if 'cnn' in args.input:
        encodedtext=np.zeros([1,tweetlen,args.cnn_vocabsize])

        for i in range(min(tweetlen,len(data['text']))):
            for k in range(0,args.cnn_khot):
                char=ord(data['text'][i])
                index=(5381*char + 88499*k)%args.cnn_vocabsize
                encodedtext[0][i][index]=1

        batch_dict['text_']=encodedtext

    if 'time' in args.input:
        timestamp = np.array(float(data['timestamp_ms']))
        batch_dict['timestamp_ms_']=timestamp

    # get true output
    if 'pos' in args.output:
        if data['geo']:
            lat=data['geo']['coordinates'][0]
            lon=data['geo']['coordinates'][1]
            coord=(lat,lon)
        else:
            def centroid(xs):
                coords=xs[0]
                lats=[coord[0] for coord in coords]
                lons=[coord[1] for coord in coords]
                lat=sum(lats)/float(len(lats))
                lon=sum(lons)/float(len(lons))
                coord=(lat,lon)
                return coord

            coord=centroid(data['place']['bounding_box']['coordinates'])
            # the twitter format stores bounding boxes as (lon,lat) pairs
            # instead of (lat,lon) pairs, so we need to flip them around
            coord=(coord[1],coord[0])
        batch_dict['gps_']=np.array(coord)

    if 'loc' in args.output:
        try:
            loc_code=hash_loc(data['place']['full_name'])
        except:
            loc_code=0
        batch_dict['loc_']=np.array([loc_code])

    return batch_dict

################################################################################

def mk_feed_dict(args,batch):
    import numpy as np
    from collections import defaultdict
    feed_dict = {}
    batch_dict=defaultdict(list)

    for data in batch:
        for k in data:
            batch_dict[k].append(data[k])

    feed_dict['lang_:0'] = np.vstack(batch_dict['lang_'])
    feed_dict['country_:0'] = np.vstack(batch_dict['country_'])
    feed_dict['newuser_:0'] = np.vstack(batch_dict['newuser_'])

    if 'bow' in args.input:
        def mkSparseTensorValue(m):
            m2=sp.sparse.coo_matrix(m)
            if args.bow_dense:
                return m2.toarray()
            else:
                return tf.SparseTensorValue(
                    zip(m2.row,m2.col),
                    m2.data,
                    m2.shape,
                    )
        feed_dict['hash_:0'] = mkSparseTensorValue(sp.sparse.vstack(batch_dict['hash_']))

    if 'cnn' in args.input:
        feed_dict['text_:0'] = np.vstack(batch_dict['text_'])

    if 'time' in args.input:
        feed_dict['timestamp_ms_:0'] = np.vstack(batch_dict['timestamp_ms_'])

    if 'pos' in args.output:
        feed_dict['gps_:0'] = np.vstack(batch_dict['gps_'])

    if 'loc' in args.output:
        feed_dict['loc_:0'] = np.vstack(batch_dict['loc_'])

    return feed_dict

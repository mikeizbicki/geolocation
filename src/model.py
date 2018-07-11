import math
import city_loc
import myhash

tweetlen=280

################################################################################

def update_parser(parser):
    parser.add_argument('--batchsize',type=int,default=100)
    parser.add_argument('--learningrate',type=float,default=5e-5)
    parser.add_argument('--optimizer',choices=['adam','sgd'],default='adam')
    parser.add_argument('--momentum',type=float,default=0.9)
    parser.add_argument('--decay',type=float,default=None)
    parser.add_argument('--dropout',type=float,default=0.5)
    parser.add_argument('--l1',type=float,default=0.0)
    parser.add_argument('--l2',type=float,default=1e-5)

    parser.add_argument('--input',choices=['cnn','bow','lang','time','const'],nargs='+',default=['cnn','lang','time'])
    parser.add_argument('--remove_urls',type=bool,default=True)
    parser.add_argument('--bow_hashsize',type=int,default=20)
    parser.add_argument('--bow_layersize',type=int,default=2)
    parser.add_argument('--bow_dense',action='store_true')
    parser.add_argument('--cnn_type',choices=['vdcnn','cltcc'],default='cltcc')
    parser.add_argument('--cnn_vocabsize',type=int,default=128)
    parser.add_argument('--cnn_khot',type=int,default=1)
    parser.add_argument('--text_naive',type=bool,default=False)
    parser.add_argument('--text_multichar_init_bit',type=bool,default=True)
    parser.add_argument('--text_latin_bit',type=bool,default=True)
    parser.add_argument('--text_transliterate',type=bool,default=True)
    parser.add_argument('--text_hashsize_combiners',type=int,default=16)
    parser.add_argument('--vdcnn_numfilters',type=int,default=64)
    parser.add_argument('--vdcnn_size',type=int,default=1)
    parser.add_argument('--vdcnn_resnet',action='store_true')
    parser.add_argument('--vdcnn_no_bn',action='store_true')
    parser.add_argument('--cltcc_numfilters',type=int,default=1024)
    parser.add_argument('--cltcc_variance',type=float,default=0.02)

    parser.add_argument('--full',type=int,nargs='*',default=[2048,2048])
    parser.add_argument('--full_per_lang',action='store_true')

    parser.add_argument('--output',choices=['pos','country','loc'],default=['pos','country'],nargs='*')
    parser.add_argument('--loss_weights',choices=['auto','ave','manual','manual2','manual3','prod'],default='ave')
    parser.add_argument('--loss_staircase',action='store_true')
    parser.add_argument('--pos_type',choices=['naive','aglm','aglm_mix'],default='aglm')
    parser.add_argument('--pos_loss',choices=['l2','chord','dist','dist_sqrt','dist2','angular'],default='dist')
    parser.add_argument('--pos_shortcut',choices=['lang','loc','country'],default=[],nargs='*')
    parser.add_argument('--pos_warmstart',type=bool,default=True)
    parser.add_argument('--gmm_type',choices=['verysimple','simple','complex'],default='verysimple')
    parser.add_argument('--gmm_distribution',choices=['fvm','gaussian','efam'],default='fvm')
    parser.add_argument('--gmm_lrfactor',type=float,default=1e-2)
    parser.add_argument('--gmm_notrain',action='store_true')
    parser.add_argument('--gmm_components',type=int,default=10000)
    parser.add_argument('--gmm_prekappa0',type=float,default=10.0)
    parser.add_argument('--gmm_maxprob',type=float,default=None)
    parser.add_argument('--gmm_distloss',action='store_true')
    parser.add_argument('--gmm_decomposed',type=int,nargs='*',default=[])
    parser.add_argument('--country_shortcut',choices=['bow','lang'],default=[],nargs='*')
    parser.add_argument('--loc_type',choices=['popular','myhash'],default='myhash')
    parser.add_argument('--loc_filter',action='store_true')
    parser.add_argument('--loc_hashsize',type=int,default=16)
    parser.add_argument('--loc_bottleneck',type=int,default=None)
    parser.add_argument('--loc_shortcut',choices=['bow','lang','country'],default=[],nargs='*')
    parser.add_argument('--enable_shortcuts',action='store_true')

    parser.add_argument('--predict_lang',type=bool,default=False)
    parser.add_argument('--predict_lang_use',type=bool,default=False)
    parser.add_argument('--predict_lang_layers',type=int,default=[1024,1024],nargs='*')

    parser.add_argument('--summary_size',choices=['small','med','all'],default='med')
    parser.add_argument('--summary_newusers',action='store_true')
    parser.add_argument('--hashes_uniq',type=bool,default=True)
    parser.add_argument('--hashes_true',type=bool,default=False)

################################################################################

def inference(args,input_tensors):
    import tensorflow as tf
    op_losses={}
    op_metrics={}
    op_outputs={}
    epsilon = 1e-6

    # adjust hash functions for backwards compatibility
    if args.hashes_uniq:
        myhash.langs=myhash.langs_uniq
        myhash.country_codes=myhash.country_codes_uniq

    if args.hashes_true:
        myhash.langs=myhash.langs_true

    # preprocess args
    if args.enable_shortcuts:
        args.pos_shortcut=['lang','loc','country']
        args.country_shortcut=['bow','lang']
        args.loc_shortcut=['bow','lang','country']

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

    # helpers for defining summaries
    def make_summaries(mk_summary):
        make_summaries_without_newuser(mk_summary)
        if args.summary_newusers:
            def mk_summary2(basename,weights):
                return mk_summary(basename+'newuser/',weights*newuser_vec)
            make_summaries_without_newuser(mk_summary2)

    def make_summaries_without_newuser(mk_summary):
        with tf.name_scope('mk_summaries'):
            if args.summary_size=='small':
                summary_langs=[]
                summary_countries=[]
            elif args.summary_size=='med':
                summary_langs=['en','ja','es','ar','fr','zh','pt','tr','tl','in','und','de','vi']
                summary_countries=['US','MX','ES','FR','JP']
            elif args.summary_size=='all':
                summary_langs=myhash.langs
                summary_countries=myhash.country_codes

            for lang in summary_langs:
                weights = tf.cast(tf.equal(input_tensors['lang_'],myhash.lang2int(lang)),tf.float32)
                weights = tf.reshape(weights,[args.batchsize])
                mk_summary('filter_'+lang+'/',weights)

            for country in summary_countries:
                weights = tf.cast(tf.equal(input_tensors['country_'],myhash.country2int(country)),tf.float32)
                weights = tf.reshape(weights,[args.batchsize])
                mk_summary('filter_'+country+'/',weights)

            weights = tf.cast(tf.not_equal(input_tensors['lang_'],myhash.lang2int('en')),tf.float32)
            weights = tf.reshape(weights,[args.batchsize])
            mk_summary('all_minus_en/',weights)

            weights = tf.cast(tf.not_equal(input_tensors['country_'],myhash.country2int('US')),tf.float32)
            weights = tf.reshape(weights,[args.batchsize])
            mk_summary('all_minus_us/',weights)

            weights = tf.ones(input_tensors['lang_'].get_shape())
            weights = tf.reshape(weights,[args.batchsize])
            mk_summary('all/',weights)

    # summarize newusers
    with tf.name_scope('newusers'):
        newuser_vec = tf.reshape(input_tensors['newuser_'],[args.batchsize])
        def mk_newuser_summary(basename,weights):
            tf.summary.scalar(basename+'newuser', tf.reduce_mean(weights*input_tensors['newuser_']))
        make_summaries_without_newuser(mk_newuser_summary)

    # helper for making xentropy layers with appropriate summaries
    def mk_xentropy_layer(valname,val_,logits):

        xentropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.to_int64(tf.reshape(val_,[args.batchsize])),
                logits=logits+epsilon,
                name='xentropy'
                ))
        op_losses[valname+'_xentropy']=xentropy
        op_metrics['optimization/'+valname+'_xentropy']=(args.batchsize,xentropy)

        val=tf.reshape(tf.argmax(logits,axis=1),shape=[args.batchsize,1])

        def mk_metric(groupname,weights):
            xentropy_sum = tf.reduce_sum(
                weights*tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.to_int64(tf.reshape(val_,[args.batchsize])),
                    logits=logits,
                    name=valname+'_xentropy'
                    )
                )
            total_weights=tf.reduce_sum(weights)
            xentropy = tf.cond(total_weights>0,lambda: xentropy_sum/total_weights,lambda: 0.0)
            op_metrics[groupname+valname+'_xentropy']=(total_weights,xentropy)

            accuracy=tf.metrics.accuracy(val,val_,weights=weights)
            op_metrics[groupname+valname+'_acc']=(total_weights,accuracy)
        make_summaries(mk_metric)

    # helper for defining fully connected layers
    def mk_full_layers(input_layer,layers):
        layerindex=0
        input_layer_size=int(input_layer.get_shape()[1])
        for layersize in layers:
            with tf.name_scope('full%d'%layerindex):
                w = tf.Variable(var_init([input_layer_size,layersize],1.0/math.sqrt(float(layersize))),name='w')
                b = tf.Variable(tf.constant(0.1,shape=[layersize]),name='b')
                h = tf.nn.relu(tf.matmul(input_layer,w)+b)
                input_layer=tf.nn.dropout(h,args.dropout)
                input_layer_size=layersize
            layerindex+=1
        return input_layer

    # tf inputs
    with tf.name_scope('inputs'):

        regularizers=[]
        inputs=[]

        # myhash bow inputs
        if 'bow' in args.input:
            with tf.name_scope('bow'):
                # FIXME: why does this need to be global?
                global hash_
                bow_size=2**args.bow_hashsize
                if args.bow_dense:
                    hash_ = tf.placeholder(tf.float32,[args.batchsize,bow_size],'hash_')
                    raise ValueError('fixme')
                    matmul = tf.matmul
                    hash_reg=args.l1*tf.reduce_sum(tf.abs(hash_))
                else:
                    hash_ = input_tensors['hash_'] #tf.sparse_placeholder(tf.float32,name='hash_')
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
                    cnn_layer=tf.reshape(net,[args.batchsize,input_size])
                    inputs.append(cnn_layer)

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
                    cnn_layer=tf.reshape(last,[args.batchsize,input_size])
                    inputs.append(cnn_layer)

        # language inputs
        if 'lang' in args.input:

            # lang predictor
            if args.predict_lang:
                with tf.name_scope('lang_predictor'):
                    final_layer= mk_full_layers(cnn_layer,args.predict_lang_layers)
                    w = tf.Variable(tf.zeros([final_layer.get_shape()[1],len(myhash.country_codes)]),name='w')
                    b = tf.Variable(tf.zeros([len(myhash.country_codes)]),name='b')
                    logits = tf.matmul(final_layer,w)+b
                    lang_softmax=tf.nn.softmax(logits)
                    mk_xentropy_layer('lang',input_tensors['lang_'],logits)
                    op_outputs['lang_softmax']=lang_softmax

            # the lang layer
            with tf.name_scope('lang'):
                if args.predict_lang_use:
                    lang_one_hot=lang_softmax
                else:
                    lang_one_hot = tf.reshape(tf.one_hot(input_tensors['lang_'],len(myhash.langs),axis=1),shape=[args.batchsize,len(myhash.langs)])

                inputs.append(lang_one_hot)

        # time inputs
        if 'time' in args.input:
            with tf.name_scope('time'):

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
        final_layer=tf.concat(map(tf.contrib.layers.flatten,inputs),axis=1)
        final_layer_size=int(final_layer.get_shape()[1])
        if args.full_per_lang:
            layersize=1024
            with tf.name_scope('full0'):
                w = tf.Variable(var_init([len(myhash.langs),final_layer_size,layersize]),name='w')
                b = tf.Variable(tf.constant(0.1,shape=[len(myhash.langs),layersize]),name='b')
                final_layer = tf.tensordot(final_layer,w,axes=[[1],[1]])+b
                final_layer = tf.nn.relu(final_layer)
                final_layer = tf.nn.dropout(final_layer,args.dropout)
                final_layer = final_layer*tf.reshape(lang_one_hot,[args.batchsize,len(myhash.langs),1])
                final_layer = tf.reduce_sum(final_layer,axis=1)
            with tf.name_scope('full1'):
                w = tf.Variable(var_init([len(myhash.langs),layersize,layersize]),name='w')
                b = tf.Variable(tf.constant(0.1,shape=[len(myhash.langs),layersize]),name='b')
                final_layer = tf.tensordot(final_layer,w,axes=[[1],[1]])+b
                final_layer = tf.nn.relu(final_layer)
                final_layer = tf.nn.dropout(final_layer,args.dropout)
                final_layer = final_layer*tf.reshape(lang_one_hot,[args.batchsize,len(myhash.langs),1])
                final_layer = tf.reduce_sum(final_layer,axis=1)

            final_layer_size=layersize
        else:
            final_layer=mk_full_layers(final_layer, args.full)
            final_layer_size=int(final_layer.get_shape()[1])

    # rf outputs
    with tf.name_scope('output'):
        # country loss
        if 'country' in args.output:
            with tf.name_scope('country'):

                # shortcuts
                final_layer_country = final_layer
                final_layer_country_size = final_layer_size

                if 'lang' in args.country_shortcut:
                    if 'lang' in args.input:
                        final_layer_country = tf.concat([final_layer_country,lang_one_hot],axis=1)
                        final_layer_country_size += int(lang_one_hot.get_shape()[1])

                if 'bow' in args.country_shortcut:
                    if 'bow' in args.output:
                        final_layer_country = tf.concat([final_layer_country,bow],axis=1)
                        final_layer_country_size += args.bow_layersize

                # layer
                w = tf.Variable(tf.zeros([final_layer_country_size,len(myhash.country_codes)]),name='w')
                b = tf.Variable(tf.zeros([len(myhash.country_codes)]),name='b')
                logits = tf.matmul(final_layer_country,w)+b
                country_softmax=tf.nn.softmax(logits)
                op_outputs['country_softmax']=country_softmax

                mk_xentropy_layer('country',input_tensors['country_'],logits)


        # loc buckets
        myhash.init_loc_hash(args)
        if 'loc' in args.output:
            #loc_ = tf.placeholder(tf.int64, [args.batchsize,1],name='loc_')
            loc_ = input_tensors['loc_']

            with tf.name_scope('loc'):
                # shortcuts
                final_layer_loc = final_layer
                final_layer_loc_size = final_layer_size

                if 'lang' in args.loc_shortcut:
                    if 'lang' in args.input:
                        final_layer_loc = tf.concat([final_layer_loc,lang_one_hot],axis=1)
                        final_layer_loc_size += int(lang_one_hot.get_shape()[1])

                if 'bow' in args.loc_shortcut:
                    if 'bow' in args.output:
                        final_layer_loc = tf.concat([final_layer_loc,bow],axis=1)
                        final_layer_loc_size += args.bow_layersize

                if 'country' in args.loc_shortcut:
                    if 'country' in args.output:
                        final_layer_loc = tf.concat([final_layer_loc,country_softmax],axis=1)
                        final_layer_loc_size += int(country_softmax.get_shape()[1])

                # layer
                if args.loc_bottleneck:
                    w0 = tf.Variable(tf.zeros([final_layer_loc_size, args.loc_bottleneck]),name='w0')
                    w1 = tf.Variable(tf.zeros([args.loc_bottleneck, myhash.loc_max]),name='w1')
                    b1 = tf.Variable(tf.zeros([myhash.loc_max]),name='b1')
                    logits = tf.matmul(final_layer_loc,tf.matmul(w0,w1))+b1
                else:
                    w1 = tf.Variable(tf.zeros([final_layer_loc_size, myhash.loc_max]),name='w1')
                    b1 = tf.Variable(tf.zeros([myhash.loc_max]),name='b1')
                    logits = tf.matmul(final_layer_loc,w1)+b1

                loc_softmax=tf.nn.softmax(logits)
                op_outputs['loc_softmax']=loc_softmax
                mk_xentropy_layer('loc',loc_,logits)

        # position based losses
        if 'pos' in args.output:
            gps_=input_tensors['gps_']

            with tf.name_scope('pos'):

                # shortcuts
                pos_final_layer = final_layer
                pos_final_layer_size = final_layer_size

                if 'lang' in args.country_shortcut:
                    if 'lang' in args.input:
                        pos_final_layer = tf.concat([pos_final_layer,lang_one_hot],axis=1)
                        pos_final_layer_size += int(lang_one_hot.get_shape()[1])

                if 'bow' in args.country_shortcut:
                    if 'bow' in args.output:
                        pos_final_layer = tf.concat([pos_final_layer,bow],axis=1)
                        pos_final_layer_size += args.bow_layersize

                if 'loc' in args.pos_shortcut:
                    if 'loc' in args.output:
                        pos_final_layer = tf.concat([pos_final_layer,loc_softmax],axis=1)
                        pos_final_layer_size += myhash.loc_max

                if 'country' in args.pos_shortcut:
                    if 'country' in args.output:
                        pos_final_layer = tf.concat([pos_final_layer,country_softmax],axis=1)
                        pos_final_layer_size += int(country_softmax.get_shape()[1])

                # decompose true labels
                with tf.name_scope('reshape'):
                    op_lat_ = gps_[:,0]
                    op_lon_ = gps_[:,1]
                    op_lat_rad_ = op_lat_/360*2*math.pi
                    op_lon_rad_ = op_lon_/360*math.pi
                    #op_lon_rad_ = op_lon_/360*2*math.pi

                # treat gps coords as R^2
                if 'naive' == args.pos_type:
                    w = tf.Variable(tf.zeros([pos_final_layer_size, 2]),name='w')
                    if args.pos_warmstart:
                        b = tf.Variable([34.052235,-118.243683],name='b')
                    else:
                        b = tf.Variable(tf.zeros([2]),name='b')
                    gps = tf.matmul(pos_final_layer,w) + b
                    op_lat = gps[:,0]
                    op_lon = gps[:,1]

                # angular generalized linear model
                # See: "Regression Models for Angular Response" by Fisher and Lee
                if 'aglm' == args.pos_type:
                    w = tf.Variable(tf.zeros([pos_final_layer_size, 2]),name='w')
                    if args.pos_warmstart:
                        b = tf.Variable([0.6745,-2],name='b')
                    else:
                        b = tf.Variable(tf.zeros([2]),name='b')
                    response = tf.matmul(pos_final_layer,w) + b
                    op_lat = tf.atan(response[:,0])*360/2/math.pi
                    op_lon = tf.atan(response[:,1])*360/math.pi
                    gps = tf.stack([op_lat,op_lon],1)

                # Mixture of fisher distributions
                # see "directional statistics" by Mardia and Jupp for Fisher distribution
                if 'aglm_mix' == args.pos_type:

                    safeish_exp = lambda x: tf.exp(tf.minimum(x,30+tf.log(x+epsilon)),name='safeish_exp')

                    if args.pos_warmstart:
                        gps_coords_rad = [ [ city['lat']/360*2*math.pi,
                                             city['lon']/360*math.pi
                                           ]
                                           for city in city_loc.get_cities(args.gmm_components)
                                         ]
                        pre_mu_gps_rad0 = tf.constant(gps_coords_rad,shape=[args.gmm_components,2])
                        pre_mu_tan = tf.tan(pre_mu_gps_rad0)
                    else:
                        pre_mu_gps0 = 0.1
                    pre_kappa_constant = tf.constant(args.gmm_prekappa0,shape=[args.gmm_components])

                    if args.gmm_type=='simple' or args.gmm_type=='verysimple':
                        trainable=args.gmm_type!='verysimple'
                        pre_mu_var = tf.Variable(pre_mu_gps_rad0,name='pre_mu',trainable=trainable)
                        pre_mu_reshape = tf.reshape(pre_mu_var,[1,args.gmm_components,2])
                        pre_mu = args.gmm_lrfactor*pre_mu_reshape+(1-args.gmm_lrfactor)*tf.stop_gradient(pre_mu_reshape)
                        pre_kappa = tf.Variable(pre_kappa_constant,name='pre_kappa',trainable=trainable)
                        kappa = safeish_exp(pre_kappa)

                    else:
                        with tf.name_scope('pre_mu'):
                            w = tf.Variable(tf.zeros([pos_final_layer_size,args.gmm_components,2]),name='w')
                            b = tf.Variable(pre_mu_tan,name='b')
                            pre_mu_tan = tf.tensordot(pos_final_layer,w,axes=[1,0])+b
                            pre_mu = tf.atan(pre_mu_tan)

                            pre_kappa = tf.Variable(pre_kappa_constant,name='pre_kappa')
                            kappa = safeish_exp(pre_kappa)

                        # FIXME: making kappa depend on pos_final_layer introduces a degenerate dependency on mu which prevents optimization
                        #with tf.name_scope('pre_kappa'):
                            #w = tf.Variable(var_init([pos_final_layer_size,args.gmm_components],0.1),name='w')
                            #b = tf.Variable(pre_kappa_constant,name='b')
                            #pre_kappa = tf.tensordot(pos_final_layer,w,axes=[1,0])+b
                        #kappa = tf.exp(pre_kappa)

                    #kappa = tf.Variable(tf.exp(pre_kappa_constant))
                    #pre_kappa = tf.log(tf.abs(kappa)+epsilon)

                    mu = tf.stack([ tf.sin(pre_mu[:,:,0])
                                  , tf.cos(pre_mu[:,:,0]) * tf.sin(pre_mu[:,:,1]*2)
                                  , tf.cos(pre_mu[:,:,0]) * tf.cos(pre_mu[:,:,1]*2)
                                  ])
                    pre_mu_lat = pre_mu[:,:,0]*360/2/math.pi
                    pre_mu_lon = pre_mu[:,:,1]*360/math.pi
                    pre_mu_gps = tf.stack([pre_mu_lat,pre_mu_lon],axis=1)

                    x = tf.stack([ tf.sin(op_lat_rad_)
                                 , tf.cos(op_lat_rad_) * tf.sin(op_lon_rad_*2)
                                 , tf.cos(op_lat_rad_) * tf.cos(op_lon_rad_*2)
                                 ])
                    x_reshape = tf.reshape(x,[3,args.batchsize,1])

                    def decomposed_linear_layer(input_layer,mid_layer_sizes,output_size):
                        t={}
                        input_layer_size=int(input_layer.get_shape()[1])
                        with tf.name_scope('decomposed_linear_layer'):
                            for i in mid_layer_sizes:
                                with tf.name_scope('grouping_'+str(i)):
                                    w1 = tf.Variable(var_init([input_layer_size,i]),name='w1')
                                    w2 = tf.Variable(var_init([i,output_size]),name='w2')
                                    t[i] = tf.matmul(tf.matmul(input_layer,w1),w2)
                            with tf.name_scope('grouping_all'):
                                w = tf.Variable(var_init([input_layer_size,output_size]),name='w')
                                b = tf.Variable(tf.constant(0.1,shape=[output_size]),name='b')
                                t['all']=tf.matmul(input_layer,w)+b
                            logits=sum(t.values())
                        return logits


                    w = tf.Variable(var_init([pos_final_layer_size,args.gmm_components],0.01),name='w')
                    b = tf.Variable(tf.constant(0.1,shape=[args.gmm_components]),name='b')
                    mixture_logits = tf.matmul(pos_final_layer,w)+b
                    #mixture_logits = decomposed_linear_layer(pos_final_layer,args.gmm_decomposed,args.gmm_components)
                    mixture = tf.nn.softmax(mixture_logits + epsilon) + epsilon

                    #safe_logsinh = lambda x: tf.where(
                            #tf.greater(x,1.0),
                            #x-0.693147,
                            #tf.log(tf.sinh(x+epsilon)+epsilon),
                            #)
                    #log_likelihood_per_component = pre_kappa - safe_logsinh(kappa) + (kappa * tf.reduce_sum(x_reshape*mu,axis=0))

                    if args.gmm_distribution=='fvm':
                        log_likelihood_per_component = tf.where(
                            tf.greater(kappa,1.0),
                            -kappa,
                            -kappa*kappa
                            )+(kappa * tf.reduce_sum(x_reshape*mu,axis=0))
                    elif args.gmm_distribution=='gaussian':
                        vecsum=tf.reduce_sum(tf.abs(x_reshape-mu)**2.0,axis=0)
                        log_likelihood_per_component = tf.exp(-kappa*vecsum)
                        # FIXME: double exp
                    elif args.gmm_distribution=='efam':
                        pow_var = tf.Variable(tf.constant(0.1,shape=[args.gmm_components]),name='pow',trainable=True)
                        pow = tf.maximum(pow_var,1e-9)
                        op_outputs['aglm_mix/pow']=pow
                        vecsum=tf.reduce_sum(tf.abs(x_reshape-mu)**pow,axis=0)
                        log_likelihood_per_component = tf.exp(-kappa*vecsum)
                        #safeish_exp2 = lambda x: tf.exp(tf.minimum(x,2+tf.log(x+epsilon)),name='safeish_exp')
                        #log_likelihood_per_component = tf.exp(-pre_kappa*(x_reshape-mu))**pow

                    likelihood_mixed = tf.reduce_sum(safeish_exp(log_likelihood_per_component)*mixture,axis=1)
                    log_loss = - tf.log(likelihood_mixed + epsilon)
                    loss=tf.reduce_mean(log_loss,name='dbgloss')
                    op_losses['pos_loss_mix']=loss
                    op_metrics['optimization/aglm_mix']=(args.batchsize,loss)

                    with tf.name_scope('summaries'):
                        vals,indices=tf.nn.top_k(mixture,k=args.gmm_components)
                        mixture_sum=tf.reduce_mean(tf.reduce_sum(mixture,axis=1))

                        def summarize_vector(v,n):
                            with tf.name_scope(n):
                                op_metrics['mix/'+n+'/max']=tf.reduce_max(v)
                                op_metrics['mix/'+n+'/min']=tf.reduce_min(v)
                                op_metrics['mix/'+n+'/mean']=tf.reduce_mean(v)

                        summarize_vector(pre_kappa,'pre_kappa')
                        summarize_vector(logits,'logits')
                        summarize_vector(w,'w')
                        try:
                            summarize_vector(pow,'pow')
                        except:
                            pass

                        op_metrics['mix/sum']=mixture_sum

                        for k in [0,1,2]:
                            topk=vals[:,min(k,args.gmm_components-1)]
                            op_metrics['mix/top'+str(k)+'_loss']=topk
                        for p in [0.0,0.25,0.50,0.75]:
                            topp=vals[:,int((1.0-p)*args.gmm_components)-1]
                            op_metrics['mix/percentile_'+str(p)]=topp

                    main_component=mixture
                    main_component=tf.where(
                            tf.equal(tf.reduce_max(mixture, axis=1, keep_dims=True), mixture),
                            tf.constant(1.0, shape=mixture.shape),
                            tf.constant(0.0, shape=mixture.shape)
                            )
                    main_component_reshape=tf.reshape(main_component,[args.batchsize,1,args.gmm_components])

                    gps = tf.reduce_sum(main_component_reshape*pre_mu_gps,axis=2)
                    op_lat = gps[:,0]
                    op_lon = gps[:,1]
                    #op_gps_rad = tf.reduce_sum(main_component_reshape*pre_mu_gps,axis=1)
                    #op_gps_rad = tf.tensordot(main_component,pre_mu,axes=[1,0])
                    #op_lat = tf.atan(op_gps_rad[:,0])*360/2/math.pi
                    #op_lon = tf.atan(op_gps_rad[:,1])*360/math.pi
                    #gps = tf.stack([op_lat,op_lon])
                    #gps = tf.Print(gps,[gps,op_gps_rad,loss])

                    op_outputs['aglm_mix/pre_mu_gps']=pre_mu_gps
                    op_outputs['aglm_mix/mu']=mu
                    op_outputs['aglm_mix/pre_kappa']=pre_kappa
                    op_outputs['aglm_mix/kappa']=kappa
                    op_outputs['aglm_mix/mixture']=mixture
                    op_outputs['aglm_mix/log_loss']=log_loss
                op_outputs['gps']=gps

                # common outputs

                op_lat_rad = op_lat/360*2*math.pi
                op_lon_rad = op_lon/360*math.pi

                hav = lambda x: tf.sin(x/2)**2
                squared_angular_dist = ( hav(op_lat_rad-op_lat_rad_)
                    +tf.cos(op_lat_rad)*tf.cos(op_lat_rad_)*hav(op_lon_rad-op_lon_rad_)
                    )

                # radius of earth = 3959 miles, 6371 kilometers
                op_dist = 2*6371*tf.asin(tf.sqrt(tf.maximum(0.0,squared_angular_dist)))
                op_dist_ave = tf.reduce_sum(op_dist)/args.batchsize

                op_delta_x = tf.cos(op_lat_rad)*tf.cos(op_lon_rad)-tf.cos(op_lat_rad_)*tf.cos(op_lon_rad_)
                op_delta_y = tf.cos(op_lat_rad)*tf.sin(op_lon_rad)-tf.cos(op_lat_rad_)*tf.sin(op_lon_rad_)
                op_delta_z = tf.sin(op_lat_rad) - tf.sin(op_lat_rad_)
                op_chord = tf.sqrt(epsilon + op_delta_x**2 + op_delta_y**2 + op_delta_z**2)

                # set loss function
                if args.pos_loss=='l2':
                    op_loss = tf.reduce_sum((gps - gps_) * (gps - gps_))
                if args.pos_loss=='chord':
                    op_loss = tf.reduce_sum(op_chord)/args.batchsize
                if args.pos_loss=='dist':
                    op_loss = op_dist_ave
                if args.pos_loss=='dist2':
                    op_loss = tf.reduce_sum(op_dist*op_dist)/args.batchsize
                if args.pos_loss=='dist_sqrt':
                    op_loss = tf.reduce_sum(tf.sqrt(op_dist))/args.batchsize
                if args.pos_loss=='angular':
                    op_loss = tf.reduce_sum(squared_angular_dist)

                if not args.pos_type=='aglm_mix' or args.gmm_distloss:
                    op_losses['pos_loss']=op_loss/1000
                op_metrics['optimization/dist']=op_dist_ave

                def mk_metric(basename,weights):
                    total_weights = tf.reduce_sum(weights)
                    op_dist_ave = tf.cond(total_weights>0,
                        lambda: tf.reduce_sum(weights*op_dist)/total_weights,
                        lambda: 0.0
                        )
                    op_metrics[basename+'dist']=(total_weights,op_dist_ave)
                    def mk_threshold(threshold):
                        op_threshold = tf.sign(op_dist-threshold)/2+0.5
                        op_threshold_ave = tf.cond(total_weights>0,
                            lambda: tf.reduce_sum(weights*op_threshold)/total_weights,
                            lambda: 0.0
                            )
                        name=basename+'k'+str(threshold)
                        op_metrics[name]=(weights,op_threshold_ave)
                    mk_threshold(10)
                    mk_threshold(50)
                    mk_threshold(100)
                    mk_threshold(500)
                    mk_threshold(1000)
                    mk_threshold(2000)
                    mk_threshold(3000)
                make_summaries(mk_metric)

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

        if args.loss_weights == 'ave':
            op_loss = tf.reduce_mean(op_losses.values())

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

        if args.predict_lang:
            op_loss += op_losses['lang_xentropy']

        op_metrics['optimization/op_loss']=op_loss

        # add regularizers
        with tf.name_scope('l2_regularization'):
            vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            for var in vars:
                regularizers.append(args.l2*tf.nn.l2_loss(var))

        op_loss_regularized=op_loss+tf.reduce_sum(regularizers)
        op_losses['optimization/op_loss_regularized']=op_loss_regularized

        return op_metrics,op_loss_regularized,op_losses,op_outputs

########################################

def metrics2summaries(args,op_metrics):
    import tensorflow as tf
    summaries={}
    with tf.name_scope('streaming_mean'):
        for k,v in op_metrics.iteritems():
            try:
                (weights,metric)=v
                #metric *= weights/float(args.batchsize)
            except:
                metric=v
            summaries[k]=tf.contrib.metrics.streaming_mean(metric,name=k)
    return summaries


################################################################################

def preprocess_text(args,str):
    import unicodedata

    # simplify the unicode representation
    str=unicodedata.normalize('NFKC',unicode(str.lower()))

    # remove urls if needed
    if args.remove_urls:
        import re
        str = re.sub(r'https?:\/\/\S*', '', str, flags=re.MULTILINE)

    return str

################################################################################

def json2dict(args,str):
    import sklearn.feature_extraction.text
    hv=sklearn.feature_extraction.text.HashingVectorizer(n_features=2**args.bow_hashsize,norm=None)

    import simplejson as json
    import unicodedata
    import numpy as np
    data=json.loads(str)
    data['text']=preprocess_text(args,data['text'])

    batch_dict={}

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
    batch_dict['lang_']=myhash.lang2int(data['lang'])
    try:
        country_code=myhash.country2int(data['place']['country_code'])
    except:
        country_code=0
    batch_dict['country_']=np.array([country_code])

    # get inputs
    if 'bow' in args.input:
        batch_dict['hash_']=hv.transform([data['text']])

    if 'cnn' in args.input:
        encodedtext=np.zeros([1,tweetlen,args.cnn_vocabsize])

        encodedtext_index=0
        for i in range(min(tweetlen,len(data['text']))):
            #pass
            bitmasks=myhash.unicode2bitmask(
                data['text'][i],
                hashsize=args.cnn_vocabsize,
                naive=args.text_naive,
                multichar_init_bit=args.text_multichar_init_bit,
                latin_bit=args.text_latin_bit,
                transliterate=args.text_transliterate,
                hashsize_combiners=args.text_hashsize_combiners,
                debug=False
                )
            for bitmask in bitmasks:
                j=0
                for bit in bitmask:
                    if bit and encodedtext_index<tweetlen:
                        encodedtext[0][encodedtext_index][j]=1
                    j+=1
                encodedtext_index+=1

                #char=ord(data['text'][i])
                #index=(5381*char + 88499*k)%args.cnn_vocabsize
                #encodedtext[0][i][index]=1

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
            loc_code=myhash.loc2int(data['place']['full_name'])
        except:
            loc_code=0
        batch_dict['loc_']=np.array([loc_code])

    return batch_dict

################################################################################

def mk_feed_dict(args,batch):
    import numpy as np
    import scipy as sp
    import tensorflow as tf
    from collections import defaultdict
    feed_dict = {}
    batch_dict=defaultdict(list)

    for data in batch:
        for k in data:
            batch_dict[k].append(data[k])

    feed_dict['lang_:0'] = np.vstack(batch_dict['lang_'])
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
        feed_dict[hash_] = mkSparseTensorValue(sp.sparse.vstack(batch_dict['hash_']))

    if 'cnn' in args.input:
        feed_dict['text_:0'] = np.vstack(batch_dict['text_'])

    if 'time' in args.input:
        feed_dict['timestamp_ms_:0'] = np.vstack(batch_dict['timestamp_ms_'])

    #if 'country' in args.output:
    feed_dict['country_:0'] = np.vstack(batch_dict['country_'])

    if 'pos' in args.output:
        feed_dict['gps_:0'] = np.vstack(batch_dict['gps_'])

    if 'loc' in args.output:
        feed_dict['loc_:0'] = np.vstack(batch_dict['loc_'])

    return feed_dict

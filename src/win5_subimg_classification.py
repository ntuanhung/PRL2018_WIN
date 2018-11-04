import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import logging
from datetime import datetime
import argparse
import WIN5_SUBIMG
from imageData import ImageData
# from const import *
import time

def addDest2Logger(logger, prefix):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    log_filename = prefix + '@'+datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
    fh = logging.FileHandler(log_filename+'.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return log_filename
    
def sec_to_time(sec):
    days=-1
    hrs=-1
    mins=-1
    result=''

    days = int(sec / 86400)
    sec -= 86400*days

    hrs = int(sec / 3600)
    sec -= 3600*hrs

    mins = int(sec / 60)
    sec -= 60*mins
    if days>0:
        result = str(days)+' day(s) ' + str(hrs)+ ' hour(s) ' + str(mins)+' min(s) ' + str(round(sec,2))+ ' sec(s)'
    elif hrs>0:
        result = str(hrs)+ ' hour(s) ' + str(mins)+' min(s) ' + str(round(sec,2))+ ' sec(s)'
    elif mins>0:
        result = str(mins)+' min(s) ' + str(round(sec,2))+ ' sec(s)'
    else:
        result = str(round(sec,2))+ ' sec(s)'
    return result
    
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
if __name__ == "__main__":
    ### PARSE ARGUMENTS
    parser = argparse.ArgumentParser(description=' Writer Identification Network - 5 layers')
    ## Overall parameters
    parser.add_argument('--user_name', '-un', type=str, default="hung")
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--gpu_mem_ratio', '-gmr', type=float, default=0.4)
    parser.add_argument('--img_size', '-im', type=int, default=64)
    parser.add_argument('--image_type', '-it', type=str, default="BIN")
    parser.add_argument('--num_writers', '-nw', type=int, default=50)
    parser.add_argument('--selection_mode', '-sm', type=str, default="SAME")
    parser.add_argument('--dataset_name', '-dn', type=str, default="JEITA-HP")
    parser.add_argument('--directory_path', '-dp', type=str, default='C:\\PRL2018_WIN\\')

    ## These parameters are changed in our experiments
    parser.add_argument('--num_training_patterns', '-ntp', type=int, default=100)
    parser.add_argument('--local_feature', '-lf', type=str, default="subimg")
    parser.add_argument('--agg_mode', '-am', type=str, default="average")
    parser.add_argument('--writer_per_batch', '-wpb', type=int, default=50)
    parser.add_argument('--n_tuple', '-nt', type=int, default=20)
    parser.add_argument('--train_num_permutations', '-trnp',type=int, default = 20)
    parser.add_argument('--kmax', '-k', type=int, default = 50)
    
    parser.add_argument('--valid_num_permutations', '-vanp',type=int, default = 5)
    parser.add_argument('--test_num_permutations', '-tenp',type=int, default = 5)
    parser.add_argument('--eval_num_permutations', '-evnp',type=int, default = 20)
    parser.add_argument('--max_epochs', '-me', type=int, default=10000)
    parser.add_argument('--max_no_best', '-mnb', type=int, default=20)
    
    ## Parameters for training/evaluating
    parser.add_argument('--training', '-t', type=str2bool, default=True)
    parser.add_argument('--resume', '-r', type=str, default="none")
    parser.add_argument('--global_step_start', '-gss', type=int, default=0)
    parser.add_argument('--global_step_eval', '-gse', type=int, default=0)
    parser.add_argument('--model_name', '-mn', type=str, default="")
    parser.add_argument('--lr', '-l', type=float, default=1e-4)
    parser.add_argument('--use_valid_data', '-uvd', type=str2bool, default=True)
    parser.add_argument('--use_test_data', '-utd', type=str2bool, default=True)
    parser.add_argument('--eval_test_data', '-etd', type=str2bool, default=True)
    
    # parser.add_argument('--out', '-o', default='result',
                        # help='Output directory')
    # parser.add_argument('--seed', '-s', type=int, default=0)
    # parser.add_argument('--step_size', '-ss', type=int, default=50000)
    # parser.add_argument('--iteration', '-i', type=int, default=70000)
    
    args = parser.parse_args()
    
    prefix = 'win5_'+args.local_feature+'_' + args.agg_mode
    directory_path= args.directory_path
    dataset_name = args.dataset_name
    dataFolderPath=directory_path+dataset_name+"\\images"
    
    ### SETUP LOGGER
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_filename = addDest2Logger(logger, prefix)
    
    model_filename= args.directory_path + dataset_name + '\\models\\'+ log_filename
    log_filename  = args.directory_path + dataset_name + '\\logs\\' + log_filename
    output_filename = log_filename+'.output'
    print(log_filename)
    logger.info(log_filename)
    logger.info(args)
    
    ### SETUP DIRECTORIES
    nb_class = args.num_writers
    nb_training_patterns = args.num_training_patterns
    selection_mode = args.selection_mode
    suffix_input_filename = str(nb_class)+'users_'+str(nb_training_patterns)+'patPerUser_'+selection_mode+'_'+dataset_name+'.txt'
    train_data = ImageData(img_dir=dataFolderPath, meta_data=directory_path+dataset_name+'\\configs\\train-files_'+suffix_input_filename)
    valid_data = ImageData(img_dir=dataFolderPath, meta_data=directory_path+dataset_name+'\\configs\\valid-files_'+suffix_input_filename)
    test_data  = ImageData(img_dir=dataFolderPath, meta_data=directory_path+dataset_name+'\\configs\\test-files_'+suffix_input_filename)
    
    ### SETUP CONSTANTS    
    user_name = args.user_name
    if args.image_type=="BIN":
        nb_channel = 1
    elif args.image_type=="RGB":
        nb_channel = 3
    TEST = args.use_test_data
    VALID = args.use_valid_data
    EVAL = args.eval_test_data
    
    writer_per_batch = args.writer_per_batch
    
    img_size = args.img_size
    train_data.img_width = img_size
    train_data.img_height= img_size
    valid_data.img_width = img_size
    valid_data.img_height= img_size
    test_data.img_width = img_size
    test_data.img_height= img_size
    
    n_tuple = args.n_tuple
    train_data.n_tuple = n_tuple
    valid_data.n_tuple = n_tuple
    test_data.n_tuple = n_tuple
    
    train_num_permutations = args.train_num_permutations
    valid_num_permutations = args.valid_num_permutations
    test_num_permutations = args.test_num_permutations
    eval_num_permutations = args.eval_num_permutations
    nb_max_epochs = args.max_epochs
    nb_max_no_best= args.max_no_best
    
    if args.gpu >= 0:
        use_GPU = True
        os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem_ratio, allow_growth=True)
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        logger.info("use gpu %d"%args.gpu)
    else:
        logger.info("no use gpu")
        sess = tf.InteractiveSession()
    global_step = tf.contrib.framework.get_or_create_global_step()

    logger.info('Create new model')
    x, y_ = WIN5_SUBIMG.input(nb_class, img_size, nb_channel, n_tuple)
    
    if args.agg_mode == "average":
        y_conv = WIN5_SUBIMG.average_pool_inference(x, y_, nb_class, img_size, nb_channel, n_tuple, logger=logger)
    elif args.agg_mode == "kmax":
        y_conv = WIN5_SUBIMG.kmax_pool_inference(x, y_,nb_class, img_size, nb_channel, n_tuple, kmax=args.kmax, logger=logger)
    elif args.agg_mode == "max":
        y_conv = WIN5_SUBIMG.kmax_pool_inference(x, y_, kmax=1, logger=logger)
    else:
        print("ERROR not found the appropriate aggregation mode...")
        sys.exit(0)
        
    WIN5_SUBIMG.define_additional_variables()
    loss = WIN5_SUBIMG.loss(y_conv, y_, mode="cross_entropy")
    train_step = WIN5_SUBIMG.train_op(loss, global_step, args.lr)
    regul_loss = tf.get_collection('regul_loss')[0]
    # triplet_loss = tf.get_collection('triplet_loss')[0]
    # same_loss = tf.get_collection('triplet_loss')[1]
    # different_loss = tf.get_collection('triplet_loss')[2]
    #total_loss = tf.get_collection('total_loss')[0]
    #learning_rate = tf.get_collection('learning_rate')[0]
    #train_step = tf.get_collection('train_step')[0]
    # keep_prob_fc = tf.get_collection('keep_prob_fc')[0]

    # x = tf.get_collection('x')[0]
    # y_ = tf.get_collection('y_')[0]

    # y_conv = tf.get_collection('y_out')[0]
    # is_training = tf.get_collection('is_training')[0]

    correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
    true_pred = tf.reduce_sum(tf.cast(correct_prediction, dtype=tf.float32))

    correct_prediction_top5 = tf.nn.in_top_k(y_conv, tf.cast(tf.argmax(y_, 1), "int32"), 5)
    true_pred_top5 = tf.reduce_sum(tf.cast(correct_prediction_top5, dtype=tf.float32))

    # en_free_all, en_copied_all, en_free_valid, en_free_test = PrepareData()

    ## VARIABLES
    best_valid_acc = 0
    last_epoch = -1
    noBestEpochs = 0
    best_val_loss_epoch=-1
    
    bestAcc_saver = tf.train.Saver( max_to_keep=1)
    latest_saver=tf.train.Saver( max_to_keep=1)
    sess.run(tf.global_variables_initializer())
    if args.resume!="none":
        try:
            #saver.restore(sess, MODEL_PATH + '-' + str(global_step_init))
            bestAcc_saver.restore(sess, args.resume+"_bestAcc.ckpt"+ '-' + str(args.global_step_start))
            logger.info("Resuming training process with "+args.resume+"_bestAcc.ckpt"+ '-' + str(args.global_step_start)+" model restored.")
        except:
            logger.info ('warning')
    print("args.training="+str(args.training) )
    if args.training==True:
        start_train = time.clock()
        for epoch in range(args.global_step_start+1, nb_max_epochs):
            start_time = time.clock()
            '''         TRAIN MODEL         '''
            avg_ttl = []
            avg_rgl = []
            avg_triplet = []
            avg_same = []
            avg_diff = []
            nb_true_pred_train = 0
            num_samples_train = 0
            logger.info("---------------- Epoch: %d ----------------"%epoch)
            pbar = tqdm(total=train_num_permutations * len(range(train_data.images_idx.shape[1] // train_data.n_tuple)))
            for p_perm in range(train_num_permutations):
                train_data.shuffleImages()
                # print("train_data.images_idx.shape[1], train_data.n_tuple", train_data.images_idx.shape[1], train_data.n_tuple)
                for start_image_index in range(train_data.images_idx.shape[1] // train_data.n_tuple):
                    train_data.shuffleWriter()
                    #nb_of_writer_batch = len(train_data.writer_index) // writer_per_batch
                    #batch_idx = 0
                    for batch_data in train_data.generate_minibatch_JEITA(writer_per_batch, start_image_index * train_data.n_tuple, mode = "train", logger=logger):
                     ## shuffle the order of writers in training data
                        #for i in range(nb_of_writer_batch):
                            # batch_idx += 1
                            # print('Training on batch ',str(batch_idx),', ',str(start_image_index), '/',str(nb_of_writer_batch))
                            # print('Training on batch ',str(batch_idx),', ',str(start_image_index), '/',str(nb_of_writer_batch), end = '\r')
                            # top = i * batch_size
                            # bot = min((i+1) * batch_size,len(train_image))
                        
                        x_batch, y_batch = batch_data[0], batch_data[1]
                        #print(y_batch)
                        num_samples_train += x_batch.shape[0]
                        
                        #triplet, rgl, ttl, sl, dl, _ = sess.run(# [triplet_loss, regul_loss, total_loss, same_loss, different_loss, train_step],
                        rgl, ttl, _, tr_pred  = sess.run([regul_loss, loss, train_step, true_pred], feed_dict= {x: x_batch, y_: y_batch})
                        #, is_training:True , keep_prob_fc: 0.8})
                        avg_rgl.append(rgl)
                        avg_ttl.append(ttl) #cross_entropy_loss
                        nb_true_pred_train += tr_pred
                        # avg_triplet.append(triplet)
                        # avg_same.append(sl)
                        # avg_diff.append(dl)
                        # yconv_vect = y_conv.eval(session=sess, feed_dict= {x: x_batch, y_: y_batch, is_training:True, keep_prob_fc: 0.5})
                        # print(yconv_vect)
                        # nb_true_pred += true_pred.eval(session=sess, feed_dict= {x: x_batch, y_: y_batch, is_training:True, keep_prob_fc: 0.8})
                    pbar.update(1)
            pbar.close()
            
            sum_rgl = np.average(avg_rgl)
            sum_ttl = np.average(avg_ttl)
            # sum_triplet = np.average(avg_triplet)
            # sum_same = np.average(avg_same)
            # sum_diff = np.average(avg_diff)
            train_accuracy = nb_true_pred_train * 1.0 / num_samples_train
            if (epoch != 0):
                latest_saver.save(sess, model_filename+"_last.ckpt", global_step=epoch)
            
            valid_acc = []
            test_acc = []
            avg_valid_acc = 0.0
            std_valid_acc = 0.0
            avg_test_acc = 0.0
            std_test_acc = 0.0
            if (VALID == True):
                pbar = tqdm(total=valid_num_permutations*len(range(valid_data.images_idx.shape[1] // valid_data.n_tuple)))
                for p_perm in range(valid_num_permutations):
                    num_samples = 0
                    nb_true_pred = 0
                    valid_data.shuffleImages()
                    for start_image_index in range(valid_data.images_idx.shape[1] // valid_data.n_tuple):                    
                        # nb_of_writer_batch = len(valid_data.writer_index) // writer_per_batch
                        # batch_idx = 0
                        valid_data.shuffleWriter()
                        for batch_data in valid_data.generate_minibatch_JEITA(writer_per_batch, start_image_index * valid_data.n_tuple, mode = "valid", logger=logger):
                            # batch_idx += 1
                            #print('Test on batch ',str(batch_idx),', ',str(start_image_index), '/',str(nb_of_writer_batch), end="\r")
                            x_batch = batch_data[0]
                            y_batch = batch_data[1]
                            
                            num_samples += x_batch.shape[0]
                            nb_true_pred += true_pred.eval(session=sess, feed_dict={x: x_batch, y_: y_batch}) #, is_training: False, keep_prob_fc: 1.0})
                        pbar.update(1)
                    valid_acc.append(nb_true_pred * 1.0 / num_samples)
                pbar.close()
                avg_valid_acc = np.average(np.array(valid_acc))
                std_valid_acc = np.std(np.array(valid_acc))
                # print('Valid accuracy: %.5f with std=%.5f'%( avg_valid_acc*100, std_valid_acc*100))
                if avg_valid_acc > best_valid_acc:
                    # print("Network with best accuracy")
                    best_valid_acc = avg_valid_acc
                    best_val_loss_epoch = epoch
                    noBestEpochs = 0
                    bestAcc_saver.save(sess, model_filename+"_bestAcc.ckpt", global_step=epoch)
                else:
                    noBestEpochs += 1
            '''-----------------------------------------------------------------------------------------------------------'''
            if (TEST == True):
                '''         TEST MODEL      '''
                test_label = []
                pbar = tqdm(total=test_num_permutations*len(range(test_data.images_idx.shape[1] // test_data.n_tuple)))
                for p_perm in range(test_num_permutations):
                    nb_true_pred = 0
                    num_samples = 0
                    test_data.shuffleImages()
                    for start_image_index in range(test_data.images_idx.shape[1]//test_data.n_tuple):                    
                        #nb_of_writer_batch = len(test_data.writer_index) // writer_per_batch
                        #batch_idx = 0
                        test_data.shuffleWriter()
                        for batch_data in test_data.generate_minibatch_JEITA(writer_per_batch, start_image_index*test_data.n_tuple, mode = "test", logger=logger):
                            #batch_idx += 1
                            #print('Test on batch ',str(batch_idx),', ',str(start_image_index), '/',str(nb_of_writer_batch), end="\r")
                            x_batch, y_batch = batch_data[0], batch_data[1]
                            num_samples += x_batch.shape[0]
                            nb_true_pred += true_pred.eval(session=sess, feed_dict={x: x_batch, y_: y_batch})#, is_training: False, keep_prob_fc: 1.0})
                        pbar.update(1)
                    test_acc.append(nb_true_pred*1.0/num_samples)
                pbar.close()
                avg_test_acc = np.average(np.array(test_acc))
                std_test_acc = np.std(np.array(test_acc))
            
            end_time = time.clock()
            logger.info('training time  = ' + str(end_time - start_time)[0:10] + ' sec(s)')
            logger.info(str(train_accuracy)+ ' ' + str(nb_true_pred_train) + '/' + str(num_samples_train))
            # if nb_true_pred > num_samples:
            # print(train_accuracy, nb_true_pred, num_samples)
            logger.info('Total loss: ' + str(sum_ttl) + '. L2-loss: ' + str(sum_rgl))
            # print('Triplet-loss: ' + str(sum_triplet) + '. Same loss: ' + str(sum_same) + '. Different loss: ' + str(sum_diff))
            logger.info('Train accuracy: %.3f%%'%(train_accuracy * 100))
            
            if len(valid_acc)>0:
                logger.info('Valid accuracy: %.3f%% with std=%.3f'%( avg_valid_acc*100, std_valid_acc*100))
            if len(test_acc)>0:
                logger.info('Test accuracy: %.3f%% with std=%.3f'%( avg_test_acc*100, std_test_acc*100))
            if noBestEpochs == 0:
                logger.info("Best accuracy network saved.")
            else:
                logger.info("No best epoch = %d"%noBestEpochs)
            
            if noBestEpochs >= nb_max_no_best:
                last_epoch = epoch
                break
        end_train=time.clock()
        # summary training process
        logger.info('--------------------------------------------')
        if last_epoch!=-1:
            logger.info('training stop after '+ str(last_epoch+1)+' epoch(s)')
        else:
            logger.info('training stop after '+ str(nb_max_epochs)+' epoch(s)')
        takeTime = sec_to_time(end_train-start_train)
        logger.info('total time takes ' + takeTime)
        logger.info('best network on val_loss ' + str(best_valid_acc)+ ' at epoch '+ str(best_val_loss_epoch+1))
    
    if EVAL==True:
        if args.training==True :
            model_name = model_filename
            if best_val_loss_epoch>=0:
                global_step = best_val_loss_epoch
            elif last_epoch>=0:
                global_step = last_epoch
            else:
                global_step = nb_max_epochs - 1
        else:
            global_step = args.global_step_eval
            model_name = args.model_name
        try:
            #saver.restore(sess, MODEL_PATH + '-' + str(global_step_init))
            bestAcc_saver.restore(sess, model_name+"_bestAcc.ckpt"+ '-' + str(global_step))
            #logger.info(model_name+"_bestAcc.ckpt"+ '-' + str(global_step)+" model restored.")
        except:
            logger.info ('warning')
        
        eval_acc=[]
        pbar = tqdm(total=eval_num_permutations*len(range(test_data.images_idx.shape[1] // test_data.n_tuple)))
        for p_perm in range(eval_num_permutations):
            nb_true_pred = 0
            num_samples = 0
            test_data.shuffleImages()
            for start_image_index in range(test_data.images_idx.shape[1] // test_data.n_tuple):                    
                nb_of_writer_batch = len(test_data.writer_index) // writer_per_batch
                batch_idx = 0
                test_data.shuffleWriter()
                for batch_data in test_data.generate_minibatch_JEITA(writer_per_batch, start_image_index, mode = "eval", logger=logger):
                    batch_idx += 1
                    #print('Test on batch ',str(batch_idx),', ',str(start_image_index), '/',str(nb_of_writer_batch), end="\r")
                    x_batch, y_batch = batch_data[0], batch_data[1]
                    num_samples += x_batch.shape[0]
                    nb_true_pred += true_pred.eval(session=sess, feed_dict={x: x_batch, y_: y_batch})#, is_training: False, keep_prob_fc: 1.0})
                pbar.update(1)
            eval_acc.append(nb_true_pred*1.0/num_samples)
        pbar.close()
        avg_eval_acc = np.average(np.array(eval_acc))
        std_eval_acc = np.std(np.array(eval_acc))
        logger.info('Eval accuracy: %.3f%% with std=%.3f'%( avg_eval_acc*100, std_eval_acc*100))
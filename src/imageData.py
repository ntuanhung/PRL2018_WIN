from header import *

class ImageData():
    def __init__(self, img_dir='', meta_data='', labels=[]):
        self.img_dir = img_dir
        self.meanPixel = []
        
        self.image_mode= 'binary'
        self.framework = 'Tensorflow'
        self.inChannel = 1
        self.writer_idx= []
        self.images_idx= []
        self.writer_index = []
        self.images_index = []
        
        if meta_data!='':
            assert meta_data.endswith('JEITA-HP.txt')
            # self.meta_data = pd.read_csv(meta_data, sep=',')
            # #print(self.meta_data)
            with open(meta_data) as f:
                for line in f:
                    data = line.split()
                    self.writer_idx.append(int(data[0]))
                    tmp_images = []
                    for elem_idx in range(1, len(data)):
                        tmp_images.append(int(data[elem_idx]))
                    self.images_idx.append(tmp_images)

            self.writer_index = range(len(self.writer_idx))
            self.images_index = []
            for ii in range(len(self.images_idx)):
                tmp_images = self.images_idx[ii]
                self.images_index.append(tmp_images)
            
            self.writer_idx = np.array(self.writer_idx, dtype=np.int32)
            self.images_idx = np.array(self.images_idx, dtype=np.int32)
            self.img_width = 64
            self.img_height= 64
            self.n_tuple = 20
            # self.index = np.array(self.meta_data.index)
            # self.name_images = list(self.meta_data['ID'])
            # self.data_labels = np.array([labels.index(label) for label in self.meta_data['Unicode']])
            # self.data_images = []
            self.split = True
        else:
            self.meta_data=[]
            self.split = False
            self.index = []
        #self.data_images = []
        

    def loadAndPreprocessImages(self, img_size = 224):
        indexInfo = self.meta_data
        self.data_images = []
        for f in list(indexInfo['file_name']):
            image = Image.open(os.path.join(self.img_dir, f))
            # image = image.resize((img_size, img_size))
            self.data_images.append(np.array(image))
        self.data_images = np.array(self.data_images)
        self.data_images = self.data_images.transpose((0,3,1,2)) ## use for pretrain model by caffe
        self.data_images = self.data_images.astype(np.float32)
        if 'category_id' in indexInfo.columns:
            self.data_labels = np.array(list(indexInfo['category_id']))
            self.data_labels = self.data_labels.astype(np.int32)
        else:
            i=0
        logger.info('max value = '+str(np.max(self.data_images)))
        logger.info('min value = '+str(np.min(self.data_images)))
        
    def loadPreprocessedImages(self):
        indexInfo = self.meta_data
        self.data_images = []
        cnt=0
        for f in list(indexInfo['file_name']):
            image = ndimage.imread(os.path.join(self.img_dir, f[:-3]+'png'))
            self.data_images.append(np.array(image))
            misc.imsave(os.path.join(self.img_dir, f), self.data_images[0])
            if cnt>1:
                break
            else:
                cnt+=1
        
        if 'category_id' in indexInfo.columns:
            self.data_labels = np.array(list(indexInfo['category_id']))
            self.data_labels = self.data_labels.astype(np.int32)
        else:
            i=0
        logger.info('max value = '+str(np.max(self.data_images)))
        logger.info('min value = '+str(np.min(self.data_images)))
    def loadPreprocessedMeanPixelBinaryFile(self, infile, logger=None):
        with gzip.open(infile+'.dat','rb') as f:
            self.data_images = pickle.load(f)
            self.data_images = self.data_images.astype(np.float32) #[:,::-1,:,:] #change RGB to BGR to use with caffemodel [nSample, channel, h, w]
            logger.info('loaded file '+infile+'.dat'+' contained data with shape ' + str(self.data_images.shape))
        
    def loadPreprocessedBinaryFile(self, infile, meanImage='', use_mean_pixel=False, logger=None, concat_data=False, use_standarization=False):
        with gzip.open(infile+'.dat','rb') as f:
            tmp_data_images = pickle.load(f)
            tmp_data_images = tmp_data_images.astype(np.float32) #[:,::-1,:,:] #change RGB to BGR to use with caffemodel [nSample, channel, h, w]
            logger.info('loaded file '+infile+'.dat'+' contained data with shape ' + str(tmp_data_images.shape))
        if meanImage!='':
            mean_image=[]
            if '.dat' in meanImage:
                with gzip.open(meanImage,'rb') as f:
                    mean_image = pickle.load(f)
                    mean_image = mean_image.astype(np.float32)
            elif '.npy' in meanImage:
                mean_image = np.load(meanImage).astype(np.float32)
            logger.info('loaded file '+meanImage+' contained data with shape ' + str(mean_image.shape))
            
            mean_pixel=[np.mean(mean_image[i]) for i in range(3)]
            mean_image=np.resize(mean_image, (3,tmp_data_images.shape[2],tmp_data_images.shape[3]))
            logger.info('resized to shape ' + str(mean_image.shape))
            logger.info('avg each channel ' + str(mean_pixel))
            if use_mean_pixel==False and len(mean_image.shape)==3:
                tmp_data_images -= mean_image
                logger.info('subtract for mean_image')
            elif use_mean_pixel==True and len(mean_pixel)==3:
                tmp_data_images = tmp_data_images.transpose((0,2,3,1))
                tmp_data_images -= mean_pixel
                tmp_data_images = tmp_data_images.transpose((0,3,1,2))
                logger.info('subtract for mean_pixel')
        if use_standarization == True:
            tmp_data_images /= 255.
        # change RGB to BGR to use with caffemodel [nSample, channel, h, w]
        tmp_data_images = tmp_data_images[:,::-1,:,:] 
        logger.info('Changed images from RGB to BGR for compatible with caffemodel')
        
        if(concat_data == False):
            self.data_images=tmp_data_images
            logger.info('Loaded data from file with shape '+str(self.data_images.shape))
        else:
            logger.info('Concat data from ' + str(self.data_images.shape) )
            self.data_images = np.concatenate((self.data_images, tmp_data_images ), axis=0)
            logger.info('to shape ' + str(self.data_images.shape))
            
        if len(self.meta_data)>0 and 'category_id' in self.meta_data.columns:
            self.data_labels = np.array(list(self.meta_data['category_id']))
            self.data_labels = self.data_labels.astype(np.int32)
        elif len(self.meta_data)==0:
            self.index = np.arange(self.data_images.shape[0])
        
    def loadPreprocessedLeveldb(self, infile):
        hdf = pd.HDFStore(infile+".hdf5") # might also take a while
        
    def loadPreprocessedHDF5File(self, infile):
        hdf = pd.HDFStore(infile+".hdf5") # might also take a while
        read_dict = hdf["saved_data"] # Be careful of the way changes to
        self.data_images=read_dict['saveImages']
        self.data_labels=read_dict['saveLabels']
        hdf.close()
        
    def writeData2BinaryFileByPickle(self, outfile):
        # with gzip.open(outfile+'.dat', 'wb', compresslevel=9) as f:
            # pickle.dump(np.array(self.data_images,dtype=np.uint8), f, protocol=pickle.HIGHEST_PROTOCOL)
        #meanPixel = np.array([int(x) for x in self.meanPixel])
        data_images = np.zeros((len(self.name_images), self.inChannel, 64, 64))
        count = 0
        for file in range(len(self.name_images)):
            file_name = self.img_dir+str(self.name_images[file]).rjust(6,'0')+'.png'
            curr_image = np.array(Image.open(file_name), dtype=np.uint8)
            if int(curr_image.size / (64*64))<self.inChannel:
                curr_image = np.array([curr_image, curr_image, curr_image]).transpose(1,2,0)
            #print(np.min(curr_image),np.max(curr_image))
            data_images[count] = (curr_image - self.meanPixel).transpose(2, 0, 1)
            count += 1
        print(data_images.shape)
        print(np.min(data_images), np.max(data_images))
        with gzip.open(outfile+'.dat', 'wb', compresslevel=9) as f:
            pickle.dump(np.array(data_images,dtype=np.uint8), f, protocol=pickle.HIGHEST_PROTOCOL)
            
    def writeData2BinaryFileByHDF5(self, outfile):    
        h5f = h5py.File('data_hdf5_latest.h5', 'w', libver='latest')
        h5f.create_dataset('train_data', data=np.array(self.data_images,dtype=np.uint8), compression="gzip", compression_opts=9)
        h5f.close()
        
    def writeData2BinaryFileByLeveldb(self, outfile):    
        db = leveldb.LevelDB('./'+outfile+'_leveldb')
        batch = leveldb.WriteBatch()
        indexInfo = self.meta_data
        [batch.Put(str.encode(elem), np.array(self.data_images[count], dtype=np.uint8).tostring()) for count, elem in enumerate(indexInfo['file_name'])]
        db.Write(batch, sync=True)
        db.CompactRange()
        
    def shuffleWriter(self):
        self.writer_index = np.random.permutation(np.array(range(self.writer_idx.shape[0]), dtype=np.int32))
        
    def shuffleImages(self):
        self.images_index = []
        for ii in range(self.images_idx.shape[0]):
            tmp_images = np.random.permutation(self.images_idx[ii])
            self.images_index.append(tmp_images)
        
    def shuffle(self):
        if self.split == True:
            self.train_index = np.random.permutation(self.train_index)
        else:
            self.index = np.random.permutation(self.index)

    def split_train_val(self, train_size, logger=None):
        logger.info('split_train_val is invoked')
        self.train_index = np.random.choice(self.index, train_size, replace=False)
        self.valid_index = np.array([i for i in self.index if i not in self.train_index])
        #logger.debug('train_index '+str(self.train_index))
        #logger.debug('valid_index '+str(self.valid_index))
        #count = np.zeros((25))
        #for l in range(len(count)):
        #    count[l]=len([x for x in self.valid_index if self.data_labels[x]==l])
        #logger.debug('number each classes '+str(count))
        logger.info('number of train samples = '+str(len(self.train_index)))
        logger.info('number of valid samples = '+str(len(self.valid_index)))
        self.split = True
    
    def split_train_val_balancing(self, train_ratio, outfile, logger=None):
        logger.info('split_train_val_balancing is invoked')
        if(len(self.data_labels)>0):
            max_label = np.max(self.data_labels)
            #logger.debug(str(max_label))
            train_ind = []
            valid_ind = []
            for l in range(max_label + 1):
                label_ind = np.array([i for i, x in enumerate(self.data_labels) if x==l], dtype=np.int32)
                #logger.debug(str(label_ind))
                train_size = int(len(label_ind)*train_ratio)
                train_ind += list( np.random.choice(label_ind, train_size, replace=False) )
                valid_ind += list( np.array([i for i in label_ind if i not in train_ind]) )
                print(l)
            self.train_index = np.array(train_ind, dtype=np.int32)
            self.valid_index = np.array(valid_ind, dtype=np.int32)
            #logger.debug(str(self.train_index))
            #logger.debug(str(self.valid_index))
            logger.info('number of train samples = '+str(len(np.unique(self.train_index))) )
            logger.info('number of valid samples = '+str(len(np.unique(self.valid_index))) )
            #count = np.zeros((25))
            #for l in range(len(count)):
            #    count[l]=len([x for x in self.valid_index if self.data_labels[x]==l])
            #logger.debug('number each classes '+str(count))
            np.savetxt(outfile, self.valid_index, fmt='%d', delimiter=',')
            
            self.split = True
        else:
            logger.info('Do not have labels for split balance. Please use split_train_val() method instead!')
    def split_train_val_load(self, file_path, logger=None):
        logger.info('split_train_val_load is invoked with file '+file_path)
        if(len(self.data_labels)>0):
            if len(file_path)<=0 or os.path.isfile(file_path)==False:
                logger.info('File path for splitted_val.txt is not correct : '+file_path)
                return 
            self.valid_index = np.loadtxt(file_path, dtype = np.int32, delimiter=',')
            self.train_index = np.array([i for i in self.index if i not in self.valid_index])
            
            logger.info('number of train samples = '+str(len(np.unique(self.train_index))) )
            logger.info('number of valid samples = '+str(len(np.unique(self.valid_index))) )
            #count = np.zeros((25))
            #for l in range(len(count)):
            #    count[l]=len([x for x in self.valid_index if self.data_labels[x]==l])
            #logger.debug('number each classes '+str(count))
            
            self.split = True
        else:
            logger.info('Do not have labels for split balance. Please use split_train_val() method instead!')
    
    def generate_minibatch_JEITA(self, batchsize, start_image_index=0, mode = None, logger=None):
        i = 0
        index = self.writer_index
        # if(mode=="test"):
            # print(index)
        if self.image_mode == 'binary':
            self.inChannel = 1
        while i >= 0 and i < len(index):
            # batch_images = self.data_images[index[i:min(i+batchsize, len(index))]]
            batch_size = batchsize
            if i + batchsize > len(index):
                batch_size = len(index)-i 
            # print(batch_size)
            batch_images = np.zeros((batch_size, self.n_tuple, self.img_width, self.img_height, self.inChannel), dtype=np.float32)
            batch_labels = np.zeros((batch_size, len(self.writer_idx)), dtype=np.float32)
            count = 0
            # print(start_image_index)
            for k in range(i, i + batch_images.shape[0]):
                # print(count, len(self.images_index[index[k]]) )
                for j in range(self.n_tuple):
                    file_name = self.img_dir+"\\"+str(self.writer_idx[index[k]]) + '_' + str(self.images_index[index[k]][start_image_index+j]).rjust(4,'0')+'.png'
                    # print(file_name)
                    img = Image.open(file_name)
                    curr_image = np.array(img, dtype=np.float32)
                    if len(img.getbands()) != self.inChannel:
                        print("ERROR image contains wrong number channel...", file_name)
                    # if len(curr_image.shape) < 3:
                        # curr_image = np.expand_dims(curr_image, axis=2)
                # #print(curr_image.shape)
                    batch_images[count, j, :, :, 0] = (curr_image) #/255.0) #- self.meanPixel
                batch_labels[count, index[k]] = 1.0
                count += 1
            
            # if header.FRAMEWORK == 'CHAINER':
                # batch_images = batch_images.transpose(0,3,1,2)
            
            yield batch_images, batch_labels
            # if len(self.meta_data)>0 and len(self.data_labels)>0:
                # batch_labels = self.data_labels[index[i:min(i+batchsize, len(index))]]
                # yield batch_images, batch_labels
            # else:
                # #logger.debug('images shape =  '+str(images.shape))
                # yield batch_images
            i+=batchsize
            # print(i)
            
    def generate_minibatch(self, batchsize, mode = None, logger=None):
        i = 0
        if mode == 'train':
            assert self.split == True
            meta_data = self.meta_data.ix[self.train_index]
            index = self.train_index
        elif mode == 'valid':
            assert self.split == True
            meta_data = self.meta_data.ix[self.valid_index]
            index = self.valid_index
        else: ## mode 'unlabeled' or 'test'
            meta_data = self.meta_data
            index = self.index
            #logger.debug('index '+str(index))
        #numBatch = math.ceil(len(index)*1.0 / batchsize)
        #beginIdx = (batchIdx%numBatch) * batchsize
        #endIdx = min(beginIdx + batchsize, len(index))
        #if()
        #print(batchsize)
        #logger.propagate = False
        if self.image_mode != 'raw':
            self.inChannel = 1
        while i >= 0 and i < len(index):
            count = 0
            # batch_images = self.data_images[index[i:min(i+batchsize, len(index))]]
            batch_size = batchsize
            if i+batchsize > len(index):
                batch_size = len(index)-i 
            batch_images = np.zeros((batch_size, 64, 64, self.inChannel),dtype=np.float32)
            for k in range(i, i+batch_images.shape[0]):
                file_name = self.img_dir+str(self.name_images[index[k]]).rjust(6,'0')+'.png'
                #print(file_name)
                img = Image.open(file_name)
                curr_image = np.array(img, dtype=np.float32)
                if len(img.getbands()) != self.inChannel:
                    print("ERROR image contains wrong number channel...", file_name)
                if len(curr_image.shape) < 3:
                    curr_image = np.expand_dims(curr_image, axis=2)
                #print(curr_image.shape)
                batch_images[count] = (curr_image - self.meanPixel)
                count += 1
            
            if header.FRAMEWORK == 'CHAINER':
                batch_images = batch_images.transpose(0,3,1,2)
                
            if len(self.meta_data)>0 and len(self.data_labels)>0:
                batch_labels = self.data_labels[index[i:min(i+batchsize, len(index))]]
                yield batch_images, batch_labels
            else:
                #logger.debug('images shape =  '+str(images.shape))
                yield batch_images
            i+=batchsize
        
    def generate_minibatch_byBatchIdx(self, batchsize, mode = None, batchIdx = 0):
        #i = 0
        if mode == 'train':
            assert self.split == True
            meta_data = self.meta_data.ix[self.train_index]
            index = self.train_index
        elif mode == 'valid':
            assert self.split == True
            meta_data = self.meta_data.ix[self.valid_index]
            index = self.valid_index
        else: ## mode 'unlabeled' or 'test'
            meta_data = self.meta_data
            index = self.index
            #logger.debug('index '+str(index))
        numBatch = math.ceil(len(index)*1.0 / batchsize)
        if self.image_mode != 'raw':
            self.inChannel = 1
        if(batchIdx < numBatch):
            beginIdx = batchIdx * batchsize
            endIdx = min(beginIdx + batchsize, len(index))
            # batch_images = self.data_images[index[beginIdx:endIdx]]
            # print('\n'+mode+':'+str(beginIdx)+'->'+str(endIdx)
                  # +' ['+str(index[beginIdx])+','+str(index[endIdx])+']')
            batch_size = batchsize
            if i+batchsize > len(index):
                batch_size = len(index)-i 
            batch_images = np.zeros((batch_size, 64, 64, self.inChannel),dtype=np.float32)
            for k in range(i, i+batchsize):
                file_name = self.img_dir+str(self.name_images[index[k]]).rjust(6,'0')+'.png'
                #print(file_name)
                img = Image.open(file_name)
                curr_image = np.array(img, dtype=np.float32)
                if len(img.getbands()) != self.inChannel:
                    print("ERROR image contains wrong number channel...", file_name)
                    # curr_image = np.array([curr_image, curr_image, curr_image]).transpose(1,2,0)
                #print(curr_image.shape)
                batch_images[count] = (curr_image - self.meanPixel)
                count += 1
            if header.FRAMEWORK == 'CHAINER':
                batch_images = batch_images.transpose(0,3,1,2)
                
            if len(self.meta_data)>0 and len(self.data_labels)>0:
                batch_labels = self.data_labels[index[beginIdx:endIdx]]
                return batch_images, batch_labels
            else:
                #logger.debug('images shape =  '+str(images.shape))
                return batch_images
        else:
            print('batchIdx >= numBatch '+str(batchIdx)+' >= '+str(numBatch) )
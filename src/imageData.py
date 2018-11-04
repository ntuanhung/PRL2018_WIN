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
            assert meta_data.endswith('.txt')
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
            self.split = True
        else:
            self.meta_data=[]
            self.split = False
            self.index = []
            
    def shuffleWriter(self):
        self.writer_index = np.random.permutation(np.array(range(self.writer_idx.shape[0]), dtype=np.int32))
        
    def shuffleImages(self):
        self.images_index = []
        for ii in range(self.images_idx.shape[0]):
            tmp_images = np.random.permutation(self.images_idx[ii])
            self.images_index.append(tmp_images)

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
            self.split = True
        else:
            logger.info('Do not have labels for split balance. Please use split_train_val() method instead!')
    
    def generate_minibatch_JEITA(self, batchsize, start_image_index=0, mode = None, logger=None):
        i = 0
        index = self.writer_index
        if self.image_mode == 'binary':
            self.inChannel = 1
        while i >= 0 and i < len(index):
            batch_size = batchsize
            if i + batchsize > len(index):
                batch_size = len(index)-i 
            batch_images = np.zeros((batch_size, self.n_tuple, self.img_width, self.img_height, self.inChannel), dtype=np.float32)
            batch_labels = np.zeros((batch_size, len(self.writer_idx)), dtype=np.float32)
            count = 0
            for k in range(i, i + batch_images.shape[0]):
                for j in range(self.n_tuple):
                    file_name = self.img_dir+"\\"+str(self.writer_idx[index[k]]) + '_' + str(self.images_index[index[k]][start_image_index+j]).rjust(4,'0')+'.png'
                    
                    img = Image.open(file_name)
                    curr_image = np.array(img, dtype=np.float32)
                    if len(img.getbands()) != self.inChannel:
                        print("ERROR image contains wrong number channel...", file_name)
                    batch_images[count, j, :, :, 0] = (curr_image) #/255.0) #- self.meanPixel
                batch_labels[count, index[k]] = 1.0
                count += 1
            yield batch_images, batch_labels
            i+=batchsize


from __future__ import division

from abc import ABCMeta, abstractmethod
from itertools import combinations

import numpy as np
import sklearn.preprocessing as skpre
import tensorflow as tf

from utils import Logger, Utils
import sys
import pickle
import copy

from sklearn.decomposition import PCA



class AE(object):
    """ 
        :param input_list: a list of source embedding path
        :param output_path: a string path of output file
        :param log_path: a string path of log file
        :param ckpt_path: a string path of checkpoint file
        :param model_type: the type of model, among M2M
        :param dims: a list of dimensionalities of each source embedding
        :param learning_rate: a float number of the learning rate
        :param batch_size: a int number of the batch size
        :param epoch: a int number of the epoch
        :param activ: a string name of activation function
        :param factors: a list of coefficients of each loss part
        :param noise: a float number between 0 and 1 of the masking noise rate
        :param emb_dim: a int number of meta-embedding dimensionality
        :param oov: a boolean value whether to initialize inputs with oov or not
        :param restore: a boolean value whether to restore checkpoint from local file or 
        :property logger: a logger to record log information
        :property utils: a utility tool for I/O
        :property sess: a tensorflow session
        :property ckpt: a tensorflow saver 
    """

    def __init__(self, **kwargs):
        self.input_list = kwargs['input'] # [path, ...]
        self.output_path = kwargs['output']
        self.log_path = kwargs['log']
        self.ckpt_path = kwargs['ckpt'] + 'model.ckpt'
        self.model_type = kwargs['model']
        self.dims = 2*kwargs['dims']
        self.learning_rate = kwargs['learning_rate']
        self.batch_size = kwargs['batch']
        self.epoch = kwargs['epoch']
        self.activ = kwargs['activ']
        self.factors = kwargs['factors']
        self.noise = kwargs['noise']
        self.emb_dim = kwargs['emb']
        self.oov = kwargs['oov']
        self.restore = kwargs['restore']

        self.logger = Logger(self.model_type, self.log_path)
        self.utils = Utils(self.logger.log)

        self.sess = tf.Session()
        self.ckpt = None


    def load_data(self):
        # a list of source embedding dict {word:embedding}
        src_dict_list = [self.utils.load_emb(path) for path in self.input_list]
        src_dict_temp_1 = {}
        src_dict_temp_2 = {}
        src_dict_temp_3 = {}
        src_dict_list_temp = []
        for i in range(len(src_dict_list)):
            if i==0:
                for key, value in src_dict_list[i].items():
                    val_new = value[:-1]
                    src_dict_temp_1[key] = val_new

                pca_temp = np.empty((0, 100))
                for word,vec in src_dict_temp_1.items():
                    pca_temp = np.append(pca_temp, [vec], axis=0)
                pca = PCA(n_components=16)
                U = pca.fit_transform(pca_temp)
                dem_U_dict = {}
                x = 0
                for word,vec in src_dict_temp_1.items():
                    embed = U[x,:]
                    dem_U_dict[word] = embed
                    x+=1
                    
                src_dict_list_temp.append(dem_U_dict)
            elif i==1:
                for key, value in src_dict_list[i].items():
                    val_new = value[:-1]
                    src_dict_temp_2[key] = val_new

                pca_temp = np.empty((0, 100))
                for word,vec in src_dict_temp_2.items():
                    pca_temp = np.append(pca_temp, [vec], axis=0)
                pca = PCA(n_components=16)
                U = pca.fit_transform(pca_temp)

                lab_U_dict = {}
                x = 0
                for word,vec in src_dict_temp_2.items():
                    embed = U[x,:]
                    lab_U_dict[word] = embed
                    x+=1
                    
                src_dict_list_temp.append(lab_U_dict)


            elif i==2:
                for key, value in src_dict_list[i].items():
                    val_new = value[:-1]
                    src_dict_temp_3[key] = val_new

                pca_temp = np.empty((0, 100))
                for word,vec in src_dict_temp_3.items():
                    pca_temp = np.append(pca_temp, [vec], axis=0)
                pca = PCA(n_components=16)
                U = pca.fit_transform(pca_temp)

                notes_U_dict = {}
                x = 0
                for word,vec in src_dict_temp_3.items():
                    embed = U[x,:]
                    notes_U_dict[word] = embed
                    x+=1
                    
                src_dict_list_temp.append(notes_U_dict)

        src_dict_list = src_dict_list_temp.copy()

        with open('dem_emb_neig_dict.pickle', "rb") as input_file:
            dem_nei_dict = pickle.load(input_file)

        with open('lab_emb_neig_dict.pickle', "rb") as input_file:
            lab_nei_dict = pickle.load(input_file)

        with open('notes_emb_neig_dict.pickle', "rb") as input_file:
            notes_nei_dict = pickle.load(input_file)


        src_2_dem_dict = {}
        src_2_lab_dict = {}
        src_2_notes_dict = {}
        src_2_dict_list = []
        for i in range(len(src_dict_list)):
            if i == 0:
                for key, val in src_dict_list[i].items():
                    try:
                        dem_nei = dem_nei_dict[str(key)]
                        temp_list = []
                        for code in dem_nei:
                            embed = src_dict_list[i][code]
                            temp_list.append(embed)                        
                        a = np.array(temp_list)
                        a_mean = np.mean(a, axis=0)
                        src_2_dem_dict[key] = a_mean
                    except:
                        pass

                pca_temp = np.empty((0, 16))
                for word,vec in src_2_dem_dict.items():
                    pca_temp = np.append(pca_temp, [vec], axis=0)
                pca = PCA(n_components=16)
                U = pca.fit_transform(pca_temp)

                dem_nei_U_dict = {}
                x = 0
                for word,vec in src_2_dem_dict.items():
                    embed = U[x,:]
                    dem_nei_U_dict[word] = embed
                    x+=1

                src_2_dict_list.append(dem_nei_U_dict)

            if i == 1:
                for key, val in src_dict_list[i].items():
                    try:
                        lab_nei = lab_nei_dict[str(key)]
                        temp_list = []
                        for code in lab_nei:
                            embed = src_dict_list[i][code]
                            temp_list.append(embed)
                        a = np.array(temp_list)
                        a_mean = np.mean(a, axis=0)
                        src_2_lab_dict[key] = a_mean
                    except:
                        pass

                pca_temp = np.empty((0, 16))
                for word,vec in src_2_lab_dict.items():
                    pca_temp = np.append(pca_temp, [vec], axis=0)
                pca = PCA(n_components=16)
                U = pca.fit_transform(pca_temp)

                lab_nei_U_dict = {}
                x = 0
                for word,vec in src_2_lab_dict.items():
                    embed = U[x,:]
                    lab_nei_U_dict[word] = embed
                    x+=1

                src_2_dict_list.append(lab_nei_U_dict)


            if i == 2:
                for key, val in src_dict_list[i].items():
                    try:
                        notes_nei = notes_nei_dict[str(key)]
                        temp_list = []
                        for code in notes_nei:
                            embed = src_dict_list[i][code]
                            temp_list.append(embed)
                        a = np.array(temp_list)
                        a_mean = np.mean(a, axis=0)
                        src_2_notes_dict[key] = a_mean
                    except:
                        pass

                pca_temp = np.empty((0, 16))
                for word,vec in src_2_notes_dict.items():
                    pca_temp = np.append(pca_temp, [vec], axis=0)
                pca = PCA(n_components=16)
                U = pca.fit_transform(pca_temp)

                notes_nei_U_dict = {}
                x = 0
                for word,vec in src_2_notes_dict.items():
                    embed = U[x,:]
                    notes_nei_U_dict[word] = embed
                    x+=1

                src_2_dict_list.append(notes_nei_U_dict)

        src_nei_dem_list = []
        for i in range(len(src_dict_list)):
            if i == 0:
                src_nei_dem_list.append(src_dict_list[i])
        for i in range(len(src_2_dict_list)):
            if i == 0:
                src_nei_dem_list.append(src_2_dict_list[i])

        src_nei_lab_list = []
        for i in range(len(src_dict_list)):
            if i == 1:
                src_nei_lab_list.append(src_dict_list[i])
        for i in range(len(src_2_dict_list)):
            if i == 1:
                src_nei_lab_list.append(src_2_dict_list[i])

        src_nei_notes_list = []
        for i in range(len(src_dict_list)):
            if i == 2:
                src_nei_notes_list.append(src_dict_list[i])
        for i in range(len(src_2_dict_list)):
            if i == 2:
                src_nei_notes_list.append(src_2_dict_list[i])

        if self.oov:
            self.union_words = list(set.union(*[set(src_dict.keys()) for src_dict in src_dict_list]))
            self.logger.log('Union Words: %s' % len(self.union_words))
            source = []
            for i, src_dict in enumerate(src_dict_list):
                embed_mat = []
                for word in self.union_words:
                    embed = src_dict.get(word)
                    if embed is not None:
                        embed_mat.append(embed)
                    else:
                        embed_mat.append(np.zeros(self.dims[i]))
                source.append(skpre.normalize(embed_mat))


            for i, src_dict in enumerate(src_2_dict_list):
                embed_mat = []
                for word in self.union_words:
                    embed = src_dict.get(word)
                    if embed is not None:
                        embed_mat.append(embed)
                    else:
                        embed_mat.append(np.zeros(self.dims[i]))
                source.append(skpre.normalize(embed_mat))

        else:
            self.inter_words = list(set.intersection(*[set(src_dict.keys()) for src_dict in src_dict_list]))
            self.logger.log('Intersection Words: %s' % len(self.inter_words))
            src_tot_temp_lis = []
            for lis in src_nei_dem_list:
                src_tot_temp_lis.append(lis)
            for lis in src_nei_lab_list:
                src_tot_temp_lis.append(lis)
            for lis in src_nei_notes_list:
                src_tot_temp_lis.append(lis)
            self.sources = np.asarray(list(zip(*[skpre.normalize([src_dict[word] for word in self.inter_words]) for src_dict in src_tot_temp_lis])))
        print('self.sources.shape', self.sources.shape)
        
        del src_dict_list
        del src_2_dict_list


    def build_model(self):
        # initialize sources and inputs
        self.srcs = [tf.placeholder(tf.float32, (None, dim)) for dim in self.dims]
        self.ipts = [tf.placeholder(tf.float32, (None, dim)) for dim in self.dims]
        params = [self.dims, self.activ, self.factors]
        if self.model_type == 'M2M':
            self.ae = M2M(*params)
        self.ae.build(self.srcs,self.ipts)

    def train_model(self):
        """ Train the model.
            Variables with least losses will be stored in checkpoint file.
        """
        step = tf.Variable(0, trainable=False)
        rate = tf.train.exponential_decay(self.learning_rate, step, 50, 0.99)
        loss = self.ae.loss()
        opti = tf.train.AdamOptimizer(rate).minimize(loss, global_step=step)
        self.ckpt = tf.train.Saver(tf.global_variables())
        if self.restore:
            self.ckpt.restore(self.sess, self.ckpt_path)
        else:
            self.sess.run(tf.global_variables_initializer())
        size = len(self.sources) // self.batch_size # the number of batches
        best = float('inf')
        # loop for N epoches
        for itr in range(self.epoch):
            indexes = np.random.permutation(len(self.sources)) # shuffle training inputs
            train_loss = 0.
            # train with mini-batches
            for idx in range(size):
                batch_idx = indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
                batches = list(zip(*self.sources[batch_idx]))
                feed = {k:v for k, v in zip(self.srcs, batches)}
                feed.update({k:self._corrupt(v) for k, v in zip(self.ipts, batches)})
                _, batch_loss = self.sess.run([opti, loss], feed)
                train_loss += batch_loss
            epoch_loss = train_loss / size
            # save the checkpoint with least loss
            if epoch_loss <= best:
                self.ckpt.save(self.sess, self.ckpt_path)
                best = epoch_loss
            self.logger.log('[Epoch{0}] loss: {1}'.format(itr, epoch_loss))


    def generate_meta_embed(self):
        """ Generate meta-embedding and save as local file.
            Variables used to predict are these with least losses during training.
        """
        embed= {}
        self.logger.log('Generating meta embeddings...')
        self.ckpt.restore(self.sess, self.ckpt_path)
        if self.oov:
            vocabulary = self.union_words
        else:
            vocabulary = self.inter_words
        for i, word in enumerate(vocabulary):
            meta = self.sess.run(self.ae.extract(), {k:[v] for k, v in zip(self.ipts, self.sources[i])})
            embed[word] = np.reshape(meta, (np.shape(meta)[1],))
        self.sess.close()
        del self.sources

        self.utils.save_emb(embed, self.output_path)

    def _corrupt(self, batch):
        """ Corrupt a batch using masking noises.
            :param batch: the batch to be corrupted
            :return: a new batch after corrupting
        """
        noised = np.copy(batch)
        batch_size, feature_size = np.shape(batch)
        for i in range(batch_size):
            mask = np.random.randint(0, feature_size, int(feature_size * self.noise))
            for m in mask:
                noised[i][m] = 0.
        return noised

class AbsModel(object):
    """ Base class of all proposed methods.
        :param dims: a list of dimensionalities of each input
        :param activ: the string name of activation function
        :param factors: a list of coefficients of each loss part
    """

    __metaclass__ = ABCMeta

    def __init__(self, dims, activ, factors):
        self.dims = dims
        self.factors = factors

        if activ == 'lrelu':
            self.activ = tf.keras.layers.LeakyReLU(0.2)
        elif activ == 'prelu':
            self.activ = tf.keras.layers.PReLU()
        else:
            self.activ = tf.keras.layers.Activation(activ)

        self.meta = None

    @staticmethod
    def mse(x, y, f):
        """ Mean Squared Error with slicing.
            This method will slice vector with higher dimension to the lower one,
            if the two vector have different dimensions.
            :param x: first vector
            :param y: second vector
            :param f: coefficient
            :return: a tensor after calculating f * (1 / d) * ||x - y||^2
        """
        x_d = x.get_shape().as_list()[1]
        y_d = y.get_shape().as_list()[1]
        if x_d != y_d:
            smaller = min(x_d, y_d)
            x = tf.slice(x, [0, 0], [tf.shape(x)[0], smaller])
            y = tf.slice(y, [0, 0], [tf.shape(y)[0], smaller])
        return tf.scalar_mul(f, tf.reduce_mean(tf.squared_difference(x, y)))

    def extract(self):
        """ Extract the meta-embeddding model.
            :return: the meta-embedding model
        """
        return self.meta

    def my_dense(*args, **kwargs):   
      ##scope = tf.variable_scope(None, default_name='dense').__enter__()
      scope = tf.variable_scope(tf.get_variable_scope(), default_name='dense', reuse=True).__enter__()
      def f(input):
        r = tf.layers.dense(input, *args, name=scope, **kwargs)
        scope.reuse_variables()
        return r
      return f


    @abstractmethod
    def build(self, srcs, ipts):
        """ Abstract method.
            Build the model.
            :param srcs: source embeddings
            :param ipts: input embeddings
        """
        self.srcs = srcs
        self.ipts = ipts

        

    @abstractmethod
    def loss(self):
        """ Abstract method.
            Obtain the loss function of model.
            :return: a tensor calculating the loss function
        """
        pass

class M2M(AbsModel):
    """ Med2Meta Model
    """
    def build(self, srcs, ipts):
        AbsModel.build(self, srcs, ipts)

        inpts_1 = [self.ipts[0], self.ipts[1]]
        inpts_2 = [self.ipts[2], self.ipts[3]]
        inpts_3 = [self.ipts[4], self.ipts[5]]
        temp_dims = self.dims[0:2]
        self.encoders_1 = [tf.layers.dense(ipt, dim, self.activ) for ipt, dim in zip(inpts_1, temp_dims)]
        self.encoders_2 = [tf.layers.dense(ipt, dim, self.activ) for ipt, dim in zip(inpts_2, temp_dims)]
        self.encoders_3 = [tf.layers.dense(ipt, dim, self.activ) for ipt, dim in zip(inpts_3, temp_dims)]
        units = self.dims[0]*2
        self.meta_1 = tf.layers.dense(tf.nn.l2_normalize(tf.concat(self.encoders_1, 1), 1), units, self.activ)
        self.meta_2 = tf.layers.dense(tf.nn.l2_normalize(tf.concat(self.encoders_2, 1), 1), units, self.activ)
        self.meta_3 = tf.layers.dense(tf.nn.l2_normalize(tf.concat(self.encoders_3, 1), 1), units, self.activ)
        self.meta = tf.concat([self.meta_1, self.meta_2, self.meta_3], 1)
        temp_enc_tot = [self.meta]
        self.outs = [tf.layers.dense(temp_enc_tot[0], dim) for dim in self.dims]

    def loss(self):
        los = tf.add_n([self.mse(x, y, f) for x, y, f in zip(self.srcs, self.outs, self.factors[:-1])])
        for x, y in combinations(self.encoders_1, 2):
            los = tf.add(los, self.mse(x, y, self.factors[-1]))
        for x, y in combinations(self.encoders_2, 2):
            los = tf.add(los, self.mse(x, y, self.factors[-1]))
        for x, y in combinations(self.encoders_3, 2):
            los = tf.add(los, self.mse(x, y, self.factors[-1]))
        return los






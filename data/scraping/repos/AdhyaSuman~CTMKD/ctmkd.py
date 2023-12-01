import datetime
import os
from collections import defaultdict

import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from octis.models.contextualized_topic_models_KD.networks.decoding_network import DecoderNetwork
from octis.models.early_stopping.pytorchtools import EarlyStopping
from octis.evaluation_metrics.coherence_metrics import Coherence


class CTMKD(object):
    """Class to train the contextualized topic model
    """

    def __init__(self, input_size, bert_input_size, inference_type="zeroshot", num_topics=10, model_type='prodLDA',
                 hidden_sizes=None, activation='softplus', dropout=0.2, learn_priors=True, batch_size=64,
                 lr=2e-3, momentum=0.99, solver='adam', num_epochs=100, num_samples=10,
                 reduce_on_plateau=False, topic_prior_mean=0.0, topic_prior_variance=None, num_data_loader_workers=0, 
                 KD_loss_type='2wd'):
        """
        :param input_size: int, dimension of input
        :param bert_input_size: int, dimension of input that comes from BERT embeddings
        :param inference_type: string, you can choose between the contextual model and the combined model
        :param num_topics: int, number of topic components, (default 10)
        :param model_type: string, 'prodLDA' or 'LDA' (default 'prodLDA')
        :param hidden_sizes: None or tuple, length = n_layers, (default (100, 100))
        :param activation: string, 'softplus', 'relu', 'sigmoid', 'swish', 'tanh', 'leakyrelu', 'rrelu', 'elu',
         'selu' (default 'softplus')
        :param dropout: float, dropout to use (default 0.2)
        :param learn_priors: bool, make priors a learnable parameter (default True)
        :param batch_size: int, size of batch to use for training (default 64)
        :param lr: float, learning rate to use for training (default 2e-3)
        :param momentum: float, momentum to use for training (default 0.99)
        :param solver: string, optimizer 'adam' or 'sgd' (default 'adam')
        :param num_samples: int, number of times theta needs to be sampled
        :param num_epochs: int, number of epochs to train for, (default 100)
        :param reduce_on_plateau: bool, reduce learning rate by 10x on plateau of 10 epochs (default False)
        :param num_data_loader_workers: int, number of data loader workers (default cpu_count). set it to 0 if you are using Windows
        """

        assert isinstance(input_size, int) and input_size > 0, \
            "input_size must by type int > 0."
        assert (isinstance(num_topics, int) or isinstance(num_topics, np.int64)) and num_topics > 0, \
            "num_topics must by type int > 0."
        assert model_type in ['LDA', 'prodLDA'], \
            "model must be 'LDA' or 'prodLDA'."
        assert isinstance(hidden_sizes, tuple) or hidden_sizes is None, \
            "hidden_sizes must be type tuple or None."
        assert activation in ['softplus', 'relu', 'sigmoid', 'swish', 'tanh', 'leakyrelu',
                              'rrelu', 'elu', 'selu'], \
            "activation must be 'softplus', 'relu', 'sigmoid', 'swish', 'leakyrelu'," \
            " 'rrelu', 'elu', 'selu' or 'tanh'."
        assert dropout >= 0, "dropout must be >= 0."
        # assert isinstance(learn_priors, bool), "learn_priors must be boolean."
        assert isinstance(batch_size, int) and batch_size > 0, \
            "batch_size must be int > 0."
        assert lr > 0, "lr must be > 0."
        assert isinstance(momentum, float) and momentum > 0 and momentum <= 1, \
            "momentum must be 0 < float <= 1."
        assert solver in ['adagrad', 'adam', 'sgd', 'adadelta', 'rmsprop'], \
            "solver must be 'adam', 'adadelta', 'sgd', 'rmsprop' or 'adagrad'"
        assert isinstance(reduce_on_plateau, bool), \
            "reduce_on_plateau must be type bool."
        assert isinstance(topic_prior_mean, float), \
            "topic_prior_mean must be type float"
        # and topic_prior_variance >= 0, \
        # assert isinstance(topic_prior_variance, float), \
        #    "topic prior_variance must be type float"
        assert KD_loss_type in [None, '2wd', 'CE_only', '2wd_only'], "KD_loss_type must be '2wd', 'CE_only', '2wd_only'."

        self.input_size = input_size
        self.num_topics = num_topics
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.batch_size = batch_size
        self.lr = lr
        self.num_samples = num_samples
        self.bert_size = bert_input_size
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.reduce_on_plateau = reduce_on_plateau
        self.num_data_loader_workers = num_data_loader_workers
        self.topic_prior_mean = topic_prior_mean
        self.topic_prior_variance = topic_prior_variance
        # init inference avitm network
        self.model = DecoderNetwork(
            input_size, bert_input_size, inference_type, num_topics, model_type, hidden_sizes, activation,
            dropout, learn_priors)
        self.early_stopping = EarlyStopping(patience=5, verbose=False)
        # init optimizer
        if self.solver == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(self.momentum, 0.99))
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=self.momentum)
        elif self.solver == 'adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=lr)
        elif self.solver == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        elif self.solver == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr, momentum=self.momentum)
        # init lr scheduler
        if self.reduce_on_plateau:
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=10)

        # performance attributes
        self.best_loss_train = float('inf')

        # training attributes
        self.model_dir = None
        self.train_data = None
        self.nn_epoch = None

        # learned topics
        self.best_components = None

        # Use cuda if available
        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False
        if self.USE_CUDA:
            self.model = self.model.cuda()

        self.npmis = {}

        self.kd = False   
        self.teacher=None 
        self.temp = 1.0   
        self.alpha = 0.5  
        self.t_beta = None 
        self.KD_loss_type = KD_loss_type 
    
    def _get_kd_logits(self, X_bow, X_contextual_teacher):
       
        if self.USE_CUDA:
                X_contextual_teacher = X_contextual_teacher.cuda()
        
        theta_logits = self.teacher.get_theta_logits(X_bow, X_contextual_teacher)
        topic_pred = torch.softmax(theta_logits/self.temp, dim=1)
        
        return topic_pred

    
    def _get_kd_teacher_mean_logvar(self, X_bow, X_contextual_teacher):
       
        if self.USE_CUDA:
                X_contextual_teacher = X_contextual_teacher.cuda()
        
        mean, logvar = self.teacher.get_mean_logvar(X_bow, X_contextual_teacher)
        
        return (mean, logvar)
    

    def _get_kd_teacher_mean_logvar_recon(self, X_bow, X_contextual_teacher):
       
        if self.USE_CUDA:
                X_contextual_teacher = X_contextual_teacher.cuda()
        
        mean, logvar = self.teacher.get_mean_logvar(X_bow, X_contextual_teacher)
        
        t_theta = self.teacher.get_theta(X_bow, X_contextual_teacher)
        t_beta  = self.teacher.topic_word_matrix
        logits = torch.matmul(t_theta, t_beta)
        t_pred = torch.softmax(logits / self.temp, dim=1)
    
        return (mean, logvar, t_pred)

    
    def _get_kd_loss_mean_logvar(self, student_mean, student_logvar, pred):

        teacher_mean = pred[0]
        teacher_logvar = pred[1]
        mse_loss = torch.nn.MSELoss()
        kd_loss_mean  = mse_loss(student_mean, teacher_mean)
        kd_loss_logvar = mse_loss(student_logvar, teacher_logvar)
        kd_loss = (kd_loss_mean + kd_loss_logvar)
        return kd_loss
    

    def _get_kd_loss_mean_logvar_recon(self, student_mean, student_var, student_logvar, student_recon, pred):

        teacher_mean = pred[0]
        teacher_logvar = pred[1]
        teacher_var = torch.exp(pred[1])
        teacher_recon = pred[2]

        
        if self.KD_loss_type=='2wd':
            kd_loss1 = CTMKD.get_2_wasserstein_dist(student_mean, student_var, teacher_mean, teacher_var)

        elif self.KD_loss_type=='CE_only':
            kd_loss1 = 0.0
        
        elif self.KD_loss_type=='2wd_only':
            return CTMKD.get_2_wasserstein_dist(student_mean, student_var, teacher_mean, teacher_var)

        else:
            raise Exception('Loss type is undefined.')
        
        log_output = torch.log(student_recon + 1e-10)
        kd_loss2 = -torch.sum(teacher_recon * log_output, dim=1)
        
        kd_loss = kd_loss1 + self.temp * self.temp * kd_loss2
       
        return kd_loss

    
    def _get_CE_loss(self, t_predictions, s_predictions):

        log_output = torch.log(s_predictions + 1e-10)
        kd_loss = -torch.sum(t_predictions * log_output, dim=1)
        return self.temp * self.temp * kd_loss

    @staticmethod
    def get_kldiv(prior_mean, prior_variance,
                posterior_mean, posterior_variance, posterior_log_variance):
        
        dimension = prior_mean.shape[0]
        # var division term
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)
        # logvar det division term
        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)
        # combine terms
        return 0.5 * (var_division + diff_term - dimension + logvar_det_division)

    @staticmethod
    def get_KD_kld(mean_1, var_1, mean_2, var_2, log_var_2):
        
        dimension = mean_1.shape[1]
        # var division term
        var_division = torch.sum((var_2 / var_1), dim=-1)
        # diff means term
        diff_means = mean_1 - mean_2
        diff_term = torch.sum(
            (diff_means * diff_means) / var_1, dim=1)
        # logvar det division term
        logvar_det_division = var_1.log().sum() - log_var_2.sum(dim=1)
        # combine terms
        kld = 0.5 * (var_division + diff_term - dimension + logvar_det_division)
        
        return kld
    
    @staticmethod
    def get_2_wasserstein_dist(mean_1, var_1, mean_2, var_2):
        '''
        mean_1: shape = (Batch size, K)
        var_1: shape = (Batch size, K): diagonal cov matrix
        mean_2: shape = (Batch size, K)
        var_2: shape = (Batch size, K): diagonal cov matrix
        '''
        diff_mean_l2_sq = torch.sum((mean_1 - mean_2).square(), dim=-1) #||mean_1 - mean_2||_2 ^2
        tr_1 = var_1.sum(dim=-1)  #Tr(\Sigma_1)
        tr_2 = var_2.sum(dim=-1) #Tr(\Sigma_2)
        tr_sqrt_12 = torch.sqrt(var_1 * var_2).sum(dim=-1) #Tr(\sqrt{\Sigma_1 \Sigma_2})
        wd = diff_mean_l2_sq + tr_1 + tr_2 - 2*tr_sqrt_12
        return wd

    def _loss(self, inputs, word_dists, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance,
              recon=None, epoch_num=0, theta=None, t_predictions=None):
        
        KL = CTMKD.get_kldiv(prior_mean, prior_variance, posterior_mean, posterior_variance, posterior_log_variance)
        # Reconstruction term
        RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)
        loss = KL + RL

        if (self.kd  and epoch_num < self.KD_epochs):
            kd_loss = 0.
            
            if (self.use_topic_vector_kd):
               assert theta!=None, "theta is None. Must must be a valid vector"
               kd_loss = self._get_CE_loss(t_predictions, theta)
               loss = (self.alpha * self.temp * self.temp) * kd_loss + (1-self.alpha)*loss
            elif (self.use_mean_logvar_kd):
                    kd_loss = self._get_kd_loss_mean_logvar(posterior_mean, posterior_variance, 
                                       t_predictions)
                    loss = self.alpha * kd_loss + loss
           
            elif (self.use_mean_logvar_recon_kd):
                    kd_loss = self._get_kd_loss_mean_logvar_recon(posterior_mean, 
                                                                posterior_variance,
                                                                posterior_log_variance,
                                                                recon,
                                                                t_predictions)
                    loss = (1-self.alpha) * loss + self.alpha * kd_loss
            else:
                pass

        return loss.sum()

    def _train_epoch(self, loader, epoch_num):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0
        topic_doc_list = []
        kd_predictions=None
        KD_loss = 0

        for batch_samples in loader:
            # batch_size x vocab_size
            X_bow = batch_samples['X']
            X_bow = X_bow.reshape(X_bow.shape[0], -1)
            X_contextual = batch_samples['X_bert']
            if self.USE_CUDA:
                X_bow = X_bow.cuda()
                X_contextual = X_contextual.cuda()

            # forward pass
            self.model.zero_grad()
            prior_mean, prior_variance, posterior_mean, posterior_variance, \
               posterior_log_variance, word_dists, theta, recon =\
                self.model(X_bow, X_contextual, self.temp)
            topic_doc_list.extend(theta)


            if (self.kd and epoch_num < self.KD_epochs):  #knowledge distillation
               X_contextual_teacher = batch_samples['X_contextual_teacher']
               if (self.use_topic_vector_kd):
                  kd_predictions = self._get_kd_logits(X_bow, X_contextual_teacher)
               elif (self.use_mean_logvar_kd):
                       kd_predictions = self._get_kd_teacher_mean_logvar(X_bow, X_contextual_teacher)
               elif (self.use_mean_logvar_recon_kd):
                       kd_predictions = self._get_kd_teacher_mean_logvar_recon(X_bow, X_contextual_teacher)

            # backward pass
            loss = self._loss(
                X_bow, word_dists, prior_mean, prior_variance,
                posterior_mean, posterior_variance, posterior_log_variance, 
                recon,
                epoch_num,
                theta, kd_predictions)
            
            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += X_bow.size()[0]
            train_loss += loss.item()

        train_loss /= samples_processed
        # KD_loss /= samples_processed

        return samples_processed, train_loss, topic_doc_list

    def _validation(self, loader):
        """Train epoch."""
        self.model.eval()
        val_loss = 0
        samples_processed = 0
        for batch_samples in loader:
            # batch_size x vocab_size
            X = batch_samples['X']
            X = X.reshape(X.shape[0], -1)
            X_bert = batch_samples['X_bert']

            if self.USE_CUDA:
                X = X.cuda()
                X_bert = X_bert.cuda()

            # forward pass
            self.model.zero_grad()
            prior_mean, prior_variance, \
            posterior_mean, posterior_variance, posterior_log_variance, \
            word_dists, topic_word, topic_document = self.model(X, X_bert)

            loss = self._loss(X, word_dists, prior_mean, prior_variance,
                              posterior_mean, posterior_variance, posterior_log_variance)

            # compute train loss
            samples_processed += X.size()[0]
            val_loss += loss.item()

        val_loss /= samples_processed

        return samples_processed, val_loss

    def fit(self, train_dataset, validation_dataset=None, save_dir=None, verbose=True, KD_verbose=True, patience=5, delta=0, 
            teacher=None, alpha=0, temp=1.,
            use_topic_vector_kd=False, use_mean_logvar_kd=False, use_mean_logvar_recon_kd=False,
            KD_epochs=100,
            texts=None, gap=None):
        """
        Train the CTMKD model.

        :param train_dataset: PyTorch Dataset class for training data.
        :param validation_dataset: PyTorch Dataset class for validation data
        :param save_dir: directory to save checkpoint models to.
        :param verbose: verbose
        """
        # Print settings to output file
        if verbose:
            print("Settings: \n\
                   N Components: {}\n\
                   Topic Prior Mean: {}\n\
                   Topic Prior Variance: {}\n\
                   Model Type: {}\n\
                   Hidden Sizes: {}\n\
                   Activation: {}\n\
                   Dropout: {}\n\
                   Learn Priors: {}\n\
                   Learning Rate: {}\n\
                   Momentum: {}\n\
                   Reduce On Plateau: {}\n\
                   Save Dir: {}".format(
                self.num_topics, self.topic_prior_mean,
                self.topic_prior_variance, self.model_type,
                self.hidden_sizes, self.activation, self.dropout, self.learn_priors,
                self.lr, self.momentum, self.reduce_on_plateau, save_dir))
        
        if teacher:
            self.kd = True
            self.teacher=teacher
            if self.USE_CUDA:
                self.teacher = self.teacher.model.cuda()
            self.alpha = alpha
            self.temp = temp
            self.t_beta  = self.teacher.topic_word_matrix
            self.use_mean_logvar_kd = use_mean_logvar_kd
            self.use_topic_vector_kd = use_topic_vector_kd
            self.use_mean_logvar_recon_kd = use_mean_logvar_recon_kd
            self.KD_epochs = KD_epochs
            self.teacher.eval()

            if KD_verbose:
                print("KD_Settings: \n\
                    Alpha: {}\n\
                    Temp: {}\n\
                    use_mean_logvar_kd: {}\n\
                    use_topic_vector_kd: {}\n\
                    use_mean_logvar_recon_kd: {}\n\
                    KD_epochs: {}\n\
                    KD_loss_type: {}".format(
                    self.alpha, self.temp,self.use_mean_logvar_kd,
                    self.use_topic_vector_kd,self.use_mean_logvar_recon_kd,
                    self.KD_epochs, self.KD_loss_type))



        self.model_dir = save_dir
        self.train_data = train_dataset
        self.validation_data = validation_dataset

        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_data_loader_workers)

        # init training variables
        train_loss = 0
        samples_processed = 0
        self.KD_loss = []

        npmi = Coherence(texts=texts, measure='c_npmi')

        # train loop
        for epoch in range(self.num_epochs):
            self.nn_epoch = epoch
            # train epoch
            s = datetime.datetime.now()
            sp, train_loss, topic_document = self._train_epoch(train_loader, epoch)
            samples_processed += sp
            e = datetime.datetime.now()

            # self.KD_loss.append(kd_loss)

            if verbose:
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                    epoch + 1, self.num_epochs, samples_processed,
                    len(self.train_data) * self.num_epochs, train_loss, e - s))

            self.best_components = self.model.beta
            # self.final_topic_word = topic_word
            self.final_topic_document = topic_document
            self.best_loss_train = train_loss
            if self.validation_data is not None:
                validation_loader = DataLoader(
                    self.validation_data, batch_size=self.batch_size, shuffle=True,
                    num_workers=self.num_data_loader_workers)
                # train epoch
                s = datetime.datetime.now()
                val_samples_processed, val_loss = self._validation(validation_loader)
                e = datetime.datetime.now()

                if verbose:
                    print("Epoch: [{}/{}]\tSamples: [{}/{}]\tValidation Loss: {}\tTime: {}".format(
                        epoch + 1, self.num_epochs, val_samples_processed,
                        len(self.validation_data) * self.num_epochs, val_loss, e - s))

                if np.isnan(val_loss) or np.isnan(train_loss):
                    break
                else:
                    self.early_stopping(val_loss, self.model)
                    if self.early_stopping.early_stop:
                        if verbose:
                            print("Early stopping")
                        if save_dir is not None:
                            self.save(save_dir)
                        break
            #report NPMI
            gap = 200#2 if epoch<10 else 10
            if texts and ((epoch + 1)%gap==0):
                npmi_score = npmi.score(self.get_info())
                self.npmis[epoch + 1] = npmi_score
                if verbose:
                    print("Epoch: [{}/{}]\tNPMI: {}".format(epoch + 1, self.num_epochs, npmi_score))

        if save_dir is not None:
            self.save(save_dir)

    def predict(self, dataset):
        """Predict input."""
        self.model.eval()

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False,
                            num_workers=self.num_data_loader_workers)

        topic_document_mat = []
        with torch.no_grad():
            for batch_samples in loader:
                # batch_size x vocab_size
                X = batch_samples['X']
                X = X.reshape(X.shape[0], -1)
                X_bert = batch_samples['X_bert']

                if self.USE_CUDA:
                    X = X.cuda()
                    X_bert = X_bert.cuda()
                # forward pass
                self.model.zero_grad()
                _, _, _, _, _, _, _, topic_document = self.model(X, X_bert)
                topic_document_mat.append(topic_document)

        results = self.get_info()
        results['test-topic-document-matrix'] = np.asarray(self.get_thetas(dataset)).T

        return results

    def get_topic_word_mat(self):
        top_wor = self.model.topic_word_matrix.cpu().detach().numpy()
        return top_wor

    def get_topic_document_mat(self):
        top_doc = self.final_topic_document
        top_doc_arr = np.array([i.cpu().detach().numpy() for i in top_doc])
        return top_doc_arr

    def get_topics(self, k=10):
        """
        Retrieve topic words.

        Args
            k : (int) number of words to return per topic, default 10.
        """
        assert k <= self.input_size, "k must be <= input size."
        component_dists = self.best_components
        topics = defaultdict(list)
        topics_list = []
        if self.num_topics is not None:
            for i in range(self.num_topics):
                _, idxs = torch.topk(component_dists[i], k)
                component_words = [self.train_data.idx2token[idx]
                                   for idx in idxs.cpu().numpy()]
                topics[i] = component_words
                topics_list.append(component_words)

        return topics_list

    def get_info(self):
        info = {}
        topic_word = self.get_topics()
        topic_word_dist = self.get_topic_word_mat()
        topic_document_dist = self.get_topic_document_mat()
        info['topics'] = topic_word

        info['topic-document-matrix'] = np.asarray(self.get_thetas(self.train_data)).T

        info['topic-word-matrix'] = topic_word_dist
        return info
    
    def get_npmis(self):
        return self.npmis
    
    def get_KDloss(self):
        return self.KD_loss

    def _format_file(self):
        model_dir = "AVITM_nc_{}_tpm_{}_tpv_{}_hs_{}_ac_{}_do_{}_lr_{}_mo_{}_rp_{}". \
            format(self.num_topics, 0.0, 1 - (1. / self.num_topics),
                   self.model_type, self.hidden_sizes, self.activation,
                   self.dropout, self.lr, self.momentum,
                   self.reduce_on_plateau)
        return model_dir

    def save(self, models_dir=None):
        """
        Save model.

        :param models_dir: path to directory for saving NN models.
        """
        if (self.model is not None) and (models_dir is not None):

            model_dir = self._format_file()
            if not os.path.isdir(os.path.join(models_dir, model_dir)):
                os.makedirs(os.path.join(models_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(models_dir, model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'dcue_dict': self.__dict__}, file)

    def load(self, model_dir, epoch):
        """
        Load a previously trained model.

        :param model_dir: directory where models are saved.
        :param epoch: epoch of model to load.
        """
        epoch_file = "epoch_" + str(epoch) + ".pth"
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, 'rb') as model_dict:
            checkpoint = torch.load(model_dict)

        for (k, v) in checkpoint['dcue_dict'].items():
            setattr(self, k, v)

        self.model.load_state_dict(checkpoint['state_dict'])

    def get_thetas(self, dataset):
        """
        Get the document-topic distribution for a dataset of topics. Includes multiple sampling to reduce variation via
        the parameter num_samples.
        :param dataset: a PyTorch Dataset containing the documents
        """
        self.model.eval()

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_data_loader_workers)
        final_thetas = []
        for sample_index in range(self.num_samples):
            with torch.no_grad():
                collect_theta = []
                for batch_samples in loader:
                    # batch_size x vocab_size
                    x = batch_samples['X']
                    x = x.reshape(x.shape[0], -1)
                    x_bert = batch_samples['X_bert']
                    if self.USE_CUDA:
                        x = x.cuda()
                        x_bert = x_bert.cuda()
                    # forward pass
                    self.model.zero_grad()
                    collect_theta.extend(self.model.get_theta(x, x_bert).cpu().numpy().tolist())

                final_thetas.append(np.array(collect_theta))
        return np.sum(final_thetas, axis=0) / self.num_samples

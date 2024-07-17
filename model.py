import os
import numpy as np
import itertools
import pickle

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.optim import lr_scheduler

from data_reader import BatchDataGenerator
from utils import write_log
from mlp import MLVAE

from merlion.models.anomaly.vae import VAEConfig
from metrics.eval_metrics import *

from datetime import datetime


class ModelBaseline:
    def __init__(self, flags):
        """
        Initialize the model
        """
        # Set the default tensor type
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # Get the data parameters
        # with open(flags.data_root + flags.data_info, 'rb') as f:
        #     self.data_info = pickle.load(f)
        # self.feat_num = self.data_info['feat_num']


        self.feat_num = flags.feat_num


        # Initialize the model
        # kwargs = {'encoder_hidden_sizes': (25, 10, 5), 'decoder_hidden_sizes': (5, 10, 25)}
        kwargs = {'encoder_hidden_sizes': (40, 30, 20, 10, 5), 'decoder_hidden_sizes': (5, 10, 20, 30, 40)}
        config, _ = VAEConfig().from_dict(kwargs, return_unused_kwargs=True) ##### Usage ?? 
        self.MLAD = MLVAE(config)
        self.n_past = flags.sequence_len
        self.network = self.MLAD._build_model(self.feat_num*self.n_past)

        # Set the optimizer, scheduler, and loss function
        self.configure(flags)
        
        # Load pre-trained model
        self.load_state_dict(flags.state_dict)

        # Set paths for training, validation, and unseen(test) domains
        self.setup_path(flags)
        
        # Create paths to save log files
        if not os.path.exists(flags.logs):
            os.mkdir(flags.logs)
        # Write arguments to a log file
        flags_log = os.path.join(flags.logs, 'flags_log.txt')
        write_log(flags, flags_log)


    def setup_path(self, flags):
        """
        Set paths for training, validation, and unseen(test) domains
        """
        root_folder = flags.data_root

        # # Paths for training domains
        # self.train_paths = []
        # for data in self.data_info['train_data']:
        #     path = os.path.join(root_folder, data)
        #     self.train_paths.append(path)
        
        # # Paths for validation domains
        # self.val_paths = []
        # for data in self.data_info['val_data']:
        #     path = os.path.join(root_folder, data)
        #     self.val_paths.append(path)

        # # Take the unseen domain for testing
        # unseen_index = flags.unseen_index
        # self.unseen_data_path = os.path.join(root_folder, self.data_info['test_data'][unseen_index])
        # # Remove the unseen domain from training and validation
        # self.train_paths.remove(self.train_paths[unseen_index]) 
        # self.val_paths.remove(self.val_paths[unseen_index])

        # # Log the paths to file
        # if not os.path.exists(flags.logs):
        #     os.mkdir(flags.logs)
        # flags_log = os.path.join(flags.logs, 'path_log.txt')
        # write_log(str(self.train_paths), flags_log)
        # write_log(str(self.val_paths), flags_log)
        # write_log(str(self.unseen_data_path), flags_log)


        tr_data, val_data = [], []
        for f in flags.train_data: 
            data_path = flags.data_root + '_'.join([f, flags.test_data])
            tmp_data = np.load(data_path + '/' + flags.dataset + '_train.npy') 
            print(np.shape(tmp_data))

            val_num = int(len(tmp_data)*flags.val_ratio)
            tr_data.append(tmp_data[:-val_num])
            val_data.append(tmp_data[-val_num:])

        print([np.shape(i) for i in tr_data])
        print([np.shape(i) for i in val_data])


        te_data = np.load(data_path + '/' + flags.dataset + '_test.npy') 
        te_label = np.load(data_path + '/' + flags.dataset + '_test_label.npy') 


        # Initialize the batch generator for training and validation
        self.batDataGenTrains = []
        for data in tr_data:
            batDataGenTrain = BatchDataGenerator(flags=flags, stage='train', data=data, n_past=self.n_past)
            self.batDataGenTrains.append(batDataGenTrain)

        self.batDataGenVals = []
        for data in val_data: #val_path in self.val_paths:
            batDataGenVal = BatchDataGenerator(flags=flags, stage='val', data=data, n_past=self.n_past)
            batDataGenVal.get_batches()
            self.batDataGenVals.append(batDataGenVal)

        # Initialize the batch generator for testing and get the testing labels
        self.batDataGenTest = BatchDataGenerator(flags=flags, stage='test', data=te_data, labels=te_label, n_past=self.n_past)
        self.batDataGenTest.get_batches()
        self.test_labels = list(itertools.chain.from_iterable([list(np.argmax(test_label, axis=1)) 
                                                               for test_label in self.batDataGenTest.batch_labels]))










    def configure(self, flags, display_model=False):
        """
        Config the model optimizer, losss function, and scheduler
        """
        if display_model: 
            for name, para in self.network.named_parameters(): 
                print(name, para.size())
        
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=flags.lr)
        self.loss_fn = nn.MSELoss()
        self.scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=flags.step_size, gamma=0.1)

    
    def load_state_dict(self, state_dict=''):
        """
        Load the pre-trained model
        """
        if state_dict:
            # Load the model parameters
            try:
                tmp = torch.load(state_dict)
                pretrained_dict = tmp['state']
            except:
                pretrained_dict = model_zoo.load_url(state_dict)

            model_dict = self.network.state_dict()
            # 1. Filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and v.size() == model_dict[k].size()}
            # 2. Overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. Load the new state dict
            self.network.load_state_dict(model_dict)

    
    def heldout_test(self, flags): 
        """
        Testing the model with the heldout domain
        """
        # Load the best model in the validation data
        model_path = os.path.join(flags.model_path, 'best_model.tar')
        self.load_state_dict(state_dict=model_path)
        
        # Switch on the network test mode
        te_start = datetime.now() ### REVISED HERE ### 
        self.network.eval()

        # Get the anomaly scores for training data
        train_scores = []
        for batDataGenTrain in self.batDataGenTrains: 
            for train_batch in batDataGenTrain.batch_data: 
                train_scores.extend(list(self.MLAD.get_anomaly_score(self.network, train_batch)))

        # Get the anomaly scores for testing data
        test_scores = []
        for test_data in self.batDataGenTest.batch_data:
            test_scores.extend(list(self.MLAD.get_anomaly_score(self.network, test_data)))   

        te_end = datetime.now()
        te_time = round((te_end - te_start).total_seconds()/60, 4)
        print('Testing time (min): ', te_time)

        # Evaluate the AD model
        results = overall_evaluation(train_scores, test_scores, np.array(self.test_labels)) 

        return results, te_time

        # Log the AUC
        # auc = results['pak_results']['auc_wo_pa']
        # print('----------auc test----------:', auc)
        # flags_log = os.path.join(flags.logs, 'heldout_test_log.txt')
        # write_log(auc, flags_log)


    def test(self, batDataGenVal, ite, log_prefix, log_dir='logs/'):
        """
        Testing the model following the workflow
        """
        # Switch on the network test model 
        self.network.eval()

        # Get the loss for validation data
        val_losses = []
        for val_batch in batDataGenVal.batch_data: 
            val_losses.append(self.MLAD.get_batch_loss(self.network, val_batch, self.loss_fn).item()/len(val_batch))
        val_loss = np.mean(val_losses)
        print('----------val loss----------:', val_loss)

        # Log the loss to file 
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_path = os.path.join(log_dir, '{}.txt'.format(log_prefix))
        write_log(str('ite:{}, val_loss:{}'.format(ite, val_loss)), log_path=log_path)

        # Switch back the network train mode after test
        self.network.train() 
        return val_loss
    

    def test_workflow(self, batDataGenVals, flags, ite):
        """
        Evaluate the model periodically with the validation domains
        """
        # Get the loss for each validation domain
        losses = []
        for count, batDataGenVal in enumerate(batDataGenVals):
            loss_val = self.test(batDataGenVal, ite, log_prefix='val_index_{}'.format(count), log_dir=flags.logs) 
            losses.append(loss_val)
        mean_loss = np.mean(losses)

        # Update the best AUC
        if mean_loss < self.best_loss_val:
            self.best_loss_val = mean_loss
            # Log the best loss to file
            f = open(os.path.join(flags.logs, 'Best_val.txt'), mode='a')
            f.write('ite:{}, best val loss:{}\n'.format(ite, self.best_loss_val))
            f.close()
            # Save the model with the best AUC
            if not os.path.exists(flags.model_path):
                os.mkdir(flags.model_path)
            outfile = os.path.join(flags.model_path, 'best_model.tar')
            torch.save({'ite': ite, 'state': self.network.state_dict()}, outfile)


class ModelMLDG(ModelBaseline):
    """
    Model for Meta-Learning Domain Generalization (MLDG)
    """
    def __init__(self, flags):
        """
        Initialize the model by ModelBaseline
        """
        ModelBaseline.__init__(self, flags)


    def train(self, flags):
        # Set the network to train mode
        self.network.train()
        # Initialze the best validation loss
        self.best_loss_val = np.inf

        tr_start = datetime.now() ### REVISED HERE ### 

        for ite in range(flags.inner_loops):
            # Update the learning rate
            # self.scheduler.step()

            # Randomly select a validation domain for meta val
            index_val = np.random.choice(a=np.arange(0, len(self.batDataGenTrains)), size=1)[0]
            batDataMetaVal = self.batDataGenTrains[index_val]

            # Initialize the loss the meta training
            meta_train_loss = 0.0
            for index in range(len(self.batDataGenTrains)):
                if index == index_val:
                    continue
                
                # Update the batches when ite > training batch_num*N
                batDataGenTrain = self.batDataGenTrains[index]
                if ite % batDataGenTrain.batch_num == 0: 
                    batDataGenTrain.get_batches(shuffle=(ite!=0), seed=ite)

                # Get a batch for training 
                batch_idx = ite % batDataGenTrain.batch_num
                inputs_train = batDataGenTrain.batch_data[batch_idx]

                # Get the training loss
                loss = self.MLAD.get_batch_loss(self.network, inputs_train, self.loss_fn) 
                meta_train_loss += loss

            # Update the batches when ite > validation batch_num*N
            if ite % batDataMetaVal.batch_num == 0: 
                    batDataMetaVal.get_batches(shuffle=(ite!=0), seed=ite)

            # Get a batch for validation
            val_batch_idx = ite % batDataMetaVal.batch_num
            inputs_val = batDataMetaVal.batch_data[val_batch_idx]

            # Update the model with one-step gradient
            para_orig, para_updated = [], []
            for para in self.network.parameters():
                grad = torch.autograd.grad(meta_train_loss, para, create_graph=True)[0]
                para_orig.append(para.data)
                para_updated.append(para.data - grad * flags.meta_step_size)

            # Update the model with the updated parameters
            para_count = 0 
            for para in self.network.parameters():
                para.data = para_updated[para_count]
                para_count += 1

            # Get the validation loss
            meta_val_loss = self.MLAD.get_batch_loss(self.network, inputs_val, self.loss_fn) 

            # Reset the model by original parameters
            para_count = 0 
            for para in self.network.parameters():
                para.data = para_orig[para_count]
                para_count += 1
            
            # Get the total loss
            total_loss = meta_train_loss + meta_val_loss * flags.meta_val_beta

            # Init the grad to zeros first
            self.optimizer.zero_grad()

            # Backward the network
            total_loss.backward()

            # Optimize the parameters
            self.optimizer.step()

            # Update the learning rate
            self.scheduler.step()

            print(
                'ite:', ite,
                'meta_train_loss:', meta_train_loss.cpu().data.numpy(),
                'meta_val_loss:', meta_val_loss.cpu().data.numpy(),
                'lr:',
                self.scheduler.get_last_lr()[0])
            
            # Log the training and validation losses
            flags_log = os.path.join(flags.logs, 'loss_log.txt')
            write_log(str(meta_train_loss.cpu().data.numpy()) + '\t' + str(meta_val_loss.cpu().data.numpy()), flags_log)
            del total_loss
            
            # Evaluate the model periodically
            if ite % flags.test_every == 0 and ite != 0 or flags.debug:
                self.test_workflow(self.batDataGenVals, flags, ite)

        tr_end = datetime.now()
        tr_time = round((tr_end - tr_start).total_seconds()/60, 4)
        print('Training time (min): ', tr_time)

        return tr_time
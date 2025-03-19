from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_MEWS
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
from exp.multi_timeline_2 import categorical_collate

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, CEL

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

from functools import partial

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        print(f"[{time.strftime('%H:%M:%S')}] Initializing Informer experiment...")
        super(Exp_Informer, self).__init__(args)
        self.exp_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    
    def _build_model(self):
        print(f"[{time.strftime('%H:%M:%S')}] Building model architecture...")
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            print(f"[{time.strftime('%H:%M:%S')}] Creating {self.args.model} with {e_layers} encoder layers...")
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in, 
                self.args.c_out, 
                self.args.seq_len, 
                self.args.label_len,
                self.args.pred_len, 
                self.args.factor,
                self.args.d_model, 
                self.args.n_heads, 
                e_layers, # self.args.e_layers,
                self.args.d_layers, 
                self.args.d_ff,
                self.args.dropout, 
                self.args.attn,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.output_attention,
                self.args.distil,
                self.args.mix,
                # --- Added Conditionning ---
                self.args.condition,
                self.args.n_cond_num_in,
                self.args.n_cond_num_out,
                self.args.n_cond_cat_in,
                self.args.n_cond_cat_out,
                # ---
                self.device
            ).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            print(f"[{time.strftime('%H:%M:%S')}] Setting up model for multi-GPU...")
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        print(f"[{time.strftime('%H:%M:%S')}] Loading {flag} dataset...")
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'MEWS':Dataset_MEWS,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            # Data = Dataset_Pred # DEBUG: Disable for now
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        
        print(f"[{time.strftime('%H:%M:%S')}] Creating dataset with parameters:")
        print(f"    - Root path: {args.root_path}")
        print(f"    - Data path: {args.data_path}")
        print(f"    - Sequence length: {args.seq_len}")
        print(f"    - Label length: {args.label_len}")
        print(f"    - Prediction length: {args.pred_len}")
        
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(f"[{time.strftime('%H:%M:%S')}] {flag} dataset size: {len(data_set)}")
        
        print(f"[{time.strftime('%H:%M:%S')}] Creating DataLoader...")
        print(f"    - Batch size: {batch_size}")
        print(f"    - Shuffle: {shuffle_flag}")
        print(f"    - Num workers: {args.num_workers}")
        print(f"    - Drop last: {drop_last}")
        
        categorical_collate_fn = partial(categorical_collate, timeenc=data_set.timeenc, freq=data_set.freq)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn = categorical_collate_fn)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    # def _select_criterion(self):
    #     if self.args.loss == 'mse':
    #         criterion = nn.MSELoss()
    #     elif self.args.loss == 'l1':
    #         criterion = nn.L1Loss()
    #     elif self.args.loss == 'crossentropy':
    #         criterion = nn.CrossEntropyLoss()
    #     return criterion

    def select_criterion(self, criterion_name):
        if criterion_name == 'mse':
            criterion = nn.MSELoss()
        elif criterion_name == 'l1':
            criterion = nn.L1Loss()
        elif criterion_name == 'crossentropy':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        return criterion
    
    def compute_loss(self, continuous_preds, continuous_targets, category_logits, category_targets, alpha=1.0):
        # Regression loss (MSE)
        loss_mse = self.criterion(continuous_preds, continuous_targets)

        # Classification loss (Cross Entropy)
        loss_ce = self.criterion_category(category_logits, category_targets)

        # Total loss (weighted sum)
        loss = loss_ce + alpha * loss_mse

        # return loss
        return loss, {'loss_ce': loss_ce.item(), 'loss_mse': loss_mse.item()}

    # def vali(self, vali_data, vali_loader, criterion):
    def vali(self, vali_data, vali_loader):
        self.model.eval()
        total_loss = []
        total_loss_dict = {"loss_mse": [], "loss_ce": []}
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_x_id,batch_y_id,batch_static,batch_antibio) in enumerate(vali_loader):
            pred, true_y, true_antibio = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_static, batch_antibio)
            # loss = criterion(pred.detach().cpu(), true.detach().cpu())
            loss, loss_dict = self.compute_loss(pred[:,:,:-1].detach().cpu(), true_y.detach().cpu(), pred[:,:,-1].detach().cpu(), true_antibio.detach().cpu(), alpha=self.args.loss_alpha)
            total_loss.append(loss)
            total_loss_dict["loss_mse"].append(loss_dict["loss_mse"])
            total_loss_dict["loss_ce"].append(loss_dict["loss_ce"])
        total_loss = np.average(total_loss)
        total_loss_dict["loss_mse"] = np.average(total_loss_dict["loss_mse"])
        total_loss_dict["loss_ce"] = np.average(total_loss_dict["loss_ce"])
        self.model.train()
        return total_loss, total_loss_dict

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        self.criterion = self.select_criterion(self.args.loss)
        self.criterion_category = self.select_criterion(self.args.loss_category)

        path = self.args.root_path + self.args.checkpoints_path + '/' + setting
        # path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        # criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_x_id,batch_y_id,batch_static,batch_antibio) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true_y, true_antibio = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_static, batch_antibio)
                loss, loss_dict = self.compute_loss(pred[:,:,:-1], true_y, pred[:,:,-1], true_antibio, alpha=self.args.loss_alpha)
                train_loss.append(loss.item())
                
                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    print("\tloss_mse: {0:.7f} | loss_ce {1:.7f}".format(loss_dict['loss_mse'], loss_dict['loss_ce']))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_data, vali_loader, criterion)
            # test_loss = self.vali(test_data, test_loader, criterion)
            vali_loss, vali_loss_dict = self.vali(vali_data, vali_loader)
            test_loss, test_loss_dict = self.vali(test_data, test_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Vali Loss: loss_mse: {0:.7f} | loss_ce: {1:.7f}".format(vali_loss_dict['loss_mse'], vali_loss_dict['loss_ce']))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.args)
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        self.criterion = self.select_criterion(self.args.loss)
        self.criterion_category = self.select_criterion(self.args.loss_category)

        self.model.eval()
        
        preds_y = []
        trues_y = []
        preds_antibio = []
        trues_antibio = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_x_id,batch_y_id,batch_static,batch_antibio) in enumerate(test_loader):
            pred, true_y, true_antibio = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_static, batch_antibio)
            preds_y.append(pred.detach().cpu().numpy()[:,:,:-1])
            trues_y.append(true_y.detach().cpu().numpy())
            preds_antibio.append(pred.detach().cpu().numpy()[:,:,-1])
            trues_antibio.append(true_antibio.detach().cpu().numpy())

        preds_y = np.array(preds_y)
        trues_y = np.array(trues_y)
        preds_antibio = np.array(preds_antibio)
        trues_antibio = np.array(trues_antibio)
        print('test shape:', preds_y.shape, trues_y.shape)
        preds_y = preds_y.reshape(-1, preds_y.shape[-2], preds_y.shape[-1])
        trues_y = trues_y.reshape(-1, trues_y.shape[-2], trues_y.shape[-1])
        print('test shape:', preds_y.shape, trues_y.shape)

        # result save
        folder_path = self.args.root_path + self.args.logging_path + '/results_' + self.exp_time + '/' + setting + '/'
        # folder_path = './results/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds_y, trues_y)
        print('mse:{}, mae:{}'.format(mse, mae))
        crossentropy = CEL(preds_antibio, trues_antibio)
        print('crossentropy antibiotics:{}'.format(crossentropy))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe, crossentropy]))
        np.save(folder_path+'pred_y.npy', preds_y)
        np.save(folder_path+'true_y.npy', trues_y)
        np.save(folder_path+'pred_antibios.npy', preds_antibio)
        np.save(folder_path+'true_antibios.npy', trues_antibio)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        self.criterion = self.select_criterion(self.args.loss)
        self.criterion_category = self.select_criterion(self.args.loss_category)

        if load:
            # path = os.path.join(self.args.checkpoints, setting)
            path = self.args.root_path + self.args.checkpoints_path + '/' + setting
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_x_id,batch_y_id,batch_static,batch_antibio) in enumerate(pred_loader):
            pred, true_y, true_antibio = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_static, batch_antibio)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        # folder_path = './results/' + setting +'/'
        folder_path = self.args.root_path + self.args.logging_path + '/results_' + self.exp_time + '/' + setting + '/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_static=None, batch_antibio=None):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        if batch_static is not None: batch_static = batch_static.float().to(self.device)
        if batch_antibio is not None: batch_antibio = batch_antibio.float()

        #Concatenate batch_y (shape: (batch_size, len, 6)) with batch_antibio (shape: (batch_size, len))
        if batch_antibio is not None:
            batch_label = torch.cat([batch_y, batch_antibio.unsqueeze(-1)], dim=-1)
        else:
            batch_label = batch_y
        #Resulting shape: (batch_size, len, 7)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_label.shape[0], self.args.pred_len, batch_label.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_label.shape[0], self.args.pred_len, batch_label.shape[-1]]).float()
        dec_inp = torch.cat([batch_label[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        #Resulting shape: (batch_size, label_len + pred_len, feature_dim)

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cond=batch_static)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cond=batch_static)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cond=batch_static)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, cond=batch_static)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        batch_antibio = batch_antibio[:,-self.args.pred_len:].to(self.device) if batch_antibio is not None else None

        return outputs, batch_y, batch_antibio

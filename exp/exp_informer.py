from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_MEWS
from data.data_loader_eICU import Dataset_eICU
from exp.exp_basic import Exp_Basic
from exp.flexible_BCE_loss import FlexibleBCELoss
from models.model import Informer, InformerStack, InformerMLP
from exp.multi_timeline_2 import categorical_collate

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, CEL
from utils.logger import print_flush

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time
import json
from pathlib import Path
import sys

from functools import partial

import warnings
warnings.filterwarnings('ignore')

# For handling OmegaConf serialization
try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

# Disable output buffering
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
os.environ['PYTHONUNBUFFERED'] = '1'

# def print_flush(*args, **kwargs):
#     """Custom print function that ensures immediate output"""
#     print_flush(*args, **kwargs, flush=True)

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        print_flush(f"[{time.strftime('%H:%M:%S')}] Initializing Informer experiment...")
        super(Exp_Informer, self).__init__(args)
        self.exp_time = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

    # Helper function to check if a batch contains empty tensors
    def _is_empty_batch(self, *tensors):
        """Check if any tensor in the batch is empty (has 0 size in first dimension)"""
        for tensor in tensors:
            if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.size(0) == 0:
                return True
        return False

    def _save_checkpoint(self, epoch, model, optimizer, loss, is_best=False, save_dir=None, filename=None):
        """
        Save model checkpoint with metadata to specified directory.
        
        Args:
            epoch (int): Current epoch number
            model (nn.Module): Model to save
            optimizer (torch.optim): Optimizer state to save
            loss (float): Current loss value
            is_best (bool): Whether this is the best model so far
            save_dir (str): Directory to save the checkpoint
            filename (str): Optional specific filename
        """
        if save_dir is None:
            return

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pth"

        # Prepare checkpoint content
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
        }

        # Save the regular checkpoint
        filepath = os.path.join(save_dir, filename)
        try:
            torch.save(checkpoint, filepath)
            print_flush(f"[{time.strftime('%H:%M:%S')}] Checkpoint saved to {filepath}")
        except Exception as e:
            print_flush(f"[{time.strftime('%H:%M:%S')}] Error saving checkpoint: {str(e)}")

        # Save as best model if it's the best
        if is_best:
            best_path = os.path.join(save_dir, 'best_model.pth')
            try:
                torch.save(checkpoint, best_path)
                print_flush(f"[{time.strftime('%H:%M:%S')}] Best model saved to {best_path}")

                # Also save metadata in a readable format
                meta_info = {
                    'epoch': epoch,
                    'loss': float(loss),  # Convert to float for JSON serialization
                    'timestamp': checkpoint['timestamp'],
                    'checkpoint_path': best_path
                }
                with open(os.path.join(save_dir, 'best_model_info.json'), 'w') as f:
                    json.dump(meta_info, f, indent=4)
            except Exception as e:
                print_flush(f"[{time.strftime('%H:%M:%S')}] Error saving best model: {str(e)}")

        # Remove old checkpoints if there are too many (keep last 5)
        try:
            checkpoints = [f for f in os.listdir(save_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pth')]
            if len(checkpoints) > 5:
                checkpoints.sort(key=lambda x: int(x.split('_epoch_')[1].split('.pth')[0]))
                for old_ckpt in checkpoints[:-5]:
                    os.remove(os.path.join(save_dir, old_ckpt))
                    print_flush(f"[{time.strftime('%H:%M:%S')}] Removed old checkpoint: {old_ckpt}")
        except Exception as e:
            print_flush(f"[{time.strftime('%H:%M:%S')}] Error managing old checkpoints: {str(e)}")

    def _load_checkpoint(self, path, model=None, optimizer=None, map_location=None):
        """
        Load model checkpoint with error handling.
        
        Args:
            path (str): Path to the checkpoint file
            model (nn.Module, optional): Model to load weights into
            optimizer (torch.optim, optional): Optimizer to load state into
            map_location (str or torch.device): Location to map tensors to
            
        Returns:
            dict: Checkpoint data or None if loading failed
        """
        try:
            print_flush(f"[{time.strftime('%H:%M:%S')}] Loading checkpoint from {path}")
            if not os.path.exists(path):
                print_flush(f"[{time.strftime('%H:%M:%S')}] Checkpoint not found at {path}")
                return None

            # First try to load with weights_only=True (secure method) with numpy scalars added to safe list
            try:
                import numpy as np
                # Check if add_safe_globals function exists (PyTorch 2.6+)
                if hasattr(torch.serialization, 'add_safe_globals'):
                    from numpy._core.multiarray import scalar as np_scalar
                    # Add numpy scalars to safe globals list
                    torch.serialization.add_safe_globals([np_scalar])
                    print_flush(f"[{time.strftime('%H:%M:%S')}] Added numpy scalar to safe globals for secure loading")

                checkpoint = torch.load(path, map_location=map_location or self.device)
                print_flush(f"[{time.strftime('%H:%M:%S')}] Checkpoint loaded securely")
            except Exception as e:
                print_flush(f"[{time.strftime('%H:%M:%S')}] Secure loading failed, trying with weights_only=False: {str(e)}")
                # Fallback to weights_only=False (less secure, but works with older checkpoints)
                checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
                print_flush(f"[{time.strftime('%H:%M:%S')}] Checkpoint loaded with weights_only=False")

            if model is not None and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print_flush(f"[{time.strftime('%H:%M:%S')}] Model weights loaded successfully")

            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print_flush(f"[{time.strftime('%H:%M:%S')}] Optimizer state loaded successfully")

            print_flush(f"[{time.strftime('%H:%M:%S')}] Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            return checkpoint

        except Exception as e:
            print_flush(f"[{time.strftime('%H:%M:%S')}] Error loading checkpoint: {str(e)}")
            print_flush(f"[{time.strftime('%H:%M:%S')}] If you're using PyTorch 2.6+, try updating the code to handle the new security restrictions for torch.load()")
            return None

    def _build_model(self):
        print_flush(f"[{time.strftime('%H:%M:%S')}] Building model architecture...")
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
            'informerMLP':InformerMLP,
        }
        if self.args.model=='informer' or self.args.model=='informerstack':
            e_layers = self.args.e_layers if self.args.model=='informer' else self.args.s_layers
            print_flush(f"[{time.strftime('%H:%M:%S')}] Creating {self.args.model} with {e_layers} encoder layers...")
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
        elif self.args.model=='informerMLP':
            e_layers = self.args.e_layers
            model = model_dict[self.args.model](
                self.args.enc_in,
                self.args.dec_in,
                self.args.c_out,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.d_model,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.dropout,
                self.args.embed,
                self.args.freq,
                self.args.activation,
                self.args.distil,
                self.args.condition,
                self.args.n_cond_num_in,
                self.args.n_cond_num_out,
                self.args.n_cond_cat_in,
                self.args.n_cond_cat_out,
                self.args.mlp_hidden_mul,
                self.device
            ).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            print_flush(f"[{time.strftime('%H:%M:%S')}] Setting up model for multi-GPU...")
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        print_flush(f"[{time.strftime('%H:%M:%S')}] Loading {flag} dataset...")
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
            'eICU':Dataset_eICU,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            # shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.freq
            # Data = Dataset_Pred # DEBUG: Disable for now
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq

        print_flush(f"[{time.strftime('%H:%M:%S')}] Creating dataset with parameters:")
        print_flush(f"    - Root path data: {args.root_path_data}")
        print_flush(f"    - Data path: {args.data_path}")
        print_flush(f"    - Sequence length: {args.seq_len}")
        print_flush(f"    - Label length: {args.label_len}")
        print_flush(f"    - Prediction length: {args.pred_len}")

        data_set = Data(
            root_path=args.root_path_data,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols,
            use_preprocessed=args.use_preprocessed_data if hasattr(args, 'use_preprocessed_data') else False,
        )
        print_flush(f"[{time.strftime('%H:%M:%S')}] {flag} dataset size: {len(data_set)}")

        print_flush(f"[{time.strftime('%H:%M:%S')}] Creating DataLoader...")
        print_flush(f"    - Batch size: {batch_size}")
        print_flush(f"    - Shuffle: {shuffle_flag}")
        print_flush(f"    - Num workers: {args.num_workers}")
        print_flush(f"    - Drop last: {drop_last}")

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

    def select_criterion(self, criterion_name, weight=1):
        if criterion_name == 'mse':
            criterion = nn.MSELoss()
        elif criterion_name == 'l1':
            criterion = nn.L1Loss()
        elif criterion_name == 'crossentropy':
            criterion = FlexibleBCELoss(pos_weight=weight, device=self.device)
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

    def vali(self, vali_data, vali_loader, return_preds=False):
        self.model.eval()
        total_loss = []
        total_loss_dict = {"loss_mse": [], "loss_ce": []}
        preds_y = []
        trues_y = []
        ids_y = []
        preds_antibio = []
        trues_antibio = []

        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_x_id,batch_y_id,batch_static,batch_antibio) in enumerate(vali_loader):
            # Skip empty batches
            if self._is_empty_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_static, batch_antibio):
                print_flush(f"[{time.strftime('%H:%M:%S')}] Skipping empty batch in validation")
                continue

            pred, true_y, true_antibio = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_static, batch_antibio)
            # loss = criterion(pred.detach().cpu(), true.detach().cpu())
            loss, loss_dict = self.compute_loss(pred[:,:,:-1].detach().cpu(), true_y.detach().cpu(), pred[:,:,-1].detach().cpu(), true_antibio.detach().cpu(), alpha=self.args.loss_alpha)
            total_loss.append(loss)
            total_loss_dict["loss_mse"].append(loss_dict["loss_mse"])
            total_loss_dict["loss_ce"].append(loss_dict["loss_ce"])

            if return_preds:
                pred_np = pred.detach().cpu().numpy()
                preds_y.append(pred_np[:,:,:-1])
                trues_y.append(true_y.detach().cpu().numpy())
                preds_antibio.append(pred_np[:,:,-1])
                trues_antibio.append(true_antibio.detach().cpu().numpy())
                ids_y.append(batch_y_id)

        total_loss = np.average(total_loss) if total_loss else 0.0
        total_loss_dict["loss_mse"] = np.average(total_loss_dict["loss_mse"]) if total_loss_dict["loss_mse"] else 0.0
        total_loss_dict["loss_ce"] = np.average(total_loss_dict["loss_ce"]) if total_loss_dict["loss_ce"] else 0.0
        self.model.train()

        if return_preds:
            return total_loss, total_loss_dict, preds_y, trues_y, preds_antibio, trues_antibio, ids_y
        return total_loss, total_loss_dict

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')
        self.criterion = self.select_criterion(self.args.loss)
        self.criterion_category = self.select_criterion(self.args.loss_category, weight = self.args.loss_pos_weight)

        # Setup checkpoint directory
        checkpoint_dir = os.path.join(self.args.root_path_save, self.args.checkpoints_path, setting, self.exp_time)
        os.makedirs(checkpoint_dir, exist_ok=True)
        print_flush(f"[{time.strftime('%H:%M:%S')}] Checkpoints will be saved to {checkpoint_dir}")

        # Save experiment configuration
        try:
            with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
                # Try to detect if the config is an OmegaConf object
                if hasattr(self.args, '_metadata') and OmegaConf is not None:
                    # Use OmegaConf's built-in conversion to convert to a basic Python structure
                    config = OmegaConf.to_container(self.args, resolve=True)
                    print_flush(f"[{time.strftime('%H:%M:%S')}] Converting OmegaConf config to basic Python structure")
                else:
                    # Fallback to manual conversion
                    config = {}
                    for k, v in vars(self.args).items():
                        if not k.startswith('__') and not callable(v):
                            # Handle OmegaConf nodes
                            if hasattr(v, '_value'):
                                config[k] = v._value
                            # Handle lists and dicts
                            elif isinstance(v, (list, dict)):
                                config[k] = v
                            # Handle basic types
                            elif isinstance(v, (int, float, str, bool, type(None))):
                                config[k] = v
                            # Convert other types to string
                            else:
                                config[k] = str(v)

                # Final check for any remaining non-serializable objects
                def sanitize_config(obj):
                    if isinstance(obj, dict):
                        return {k: sanitize_config(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [sanitize_config(item) for item in obj]
                    elif isinstance(obj, (int, float, str, bool, type(None))):
                        return obj
                    else:
                        return str(obj)

                config = sanitize_config(config)
                json.dump(config, f, indent=4)
                print_flush(f"[{time.strftime('%H:%M:%S')}] Configuration saved successfully")
        except Exception as e:
            print_flush(f"[{time.strftime('%H:%M:%S')}] Error saving config: {str(e)}")
            print_flush(f"[{time.strftime('%H:%M:%S')}] Will continue without saving config")

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        # criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # Resume from checkpoint if specified
        start_epoch = 0
        best_vali_loss = float('inf')
        if hasattr(self.args, 'resume') and self.args.resume:
            resume_path = self.args.resume
            if os.path.isfile(resume_path):
                checkpoint = self._load_checkpoint(resume_path, self.model, model_optim)
                if checkpoint:
                    start_epoch = checkpoint.get('epoch', 0)
                    best_vali_loss = checkpoint.get('loss', float('inf'))
                    print_flush(f"[{time.strftime('%H:%M:%S')}] Resuming from epoch {start_epoch} with validation loss {best_vali_loss}")
            else:
                print_flush(f"[{time.strftime('%H:%M:%S')}] No checkpoint found at {resume_path}")

        for epoch in range(start_epoch, self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_x_id,batch_y_id,batch_static,batch_antibio) in enumerate(train_loader):
                iter_count += 1

                # Skip empty batches
                if self._is_empty_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_static, batch_antibio):
                    print_flush(f"[{time.strftime('%H:%M:%S')}] Skipping empty batch in training (iteration {i+1})")
                    continue

                model_optim.zero_grad()
                pred, true_y, true_antibio = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_static, batch_antibio)
                loss, loss_dict = self.compute_loss(pred[:,:,:-1], true_y, pred[:,:,-1], true_antibio, alpha=self.args.loss_alpha)
                train_loss.append(loss.item())

                if (i+1) % 100==0:
                    print_flush("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    print_flush("\tloss_mse: {0:.7f} | loss_ce {1:.7f}".format(loss_dict['loss_mse'], loss_dict['loss_ce']))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print_flush('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print_flush("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss) if train_loss else 0.0
            vali_loss, vali_loss_dict = self.vali(vali_data, vali_loader, return_preds=False)
            test_loss, test_loss_dict, preds_y, trues_y, preds_antibio, trues_antibio, ids_y = self.vali(test_data, test_loader, return_preds=True)

            print_flush("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print_flush("Vali Loss: loss_mse: {0:.7f} | loss_ce: {1:.7f}".format(vali_loss_dict['loss_mse'], vali_loss_dict['loss_ce']))

            # Create epoch-specific directory
            epoch_dir = os.path.join(checkpoint_dir, f'epoch_{epoch+1}_test_results')
            os.makedirs(epoch_dir, exist_ok=True)

            # Save checkpoint for this epoch in the epoch-specific directory
            self._save_checkpoint(
                epoch + 1,
                self.model,
                model_optim,
                vali_loss,
                is_best=False,
                save_dir=epoch_dir,
                filename=f'model_epoch_{epoch+1}.pth'
            )

            # Handle early stopping and best model
            is_best = vali_loss < best_vali_loss
            if is_best:
                best_vali_loss = vali_loss

                # Save best in main checkpoint directory for easy access
                self._save_checkpoint(
                    epoch + 1,
                    self.model,
                    model_optim,
                    vali_loss,
                    is_best=True,
                    save_dir=checkpoint_dir,
                    filename="best_model.pth"
                )

            early_stopping(vali_loss, self.model, checkpoint_dir)
            if early_stopping.early_stop:
                print_flush("Early stopping")
                break

            # Save test predictions and true values for this epoch
            if preds_y:  # Only save if we have predictions
                try:
                    #each results are a list in shape (iterations, varying batch_size, pred_len, c_out)
                    #pred_len and c_out are consistent but batch_size and iterations are not, making it impossible to convert to np array directly
                    #we want to turn them into a (n, pred_len, c_out) data shape
                    #so we need to reshape them but preds_y is a list not a numpy item

                    preds_y = np.concatenate(preds_y, axis=0)
                    trues_y = np.concatenate(trues_y, axis=0)
                    preds_antibio = np.concatenate(preds_antibio, axis=0)
                    trues_antibio = np.concatenate(trues_antibio, axis=0)
                    ids_y = np.concatenate(ids_y, axis=0)

                    # Save the predictions and true values
                    np.save(os.path.join(epoch_dir, 'preds_y.npy'), preds_y)
                    np.save(os.path.join(epoch_dir, 'trues_y.npy'), trues_y)
                    np.save(os.path.join(epoch_dir, 'preds_antibio.npy'), preds_antibio)
                    np.save(os.path.join(epoch_dir, 'trues_antibio.npy'), trues_antibio)
                    np.save(os.path.join(epoch_dir, 'id_y.npy'), ids_y)

                    # Save metrics for this epoch
                    mae, mse, rmse, mape, mspe = metric(preds_y, trues_y)
                    crossentropy = CEL(preds_antibio, trues_antibio)

                    metrics_data = {
                        'mae': float(mae),
                        'mse': float(mse),
                        'rmse': float(rmse),
                        'mape': float(mape),
                        'mspe': float(mspe),
                        'crossentropy': float(crossentropy),
                        'epoch': epoch + 1,
                        'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
                    }

                    with open(os.path.join(epoch_dir, 'metrics.json'), 'w') as f:
                        json.dump(metrics_data, f, indent=4)

                    print_flush(f"Saved test results and model checkpoint for epoch {epoch+1} to {epoch_dir}")
                except Exception as e:
                    print_flush(f"Error saving test results for epoch {epoch+1}: {str(e)}")

            adjust_learning_rate(model_optim, epoch+1, self.args)

        # Load the best model before returning
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            self._load_checkpoint(best_model_path, self.model)
        else:
            print_flush(f"[{time.strftime('%H:%M:%S')}] Best model not found at {best_model_path}")
            # Try to find the most recent checkpoint
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
            if checkpoints:
                latest = sorted(checkpoints, key=lambda x: int(x.split('_epoch_')[1].split('.pth')[0]))[-1]
                self._load_checkpoint(os.path.join(checkpoint_dir, latest), self.model)
                print_flush(f"[{time.strftime('%H:%M:%S')}] Loaded most recent checkpoint: {latest}")
            else:
                print_flush(f"[{time.strftime('%H:%M:%S')}] No checkpoints found in {checkpoint_dir}")

        return self.model

    def test(self, setting, load_best=True):
        test_data, test_loader = self._get_data(flag='test')
        self.criterion = self.select_criterion(self.args.loss)
        self.criterion_category = self.select_criterion(self.args.loss_category)

        # Load the best model if specified
        if load_best:
            checkpoint_dir = os.path.join(self.args.root_path_save, self.args.checkpoints_path, setting)
            # Try to find the experiment directory
            if os.path.exists(checkpoint_dir):
                # First, look in the specified experiment time directory if it exists
                if hasattr(self, 'exp_time') and self.exp_time:
                    exp_dir = os.path.join(checkpoint_dir, self.exp_time)
                    if os.path.exists(exp_dir):
                        best_model_path = os.path.join(exp_dir, 'best_model.pth')
                        if os.path.exists(best_model_path):
                            self._load_checkpoint(best_model_path, self.model)
                        else:
                            print_flush(f"[{time.strftime('%H:%M:%S')}] Best model not found in {exp_dir}")
                            # Try to find the most recent experiment
                            self._find_and_load_latest_model(checkpoint_dir)
                    else:
                        # Try to find the most recent experiment
                        self._find_and_load_latest_model(checkpoint_dir)
                else:
                    # Try to find the most recent experiment
                    self._find_and_load_latest_model(checkpoint_dir)
            else:
                print_flush(f"[{time.strftime('%H:%M:%S')}] Checkpoint directory not found at {checkpoint_dir}")

        self.model.eval()

        preds_y = []
        trues_y = []
        preds_antibio = []
        trues_antibio = []
        ids_y = []

        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_x_id,batch_y_id,batch_static,batch_antibio) in enumerate(test_loader):
            # Skip empty batches
            if self._is_empty_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_static, batch_antibio):
                print_flush(f"[{time.strftime('%H:%M:%S')}] Skipping empty batch in testing")
                continue

            pred, true_y, true_antibio = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_static, batch_antibio)
            pred_np = pred.detach().cpu().numpy()
            preds_y.append(pred_np[:,:,:-1])
            trues_y.append(true_y.detach().cpu().numpy())
            preds_antibio.append(pred_np[:,:,-1])
            trues_antibio.append(true_antibio.detach().cpu().numpy())
            ids_y.append(batch_y_id)

        # Safety check to ensure we have predictions before trying to process them
        if not preds_y:
            print_flush(f"[{time.strftime('%H:%M:%S')}] Warning: No valid predictions found during testing")
            return

        preds_y = np.concatenate(preds_y, axis=0)
        trues_y = np.concatenate(trues_y, axis=0)
        preds_antibio = np.concatenate(preds_antibio, axis=0)
        trues_antibio = np.concatenate(trues_antibio, axis=0)
        ids_y = np.concatenate(ids_y, axis=0)

        # result save
        folder_path = os.path.join(self.args.root_path_save, self.args.logging_path, 'results_' + self.exp_time, setting)
        os.makedirs(folder_path, exist_ok=True)

        mae, mse, rmse, mape, mspe = metric(preds_y, trues_y)
        print_flush('mse:{}, mae:{}'.format(mse, mae))
        crossentropy = CEL(preds_antibio, trues_antibio)
        print_flush('crossentropy antibiotics:{}'.format(crossentropy))

        # Save test metrics to JSON for easier reading
        try:
            metrics_data = {
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'mspe': float(mspe),
                'crossentropy': float(crossentropy),
                'timestamp': time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
            }
            with open(os.path.join(folder_path, 'metrics.json'), 'w') as f:
                json.dump(metrics_data, f, indent=4)
        except Exception as e:
            print_flush(f"[{time.strftime('%H:%M:%S')}] Error saving metrics JSON: {str(e)}")

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe, crossentropy]))
        np.save(folder_path+'pred_y.npy', preds_y)
        np.save(folder_path+'true_y.npy', trues_y)
        np.save(folder_path+'pred_antibios.npy', preds_antibio)
        np.save(folder_path+'true_antibios.npy', trues_antibio)
        np.save(folder_path+'id_y.npy', ids_y)

        return

    def _find_and_load_latest_model(self, checkpoint_dir):
        """Find and load the most recent experiment's best model"""
        try:
            # List all subdirectories (experiment timestamps)
            exp_dirs = [d for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
            if not exp_dirs:
                print_flush(f"[{time.strftime('%H:%M:%S')}] No experiment directories found in {checkpoint_dir}")
                return False

            # Sort by timestamp (assuming format YYYY-MM-DD_HH-MM-SS)
            exp_dirs.sort(reverse=True)  # Most recent first

            # Try to find a best model in each directory
            for exp_dir in exp_dirs:
                full_exp_dir = os.path.join(checkpoint_dir, exp_dir)
                best_model_path = os.path.join(full_exp_dir, 'best_model.pth')
                if os.path.exists(best_model_path):
                    self._load_checkpoint(best_model_path, self.model)
                    print_flush(f"[{time.strftime('%H:%M:%S')}] Loaded best model from {exp_dir}")
                    return True

                # If no best_model.pth, try to find the latest checkpoint
                checkpoints = [f for f in os.listdir(full_exp_dir) if f.startswith('checkpoint_epoch_')]
                if checkpoints:
                    latest = sorted(checkpoints, key=lambda x: int(x.split('_epoch_')[1].split('.pth')[0]))[-1]
                    self._load_checkpoint(os.path.join(full_exp_dir, latest), self.model)
                    print_flush(f"[{time.strftime('%H:%M:%S')}] Loaded checkpoint {latest} from {exp_dir}")
                    return True

            print_flush(f"[{time.strftime('%H:%M:%S')}] No model checkpoints found in any experiment directory")
            return False

        except Exception as e:
            print_flush(f"[{time.strftime('%H:%M:%S')}] Error finding latest model: {str(e)}")
            return False

    def predict(self, setting, load=True):
        pred_data, pred_loader = self._get_data(flag='pred')
        self.criterion = self.select_criterion(self.args.loss)
        self.criterion_category = self.select_criterion(self.args.loss_category)

        if load:
            checkpoint_dir = os.path.join(self.args.root_path_save, self.args.checkpoints_path, setting)
            # Try to load the best model similarly to the test method
            if not self._find_and_load_latest_model(checkpoint_dir):
                print_flush(f"[{time.strftime('%H:%M:%S')}] Failed to load any model for prediction")
                return

        self.model.eval()

        preds = []

        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark,batch_x_id,batch_y_id,batch_static,batch_antibio) in enumerate(pred_loader):
            # Skip empty batches
            if self._is_empty_batch(batch_x, batch_y, batch_x_mark, batch_y_mark, batch_static, batch_antibio):
                print_flush(f"[{time.strftime('%H:%M:%S')}] Skipping empty batch in prediction")
                continue

            pred, true_y, true_antibio = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark, batch_static, batch_antibio)
            preds.append(pred.detach().cpu().numpy())

        # Safety check to ensure we have predictions before trying to process them
        if not preds:
            print_flush(f"[{time.strftime('%H:%M:%S')}] Warning: No valid predictions found")
            return

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = os.path.join(self.args.root_path_save, self.args.logging_path, 'results_' + self.exp_time, setting)
        os.makedirs(folder_path, exist_ok=True)

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
            # batch_label = torch.cat([batch_y, batch_antibio.unsqueeze(-1)], dim=-1)
            batch_label = torch.cat([batch_y, torch.zeros_like(batch_antibio).unsqueeze(-1)], dim=-1)
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

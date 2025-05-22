import hydra
import time
import sys
import os
from utils.logger import print_flush
# import mac_cuda_fake

# def print_flush(*args, **kwargs):
#     """Custom print function that ensures immediate output"""
#     print(*args, **kwargs, flush=True)

# print_flush(f"[{time.strftime('%H:%M:%S')}] Starting program...")
from omegaconf import DictConfig, OmegaConf
import torch

from exp.exp_informer import Exp_Informer

DATA_PARSER = {
    'ETTh1': {'data': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTh2': {'data': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm1': {'data': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'ETTm2': {'data': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1]},
    'WTH':   {'data': 'WTH.csv',   'T': 'WetBulbCelsius', 'M': [12, 12, 12], 'S': [1, 1, 1], 'MS': [12, 12, 1]},
    'ECL':   {'data': 'ECL.csv',   'T': 'MT_320', 'M': [321, 321, 321], 'S': [1, 1, 1], 'MS': [321, 321, 1]},
    'Solar': {'data': 'solar_AL.csv', 'T': 'POWER_136', 'M': [137, 137, 137], 'S': [1, 1, 1], 'MS': [137, 137, 1]},
}

@hydra.main(config_path="./configs", config_name="config", version_base=None)
def main(args: DictConfig):
    print_flush(f"[{time.strftime('%H:%M:%S')}] Initializing configuration...")
    OmegaConf.set_struct(args, False)  # turns off struct mode
    
    # Handle backward compatibility for root_path parameter
    if hasattr(args, 'root_path') and not hasattr(args, 'root_path_data'):
        print_flush(f"[{time.strftime('%H:%M:%S')}] Converting root_path to root_path_data for backward compatibility")
        args.root_path_data = args.root_path
    
    # Set root_path_save to root_path_data if not specified
    if not hasattr(args, 'root_path_save'):
        print_flush(f"[{time.strftime('%H:%M:%S')}] root_path_save not specified, using root_path_data for saving")
        args.root_path_save = args.root_path_data
    
    print_flush(f"[{time.strftime('%H:%M:%S')}] Data loading from: {args.root_path_data}")
    print_flush(f"[{time.strftime('%H:%M:%S')}] Results saving to: {args.root_path_save}")
    
    # 1. Decide if GPU will be used
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    print_flush(f"[{time.strftime('%H:%M:%S')}] GPU usage: {args.use_gpu} (Available: {torch.cuda.is_available()})")

    # 2. Handle multi-GPU logic if requested
    if args.use_gpu and args.use_multi_gpu:
        print_flush(f"[{time.strftime('%H:%M:%S')}] Setting up multi-GPU configuration...")
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        print_flush(f"[{time.strftime('%H:%M:%S')}] Using GPU devices: {args.device_ids}")

    # 3. Override certain settings based on known data sets
    if args.data in DATA_PARSER:
        print_flush(f"[{time.strftime('%H:%M:%S')}] Setting up data configuration for {args.data}...")
        data_info = DATA_PARSER[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]
        print_flush(f"[{time.strftime('%H:%M:%S')}] Data path: {args.data_path}, Target: {args.target}")
        print_flush(f"[{time.strftime('%H:%M:%S')}] Model dimensions - Encoder in: {args.enc_in}, Decoder in: {args.dec_in}, Output: {args.c_out}")

    # 4. Convert s_layers from string to list[int] if needed
    if isinstance(args.s_layers, str):
        print_flush(f"[{time.strftime('%H:%M:%S')}] Converting s_layers from string to list...")
        args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]

    # 5. Adjust frequencies
    args.detail_freq = args.freq
    # args.freq = args.freq[-1:]  # e.g., "h" from "15min"
    args.freq = args.freq  # e.g., "h" from "15min"
    print_flush(f"[{time.strftime('%H:%M:%S')}] Frequency settings - Detail: {args.detail_freq}, Base: {args.freq}")

    print_flush(f"\n[{time.strftime('%H:%M:%S')}] Full Configuration (OmegaConf):")
    print_flush(OmegaConf.to_yaml(args))

    print_flush(f"\n[{time.strftime('%H:%M:%S')}] Initializing Informer experiment...")
    Exp = Exp_Informer

    for ii in range(args.itr):
        print_flush(f"\n[{time.strftime('%H:%M:%S')}] Starting iteration {ii+1}/{args.itr}")
        # setting record of experiments
        setting = "{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}".format(
            args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn,
            args.factor, args.embed, args.distil, args.mix, args.des, ii
        )
        print_flush(f"[{time.strftime('%H:%M:%S')}] Experiment setting: {setting}")

        print_flush(f"[{time.strftime('%H:%M:%S')}] Initializing experiment instance...")
        exp = Exp(args)  # initialize experiment
        
        print_flush(f'[{time.strftime("%H:%M:%S")}] >>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)

        print_flush(f'[{time.strftime("%H:%M:%S")}] >>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting)

        if args.do_predict:
            print_flush(f'[{time.strftime("%H:%M:%S")}] >>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.predict(setting, True)

        print_flush(f"[{time.strftime('%H:%M:%S')}] Clearing GPU cache...")
        torch.cuda.empty_cache()

    print_flush(f"[{time.strftime('%H:%M:%S')}] Program completed.")


if __name__ == "__main__":
    main()

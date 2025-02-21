import hydra
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
    OmegaConf.set_struct(args, False)  # turns off struct mode
    
    # 1. Decide if GPU will be used
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)

    # 2. Handle multi-GPU logic if requested
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # 3. Override certain settings based on known data sets
    if args.data in DATA_PARSER:
        data_info = DATA_PARSER[args.data]
        args.data_path = data_info['data']
        args.target = data_info['T']
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    # 4. Convert s_layers from string to list[int] if needed
    if isinstance(args.s_layers, str):
        args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ', '').split(',')]

    # 5. Adjust frequencies
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]  # e.g., "h" from "15min"

    print("Configuration (OmegaConf) in experiment:")
    print(OmegaConf.to_yaml(args))

    Exp = Exp_Informer

    for ii in range(args.itr):
        # setting record of experiments
        setting = "{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}".format(
            args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn,
            args.factor, args.embed, args.distil, args.mix, args.des, ii
        )

        exp = Exp(args)  # initialize experiment
        print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
        exp.train(setting)

        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting)

        if args.do_predict:
            print(f'>>>>>>>predicting : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.predict(setting, True)

        # Clear GPU cache
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

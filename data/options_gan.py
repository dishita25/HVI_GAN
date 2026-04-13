import argparse

def _str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def option():
    # Training settings
    parser = argparse.ArgumentParser(description='HVI-GAN')
    parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
    parser.add_argument('--cropSize', type=int, default=256, help='image crop size (patch size)')
    parser.add_argument('--nEpochs', type=int, default=1000, help='number of epochs to train for end')
    parser.add_argument('--start_epoch', type=int, default=0, help='number of epochs to start, >0 is retrained a pre-trained pth')
    parser.add_argument('--snapshots', type=int, default=10, help='Snapshots for save checkpoints pth')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate for generator')
    parser.add_argument('--lr_D', type=float, default=1e-4, help='Learning Rate for discriminators')
    parser.add_argument('--gpu_mode', type=_str2bool, default=True)
    parser.add_argument('--shuffle', type=_str2bool, default=True)
    parser.add_argument('--threads', type=int, default=16, help='number of threads for dataloader to use')

    # choose a scheduler
    parser.add_argument('--cos_restart_cyclic', type=_str2bool, default=False)
    parser.add_argument('--cos_restart', type=_str2bool, default=True)

    # warmup training
    parser.add_argument('--warmup_epochs', type=int, default=3, help='warmup_epochs')
    parser.add_argument('--start_warmup', type=_str2bool, default=True, help='turn False to train without warmup') 

    # train datasets
    parser.add_argument('--data_train_lol_blur'     , type=str, default='./datasets/LOL_blur/train')
    parser.add_argument('--data_train_lol_v1'       , type=str, default='/kaggle/input/datasets/soumikrakshit/lol-dataset/lol_dataset/our485')
    parser.add_argument('--data_train_lolv2_real'   , type=str, default='./datasets/LOLv2/Real_captured/Train')
    parser.add_argument('--data_train_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Train')
    parser.add_argument('--data_train_SID'          , type=str, default='./datasets/Sony_total_dark/train')
    parser.add_argument('--data_train_SICE'         , type=str, default='./datasets/SICE/Dataset/train')
    parser.add_argument('--data_train_fivek'        , type=str, default='./datasets/FiveK/train')

    # validation input
    parser.add_argument('--data_val_lol_blur'       , type=str, default='./datasets/LOL_blur/eval/low_blur')
    parser.add_argument('--data_val_lol_v1'         , type=str, default='/kaggle/input/datasets/soumikrakshit/lol-dataset/lol_dataset/eval15/low')
    parser.add_argument('--data_val_lolv2_real'     , type=str, default='./datasets/LOLv2/Real_captured/Test/Low')
    parser.add_argument('--data_val_lolv2_syn'      , type=str, default='./datasets/LOLv2/Synthetic/Test/Low')
    parser.add_argument('--data_val_SID'            , type=str, default='./datasets/Sony_total_dark/eval/short')
    parser.add_argument('--data_val_SICE_mix'       , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_val_SICE_grad'      , type=str, default='./datasets/SICE/Dataset/eval/test')
    parser.add_argument('--data_val_fivek'          , type=str, default='./datasets/FiveK/test/input')

    # validation groundtruth
    parser.add_argument('--data_valgt_lol_blur'     , type=str, default='./datasets/LOL_blur/eval/high_sharp_scaled/')
    parser.add_argument('--data_valgt_lol_v1'       , type=str, default='/kaggle/input/datasets/soumikrakshit/lol-dataset/lol_dataset/eval15/high/')
    parser.add_argument('--data_valgt_lolv2_real'   , type=str, default='./datasets/LOLv2/Real_captured/Test/Normal/')
    parser.add_argument('--data_valgt_lolv2_syn'    , type=str, default='./datasets/LOLv2/Synthetic/Test/Normal/')
    parser.add_argument('--data_valgt_SID'          , type=str, default='./datasets/Sony_total_dark/eval/long/')
    parser.add_argument('--data_valgt_SICE_mix'     , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_SICE_grad'    , type=str, default='./datasets/SICE/Dataset/eval/target/')
    parser.add_argument('--data_valgt_fivek'        , type=str, default='./datasets/FiveK/test/target/')

    parser.add_argument('--val_folder', default='./results/', help='Location to save validation datasets')

    # Original reconstruction loss weights (applied to BOTH RGB and HVI)
    parser.add_argument('--HVI_weight', type=float, default=1.0,
                        help='Weight for the HVI-space reconstruction losses')
    parser.add_argument('--L1_weight', type=float, default=1.0)
    parser.add_argument('--D_weight',  type=float, default=0.5,
                        help='Weight for SSIM loss')
    parser.add_argument('--E_weight',  type=float, default=50.0,
                        help='Weight for edge loss')
    parser.add_argument('--P_weight',  type=float, default=1e-2,
                        help='Weight for perceptual loss')

    # GAN-specific weights
    parser.add_argument('--adv_weight_rgb', type=float, default=0.01, help='Weight for the RGB adversarial loss term in G')
    parser.add_argument('--adv_weight_hvi', type=float, default=0.01, help='Weight for the HVI adversarial loss term in G')
    parser.add_argument('--D_rgb_weight', type=float, default=1.0, help='Weight scalar for the RGB discriminator loss')
    parser.add_argument('--D_hvi_weight', type=float, default=1.0, help='Weight scalar for the HVI discriminator loss')

    # How many generator steps per discriminator step (n_critic-style)
    parser.add_argument('--n_critic', type=int, default=1, help='Train discriminator every n_critic generator steps')

    # Warm-up the GAN signal: start adversarial training after this many epochs before the discriminator sees them (prevents early mode collapse).
    parser.add_argument('--gan_start_epoch', type=int, default=10, help='Epoch at which adversarial losses are switched on')

    # use random gamma function (enhancement curve) to improve generalisation
    parser.add_argument('--gamma', type=_str2bool, default=False)
    parser.add_argument('--start_gamma', type=int, default=60)
    parser.add_argument('--end_gamma', type=int, default=120)

    # auto grad
    parser.add_argument('--grad_detect', type=_str2bool, default=False, help='if gradient explosion occurs, turn on')
    parser.add_argument('--grad_clip', type=_str2bool, default=True, help='if gradient fluctuates too much, turn on')

    # choose which dataset you want to train
    parser.add_argument('--dataset', type=str, default='lol_v1',
    choices=['lol_v1',
             'lolv2_real',
             'lolv2_syn',
             'lol_blur', 
             'SID',
             'SICE_mix',
             'SICE_grad',
             'fivek'],
    help='Select the dataset to train on (default: %(default)s)')

    return parser

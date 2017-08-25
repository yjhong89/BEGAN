import numpy as np
import tensorflow as tf
import model
import time, os, sys, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_update_step', type=int, default=1e4)
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--save_step', type=int, default=100)
    parser.add_argument('--lr_for_k', type=float, default=0.001)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint')
    parser.add_argument('--sample_dir', type=str, default='./samples')
    parser.add_argument('--log_dir', type=str, default='./tensorboard_log')
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--eta', type=int, default=1)
    parser.add_argument('--showing_height', type=int, default=8)
    parser.add_argument('--showing_width', type=int, default=8)
    parser.add_argument('--use_bn', type=str2bool, default='no')
    parser.add_argument('--target_size', type=int, default=64)
    parser.add_argument('--input_size', type=int, default=108)
    parser.add_argument('--num_examples_per_epoch', type=int, default=1000)
    parser.add_argument('--data_sets', type=str, default='../CelebA')
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--embedding_size', type=int, default=128)
    parser.add_argument('--filter_depth', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--is_training', type=str2bool, default='train')   

    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    if not os.path.exists(args.sample_dir):
        os.mkdir(args.sample_dir)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    run_config.log_device_placement = False 

    with tf.Session(config=run_config) as sess:
        began = model.BEGAN(args, sess)
        if args.is_training:
            print('Start training')
            began.train()
        else:
            print('Test')
            began.test()


def str2bool(v):
    if v.lower() in ('yes', 'true', 'train', '1'):
        return True
    if v.lower() in ('no', 'test', '0', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Not expected boolean type')
    
if __name__ == "__main__":
    main()

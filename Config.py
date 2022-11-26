import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# data of dataset
parser.add_argument('--train_data_path', default="./data/preprocessed_data/train", help='train dataset  path')
parser.add_argument('--test_data_path', default='./data/preprocessed_data/test', help='test dataset path')
parser.add_argument('--index_path', default="./data/index.csv", help='index file path')

# train
parser.add_argument('--gpu', action='store_true', help='whether use gpu')
parser.add_argument('--amp', action='store_true', help='whether use automatic mixed precision')
parser.add_argument('--normalize', action='store_true', help='Whether normalize the data to the range -1 to 1')
parser.add_argument('--model', default="hdenseunet", help='option: unet3d hdenseunet')
parser.add_argument('--num_classes', type=int, default=3, help='2:only segment liver 3:both segment liver and tumor')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--epoch_num', type=int, default=100, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
parser.add_argument('--num_workers', default=8, type=int, help='dataloader num workers (default: 8)')
parser.add_argument('--train_crop_size', type=int, default=16, help='crop size for 3D data in training(default: 16)')
parser.add_argument('--save_epoch', type=int, default=2,
                    help='validate and save model epoch interval(default: 2)')
parser.add_argument('--log_iteration', type=int, default=10, help='log information iteration interval(default: 10)')
parser.add_argument('--checkpoint_path',  default='./checkpoint', help="checkpoint path for intermediate weight")

# test
parser.add_argument('--test_crop_size', type=int, default=16, help='crop size for 3D data in testing(default: 16)')


args = parser.parse_args()

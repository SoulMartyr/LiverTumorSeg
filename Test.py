import Config
from models.HDenseUNet import AmpHDenseUNet
from models.HDenseUNetV2 import AmpHDenseUNetV2
from models.UNet3D import AmpUNet3D
from utils.DataFunc import *
from utils.Logger import *
from utils.LossFunc import *


def test(test_loader, model):
    model.eval()
    test_tot_loss = []
    test_tot_dice = []
    for _, batch in enumerate(test_loader):
        ct = batch['ct'].cuda()
        seg = batch['seg'].cuda()
        with torch.no_grad():
            out = model(ct)
            test_loss = weighted_cross_entropy_loss(out, seg)
            if out.shape[1] == 2:
                test_dice = liver_dice(out, seg)
            else:
                test_dice = tumor_dice(out, seg)
        print(test_loss, test_dice)
        test_tot_loss.append(test_loss.item())
        test_tot_dice.append(test_dice.item())
    return np.mean(test_tot_loss), np.max(test_tot_dice)


if __name__ == "__main__":

    args = Config.args

    # device = torch.device('cuda' if args.gpu else 'cpu')

    # if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    is_amp = args.amp

    test_crop_slices = args.test_crop_size
    num_classes = args.num_classes

    model_name = args.model
    if model_name == "unet3d":
        model = AmpUNet3D(out_channels=num_classes)
    elif model_name == "hdenseunet":
        model = AmpHDenseUNet(num_slices=test_crop_slices, out_channels=num_classes)
    elif model_name == "hdenseunetv2":
        model = AmpHDenseUNetV2(out_channels=num_classes)
    else:
        raise NameError("No model named" + model_name)

    batch_size = args.batch_size
    num_workers = args.num_workers

    # Load Data
    index_df = pd.read_csv(args.index_path, index_col=0)
    test_index = index_df.loc["test", "index"].strip().split(" ")

    test_data_path = args.test_data_path

    test_dataset = TiLSDataSet(data_path=test_data_path, index_list=test_index,
                               crop_slices=test_crop_slices, num_classes=num_classes, is_normalize=args.normalize)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Load Model
    model = model.cuda()

    checkpoint_path = args.checkpoint_path
    weight_dir = os.path.join(checkpoint_path, '{}'.format(model.__class__.__name__))
    if not os.path.exists(weight_dir):
        raise NameError("No weight dir" + weight_dir)

    weights = os.listdir(weight_dir)
    if not os.path.exists(weight_dir):
        raise FileNotFoundError("Not found weight")

    weights = sorted(weights, key=lambda x: os.path.getmtime(os.path.join(weight_dir, x)))
    weight_path = os.path.join(weight_dir, weights[-1])

    weight = torch.load(weight_path)
    model.load_state_dict(weight['model_state_dict'], strict=False)

    # Start Test
    test_mean_loss, test_mean_dice = test(test_loader, model)
    print("mean loss:", test_mean_loss)
    print("best tumor dice:", test_mean_dice)

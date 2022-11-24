from torch import optim
from torch.cuda import amp

import Config
from models.TestModel import AmpTestModel
from utils.DataFunc import *
from utils.Logger import *
from utils.LossFunc import *
from utils.UtilFunc import *


def test(test_loader, model):
    model.eval()
    valid_tot_loss = 0.
    valid_tot_dice = 0.
    for _, batch in enumerate(test_loader):
        ct = batch['ct'].cuda()
        seg = batch['seg'].cuda()
        with torch.no_grad():
            out = model(ct)
            valid_loss = weighted_cross_entropy_loss(out, seg)
            if out.shape[1] == 2:
                valid_dice = liver_dice(out, seg)
            else:
                valid_dice = tumor_dice(out, seg)
        valid_tot_loss += valid_loss.item()
        valid_tot_dice += valid_dice.item()

    return [valid_tot_loss / len(test_loader), valid_tot_dice / len(test_loader)]


def save_weight(weight_dir, epoch, iteration, model_state_dict, optim_state_dict, is_amp, scaler_state_dict):
    checkpoint = {
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model_state_dict,
        'optim_state_dict': optim_state_dict
    }
    if is_amp:
        checkpoint['scaler_state_dict'] = scaler_state_dict
    torch.save(checkpoint, weight_dir + '/{}epoch_{}iter.pth'.format(epoch, iteration))


def train(start_epoch, start_iteration, train_loader, test_loader, model, is_amp, optimizer, grad_scaler,
          save_epoch, weight_dir, log_iteration, log_file):
    log_msg_head(epoch_num, batch_size, log_file)

    best_dice = 0.

    epoch = start_epoch
    iteration = start_iteration

    while epoch < epoch_num:
        if epoch != start_epoch:
            iteration = 0
        model.train()
        is_epoch_saved = False
        for _, batch in enumerate(train_loader):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            is_save = False
            if not is_epoch_saved and epoch % save_epoch == 0 and epoch != start_epoch:
                test_accuracy = test(test_loader, model)
                model.train()
                if test_accuracy[1] < best_dice:
                    save_weight(weight_dir, epoch, iteration, model.state_dict(), optimizer.state_dict(), is_amp,
                                grad_scaler.state_dict())
                    best_dice = test_accuracy
                    is_save = True
                is_epoch_saved = True
            else:
                test_accuracy = [None, None]

            ct = batch['ct'].cuda()
            seg = batch['seg'].cuda()
            optimizer.zero_grad()

            with amp.autocast():
                out = model(ct)
                train_loss = weighted_cross_entropy_loss(out, seg)
                if out.shape[1] == 2:
                    train_dice = liver_dice(out, seg)
                else:
                    train_dice = tumor_dice(out, seg)
                train_accuracy = [train_loss.item(), train_dice.item()]

            grad_scaler.scale(train_loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            if iteration % log_iteration == 0 and (iteration == 0 or iteration != start_iteration):
                lr = get_learning_rate(optimizer)
                log_msg(epoch, iteration, lr, train_accuracy, test_accuracy, is_save, log_file)

            iteration += 1

        epoch += 1
    return best_dice


if __name__ == "__main__":

    args = Config.args

    # device = torch.device('cuda' if args.gpu else 'cpu')

    # if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    is_amp = args.amp

    train_crop_slices = args.train_crop_size
    test_crop_slices = args.test_crop_size
    num_classes = args.num_classes

    model_name = args.model

    model = AmpTestModel(out_channels=num_classes)
    log_file = set_logfile(AmpTestModel.__name__)

    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.lr
    epoch_num = args.epoch_num
    save_epoch = args.save_epoch
    log_iteration = args.log_iteration

    # Load Data
    index_df = pd.read_csv(args.index_path, index_col=0)
    train_index = index_df.loc["train", "index"].strip().split(" ")
    test_index = index_df.loc["test", "index"].strip().split(" ")

    train_data_path = args.train_data_path
    test_data_path = args.test_data_path

    train_dataset = TiLSDataSet(data_path=train_data_path, index_list=train_index,
                                crop_slices=train_crop_slices, num_classes=num_classes)
    test_dataset = TiLSDataSet(data_path=test_data_path, index_list=test_index,
                               crop_slices=test_crop_slices, num_classes=num_classes)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True
    )

    log_hint("Load DataSet Success", log_file)

    # Load Model
    grad_scaler = amp.GradScaler()
    model = AmpTestModel(out_channels=num_classes).cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_path = args.checkpoint_path
    weight_dir = os.path.join(checkpoint_path, '{}'.format(model.__class__.__name__))
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)

    weights = os.listdir(weight_dir)
    weight_path = None

    if len(weights) != 0:
        weights = sorted(weights, key=lambda x: os.path.getmtime(os.path.join(weight_dir, x)))
        weight_path = os.path.join(weight_dir, weights[-1])

    if weight_path is not None:
        weight = torch.load(weight_path)
        start_iteration = weight['iteration']
        start_epoch = weight['epoch']
        model.load_state_dict(weight['model_state_dict'], strict=False)
        optimizer.load_state_dict(weight['optim_state_dict'])
        if is_amp:
            grad_scaler.load_state_dict(weight['scaler_state_dict'])
        print("start_epoch:", start_epoch, "start_iteration:", start_iteration)
    else:
        start_iteration = 0
        start_epoch = 0
    log_hint("Load Model And Optimizer Success", log_file)

    # Start Train
    best_dice = train(start_epoch, start_iteration, train_loader, test_loader, model, is_amp, optimizer, grad_scaler,
                      save_epoch, weight_dir, log_iteration, log_file)
    log_hint("Best Dice: {}".format(str(best_dice)), log_file)

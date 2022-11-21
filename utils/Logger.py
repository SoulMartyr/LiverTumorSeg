import time


def get_time():
    now = time.localtime()
    now_time = time.strftime("%H:%M:%S", now)
    return now_time


def log_hint(hint: str):
    print(get_time(), hint)


def log_msg_head(epoch_num: int, batch_size: int):
    print('epoch_num = {}'.format(epoch_num))
    print('batch_size = {}'.format(batch_size))
    print('|---------------Info----------------|------Train------|------Valid------|')
    print('| time       epoch    iter   lr     | loss     dice   | loss     dice   |')
    print('-------------------------------------------------------------------------')


def log_msg(epoch: int, iteration: int, lr: float, train_accuracy: float, valid_accuracy: float, is_save: bool):
    if is_save:
        sign = '*'
    else:
        sign = ' '
    train_accuracy = ['None' if elem is None else str(elem) for elem in train_accuracy]
    valid_accuracy = ['None' if elem is None else str(elem) for elem in valid_accuracy]
    print("| {:<8}   {:<6}   {:<4}   {:<6} | {:<6}   {:<6} | {:<6}   {:<6} |".format(get_time(), str(epoch)[:6],
                                                                                     (str(iteration) + sign)[:4],
                                                                                     str(lr)[:6],
                                                                                     str(train_accuracy[0])[:6],
                                                                                     str(train_accuracy[1])[:6],
                                                                                     str(valid_accuracy[0])[:6],
                                                                                     str(valid_accuracy[1])[:6]))


def log_flush():
    print('\r', end='', flush=True)


def log_epoch(iteration: int, batch_loss: float, batch_dist: float):
    print(iteration, "Batch Average Loss:", batch_loss, ", Average Dist:", batch_dist)

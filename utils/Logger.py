import os
import time


def set_logfile(file_name: str):
    file_dir = "./logs"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file = os.path.join(file_dir, file_name + ".txt")
    return file


def get_time():
    now = time.localtime()
    now_time = time.strftime("%H:%M:%S", now)
    return now_time


def log_hint(hint: str, file):
    fp = open(file, "a+")
    print(get_time(), hint, file=fp)
    fp.close()
    print(get_time(), hint)


def log_msg_head(epoch_num: int, batch_size: int, file):
    fp = open(file, "a+")
    print('epoch_num = {}'.format(epoch_num), file=fp)
    print('batch_size = {}'.format(batch_size), file=fp)
    print('|---------------Info----------------|------Train------|------Valid------|', file=fp)
    print('| time       epoch    iter   lr     | loss     dice   | loss     dice   |', file=fp)
    print('-------------------------------------------------------------------------', file=fp)
    fp.close()
    print('epoch_num = {}'.format(epoch_num))
    print('batch_size = {}'.format(batch_size))
    print('|---------------Info----------------|------Train------|------Valid------|')
    print('| time       epoch    iter   lr     | loss     dice   | loss     dice   |')
    print('-------------------------------------------------------------------------')


def log_msg(epoch: int, iteration: int, lr: float, train_accuracy: list, valid_accuracy: list, is_save: bool, file):
    fp = open(file, "w")
    if is_save:
        sign = '*'
    else:
        sign = ' '

    if train_accuracy[1] < 0.0001:
        train_accuracy[1] = 0.
    if valid_accuracy[1] and valid_accuracy[1] < 0.0001:
        valid_accuracy[1] = 0.

    train_accuracy = ['None' if elem is None else str(elem) for elem in train_accuracy]
    valid_accuracy = ['None' if elem is None else str(elem) for elem in valid_accuracy]

    print("| {:<8}   {:<6}   {:<4}   {:<6} | {:<6}   {:<6} | {:<6}   {:<6} |".format(get_time(), str(epoch)[:6],
                                                                                     (str(iteration) + sign)[:4],
                                                                                     str(lr)[:6],
                                                                                     str(train_accuracy[0])[:6],
                                                                                     str(train_accuracy[1])[:6],
                                                                                     str(valid_accuracy[0])[:6],
                                                                                     str(valid_accuracy[1])[:6]),
          file=fp)
    fp.close()
    print("| {:<8}   {:<6}   {:<4}   {:<6} | {:<6}   {:<6} | {:<6}   {:<6} |".format(get_time(), str(epoch)[:6],
                                                                                     (str(iteration) + sign)[:4],
                                                                                     str(lr)[:6],
                                                                                     str(train_accuracy[0])[:6],
                                                                                     str(train_accuracy[1])[:6],
                                                                                     str(valid_accuracy[0])[:6],
                                                                                     str(valid_accuracy[1])[:6]))


def log_epoch(iteration: int, batch_loss: float, batch_dist: float, file):
    fp = open(file, "a+")
    print(iteration, "Batch Average Loss:", batch_loss, ", Average Dist:", batch_dist, file=fp)
    fp.close()
    print(iteration, "Batch Average Loss:", batch_loss, ", Average Dist:", batch_dist)

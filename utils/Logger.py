import os
import time


def set_logfile(file_name: str):
    file_dir = "./logs/{}".format(file_name)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    file = os.path.join(file_dir, "log.txt")
    with open(file, 'a') as f:
        f.truncate(0)
    return file


def get_time():
    now = time.localtime()
    now_time = time.strftime("%H:%M:%S", now)
    return now_time


def log_hint(hint: str, file):
    with open(file, "a") as f:
        f.write(str(get_time()) + hint + '\n')
    print(get_time(), hint)


def log_msg_head(epoch_num: int, batch_size: int, file):
    with open(file, "a") as f:
        f.write('epoch_num = {}'.format(epoch_num) + '\n')
        f.write('batch_size = {}'.format(batch_size) + '\n')
        f.write('|---------------Info----------------|------Train------|------Valid------|' + '\n')
        f.write('| time       epoch    iter   lr     | loss     dice   | loss     dice   |' + '\n')
        f.write('-------------------------------------------------------------------------' + '\n')
    print('epoch_num = {}'.format(epoch_num))
    print('batch_size = {}'.format(batch_size))
    print('|---------------Info----------------|------Train------|------Valid------|')
    print('| time       epoch    iter   lr     | loss     dice   | loss     dice   |')
    print('-------------------------------------------------------------------------')


def log_msg(epoch: int, iteration: int, lr: float, train_accuracy: list, valid_accuracy: list, is_save: bool, file):
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
    with open(file, "a") as f:
        f.write("| {:<8}   {:<6}   {:<4}   {:<6} | {:<6}   {:<6} | {:<6}   {:<6} |".format(get_time(), str(epoch)[:6],
                                                                                           (str(iteration) + sign)[:4],
                                                                                           str(lr)[:6],
                                                                                           str(train_accuracy[0])[:6],
                                                                                           str(train_accuracy[1])[:6],
                                                                                           str(valid_accuracy[0])[:6],
                                                                                           str(valid_accuracy[1])[
                                                                                           :6]) + '\n')

    print("| {:<8}   {:<6}   {:<4}   {:<6} | {:<6}   {:<6} | {:<6}   {:<6} |".format(get_time(), str(epoch)[:6],
                                                                                     (str(iteration) + sign)[:4],
                                                                                     str(lr)[:6],
                                                                                     str(train_accuracy[0])[:6],
                                                                                     str(train_accuracy[1])[:6],
                                                                                     str(valid_accuracy[0])[:6],
                                                                                     str(valid_accuracy[1])[:6]))

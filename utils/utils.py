from datetime import datetime
import torch
#import copy
import random
import numpy as np
from slacker import Slacker
import os


def save_results(args, total):
    total = torch.Tensor(total)
    try:
        mean_loss = total.mean(dim=0)[:, 3].tolist()  # test loss
        mean_acc = total.mean(dim=0)[:, 4].tolist()  # test acc
        stdv_acc = total.std(dim=0)[:, 4].tolist()
    except:
        mean_loss = None
        # check # of selected label: total[:,0]
        mean_acc = total.mean(dim=0)[:,1].tolist()
        stdv_acc = total.std(dim=0)[:,1].tolist()

    now = datetime.now().isoformat().replace('T', '-').replace(':', '-')[:-7]
    file_name = args.model_name + '-' + str(args.data) + '-' + now

    try:
        del args.vis
        del args.plot_data
    except:
        pass

    with open('{}/{}-{}-{}.txt'.format(args.out_path,args.model_name,args.data,now), 'w') as f:
        f.write('START: {}\n'.format(args.start))
        f.write('END  : {}\n'.format(datetime.now()))
        f.write('SPEND: {}\n'.format(datetime.now()-args.start))
        del args.start
        f.write('OPTIONS: {}\n'.format(args))
        if mean_loss is not None:
            f.write('{} TRIALS MEAN Loss\n'.format(args.num_trial))
            for x in mean_loss:
                f.write('{:.4f},'.format(x))
            f.write('\n')
        f.write('{} TRIALS MEAN Acc\n'.format(args.num_trial))
        for x in mean_acc:
            f.write('{:.4f},'.format(x))
        f.write('\n{} TRIALS MEAN STDV\n'.format(args.num_trial))
        for x in stdv_acc:
            f.write('{:.8f},'.format(x))

        f.write('\n===============================\n')
        if total.size()[2] > 3:
            f.write('TRIAL,CYCLE,Labeled,TrainLoss,TrainAcc,TestLoss,TestAcc\n')
        else:
            f.write('TRIAL,CYCLE,Labeled,TestAcc\n')
        for t in range(len(total)):
            for c in range(len(total[0])):
                x = total[t][c]
                f.write('{},{},{}'.format(t, c, int(x[0])))
                for xx in x[1:]:
                    f.write(',{:.4f}'.format(xx))
                f.write('\n')
    return file_name


def train_test_split(indices, split_size=0.8):
    if split_size < 1:
        split_size = int(len(indices) * split_size)
    random.shuffle(indices)
    train = indices[:split_size]
    valid = indices[split_size:]
    return train, valid


def vis_plot(x, y, vis, plot_data, ymax=1, title='Acc over Time'):
    plot_data['X'].append(x)
    plot_data['Y'].append(y)
    vis.line(
        X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
        Y=np.array(plot_data['Y']),
        opts={
            'title': title,
            'legend': plot_data['legend'],
            'xlabel': 'Iterations',
            'ylabel': 'Acc',
            'width': 2000,
            'height': 500,
            'ytickmin': 0,
            'ytickmax': ymax,
        },
        win=1
    )


def alarm(msg, token_path='./utils/token.txt'):
    if os.path.isfile(token_path):
        with open(token_path, 'r') as f:
            token = f.read()
        slack = Slacker(token)
        slack.chat.post_message('#alarms', msg)
    else:
        print('not available alarm')
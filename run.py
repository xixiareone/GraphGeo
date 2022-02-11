# -*- coding: utf-8 -*-
from utils import *
import argparse
import numpy as np
import random

parser = argparse.ArgumentParser()
# parameters of initializing
parser.add_argument('--seed', type=int, default=1013, help='manual seed')
parser.add_argument('--model_name', type=str, default='GraphGeo')
parser.add_argument('--dataset', type=str, default='New_York', choices=["Shanghai", "New_York", "Los_Angeles"],
                    help='which dataset to use')

# parameters of training
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--harved_epoch', type=int, default=50)
parser.add_argument('--early_stop_epoch', type=int, default=100)
parser.add_argument('--dropout', type=float, default=0.1)

# parameters of model
parser.add_argument('--dim_in', type=int, default=30, choices=[51, 30], help="51 if Shanghai / 30 else")
parser.add_argument('--dim_as', type=int, default=64)
parser.add_argument('--dim_z', type=int, default=32)
parser.add_argument('--dim_inner', type=int, default=32)
parser.add_argument('--threshold', type=float, default=0.6)
parser.add_argument('--lambda_1', type=float, default=0.8)
parser.add_argument('--lambda_2', type=float, default=0.5)

opt = parser.parse_args()

if opt.seed is None:
    opt.seed = 1024
print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.set_printoptions(threshold=float('inf'))

warnings.filterwarnings('ignore')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("device:", device)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

train_data = np.load("./datasets/{}/Clustering_s1013_lm70_train.npz".format(opt.dataset),
                     allow_pickle=True)
test_data = np.load("./datasets/{}/Clustering_s1013_lm70_test.npz".format(opt.dataset),
                    allow_pickle=True)
train_data, test_data = train_data["data"], test_data["data"]
print("data readed.")


from model.model import *
model = GraphGeo(dim_in=opt.dim_in, dim_as=opt.dim_as, dim_z=opt.dim_z, dim_inner=opt.dim_inner,
                 dim_out=2, threshold=opt.threshold, lambda_1=opt.lambda_1, lambda_2=opt.lambda_2, dropout=opt.dropout)

model.apply(init_network_weights)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

if cuda:
    model.cuda()

# optimizer init
lr = opt.lr
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2))

if __name__ == '__main__':
    train_data, test_data = get_data_generator(opt, train_data, test_data)

    # train
    losses = [np.inf]
    no_better_epoch = 0
    early_stop_epoch = 0

    for epoch in range(2000):
        print("epoch {}.    ".format(epoch))
        beta = min([(epoch * 1.) / max([100, 1.]), 1.])
        total_loss, total_mae, train_num = 0, 0, 0
        model.train()
        for i in range(len(train_data)):
            lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, y_max, y_min = train_data[i]["lm_X"], train_data[i]["lm_Y"], \
                                                                       train_data[i][
                                                                           "tg_X"], train_data[i]["tg_Y"], \
                                                                       train_data[i]["lm_delay"], train_data[i][
                                                                           "tg_delay"], train_data[i]["y_max"], \
                                                                       train_data[i]["y_min"]

            optimizer.zero_grad()
            y_pred, reg = model(Tensor(lm_X), Tensor(lm_Y), Tensor(tg_X), Tensor(tg_Y), Tensor(lm_delay),
                                    Tensor(tg_delay))
            distance = t2_loss(Tensor(tg_Y), y_pred, y_max, y_min)
            rmse_loss = distance * distance  # mse loss
            rmse_loss = rmse_loss.sum()
            loss = rmse_loss + reg
            loss.backward()
            optimizer.step()

            total_loss += loss
            total_mae += distance.sum()
            train_num += len(tg_Y)

        total_loss = total_loss / train_num
        total_mae = total_mae / train_num

        print("train: loss: {:.4f} mae: {:.4f}".format(total_loss, total_mae))

        # test
        total_mse, total_mae, test_num = 0, 0, 0
        dislist = []

        model.eval()

        with torch.no_grad():
            for i in range(len(test_data)):
                lm_X, lm_Y, tg_X, tg_Y, lm_delay, tg_delay, y_max, y_min = test_data[i]["lm_X"], test_data[i]["lm_Y"], \
                                                                           test_data[i][
                                                                               "tg_X"], test_data[i]["tg_Y"], \
                                                                           test_data[i][
                                                                               "lm_delay"], test_data[i]["tg_delay"], \
                                                                           test_data[i]["y_max"], test_data[i]["y_min"]

                y_pred, kl_loss = model(Tensor(lm_X), Tensor(lm_Y), Tensor(tg_X), Tensor(tg_Y), Tensor(lm_delay),
                                        Tensor(tg_delay))
                distance = t2_loss(Tensor(tg_Y), y_pred, y_max, y_min)
                for i in range(len(distance.cpu().detach().numpy())):
                    dislist.append(distance.cpu().detach().numpy()[i])

                test_num += len(tg_Y)
                total_mse += (distance * distance).sum()
                total_mae += distance.sum()

            total_mse = total_mse / test_num
            total_mae = total_mae / test_num
            print("test: mse: {:.4f}  mae: {:.4f}".format(total_mse, total_mae))
            dislist_sorted = sorted(dislist)
            print('test median:', dislist_sorted[int(len(dislist_sorted) / 2)])

            batch_metric = total_mae.cpu().numpy()
            if batch_metric <= np.min(losses):
                no_better_epoch = 0
                early_stop_epoch = 0
                print("Better MAE in epoch {}: {:.4f}".format(epoch, batch_metric))
            else:
                no_better_epoch = no_better_epoch + 1
                early_stop_epoch = early_stop_epoch + 1

            losses.append(batch_metric)

            # halve the learning rate
            if no_better_epoch % opt.harved_epoch == 0 and no_better_epoch != 0:
                lr /= 2
                print("learning rate changes to {}!\n".format(lr))
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(opt.beta1, opt.beta2))
                no_better_epoch = 0

            if early_stop_epoch == opt.early_stop_epoch:
                break


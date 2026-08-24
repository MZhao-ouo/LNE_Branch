import os
import glob
import time
import torch
import torch.optim as optim
import numpy as np
import yaml
import pdb
import tqdm
import sklearn.cluster
# from kmeans_pytorch import kmeans

from tabular_model import *
from tabular_util import *

# set seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True

_, config = load_config_yaml('config.yaml')
config['device'] = torch.device('cpu')  # ('cuda:'+ config['gpu'])

def train():
    global_iter = 0
    monitor_metric_best = 100
    start_time = time.time()

    for epoch in range(start_epoch+1, config['epochs']):
        print('Epoch: ', epoch)

        # forward
        model.train()
        loss_all_dict = {'all': 0, 'recon': 0., 'dir': 0.}
        global_iter0 = global_iter
        for iter, sample in enumerate(trainDataLoader, 0):
            global_iter += 1
            tab1 = sample['tab1'].to(config['device'], dtype=torch.float).unsqueeze(1)
            tab2 = sample['tab2'].to(config['device'], dtype=torch.float).unsqueeze(1)
            age1 = sample['age1'].to(config['device'], dtype=torch.float)
            age2 = sample['age2'].to(config['device'], dtype=torch.float)
            interval = (age2 - age1).to(config['device'], dtype=torch.float)

            if tab1.shape[0] <= config['batch_size'] // 2:
                break

            # run model
            zs, recons = model(tab1, tab2)
            adj_mx = model.build_graph_batch(zs)    # no need to modify
            delta_z, delta_h = model.compute_social_pooling_delta_z_batch(zs, interval, adj_mx)

            # loss
            loss = 0
            if config['lambda_recon'] > 0:
                loss_recon = 0.5 * (model.compute_recon_loss(tab1, recons[0]) + model.compute_recon_loss(tab2, recons[1]))
                loss += config['lambda_recon'] * loss_recon
            else:
                loss_recon = torch.tensor(0.)

            if config['lambda_dir'] > 0:
                if config['model_name'] in ['LSP']:
                    loss_dir = model.compute_direction_loss(delta_z, delta_h)
                else:
                    loss_dir = model.compute_direction_loss(zs)
                loss += config['lambda_dir'] * loss_dir
            else:
                loss_dir = torch.tensor(0.)

            loss_all_dict['all'] += loss.item()
            loss_all_dict['recon'] += loss_recon.item()
            loss_all_dict['dir'] += loss_dir.item()

            loss.backward()
            for name, param in model.named_parameters():
                try:
                    if not torch.isfinite(param.grad).all():
                        print(name, param.grad)
                        pdb.set_trace()
                except:
                    continue

            optimizer.step()
            optimizer.zero_grad()

            if global_iter % 1 == 0:
                print('Epoch[%3d], iter[%3d]: loss=[%.4f], recon=[%.4f], dir=[%.4f]' \
                        % (epoch, iter, loss.item(), loss_recon.item(), loss_dir.item()))
                # writer.add_scalar("Loss/train", loss, global_iter)
                # writer.add_scalar("Loss/recon", loss_recon, global_iter)
                # writer.add_scalar("Loss/dir", loss_dir, global_iter)
                # writer.add_scalar("Loss/proto", loss_proto, global_iter)

            # if iter > 2:
            #     break

        # save train result
        num_iter = global_iter - global_iter0
        for key in loss_all_dict.keys():
            loss_all_dict[key] /= num_iter
        save_result_stat(loss_all_dict, config, info='epoch[%2d]'%(epoch))
        # writer.add_scalar("Loss/train_epoch", loss_all_dict['all'][0], epoch)
        # writer.flush()
        print(loss_all_dict)

        # validation
        stat = evaluate(phase='val', dataset='val', info='batch')
        monitor_metric = stat['all']
        scheduler.step(monitor_metric)
        save_result_stat(stat, config, info='val')
        print(stat)

        # save ckp
        is_best = False
        if monitor_metric <= monitor_metric_best:
            is_best = True
            monitor_metric_best = monitor_metric if is_best == True else monitor_metric_best
        state = {'epoch': epoch, 'monitor_metric': monitor_metric, 'stat': stat, \
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), \
                'model': model.state_dict()}
        print(optimizer.param_groups[0]['lr'])
        save_checkpoint(state, is_best, config['ckpt_path'])

def evaluate(phase='val', dataset='val', info='batch'):
    model.eval()
    if phase == 'val':
        loader = valDataLoader
    else:
        if dataset == 'train':
            loader = trainDataLoader
        elif dataset == 'val':
            loader = valDataLoader
        elif dataset == 'test':
            loader = testDataLoader
        else:
            raise ValueError('Undefined loader')

    if info == 'dataset':
        loader_train = trainDataLoader
        z1_train = []
        z2_train = []
        interval_train = []
        with torch.no_grad():
            for iter, sample in tqdm.tqdm(enumerate(loader_train, 0)):
                tab1 = sample['tab1'].to(config['device'], dtype=torch.float).unsqueeze(1)
                tab2 = sample['tab2'].to(config['device'], dtype=torch.float).unsqueeze(1)
                age1 = sample['age1'].to(config['device'], dtype=torch.float)
                age2 = sample['age2'].to(config['device'], dtype=torch.float)
                interval = (age2 - age1).to(config['device'], dtype=torch.float)
                interval_train.append(interval.cpu().numpy())

                # run model
                zs_batch, _ = model(tab1, tab2)     # zs_batch: [z1, z2]
                z1_train.append(zs_batch[0].cpu().numpy())
                z2_train.append(zs_batch[1].cpu().numpy())
        z1_train = np.concatenate(z1_train, axis=0)
        z2_train = np.concatenate(z2_train, axis=0)
        zs_train = [z1_train, z2_train]
        interval_train = torch.tensor(np.concatenate(interval_train, axis=0), dtype=torch.float)

    res_path = os.path.join(config['ckpt_path'], 'result_'+dataset)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    path = os.path.join(res_path, 'results_all'+info+'.h5')
    if os.path.exists(path):
        # raise ValueError('Exist results')
        os.remove(path)

    loss_all_dict = {'all': 0, 'recon': 0., 'dir': 0.}
    with torch.no_grad():
        for iter, sample in tqdm.tqdm(enumerate(loader, 0)):
            tab1 = sample['tab1'].to(config['device'], dtype=torch.float).unsqueeze(1)
            tab2 = sample['tab2'].to(config['device'], dtype=torch.float).unsqueeze(1)
            age1 = sample['age1'].to(config['device'], dtype=torch.float)
            age2 = sample['age2'].to(config['device'], dtype=torch.float)
            interval = (age2 - age1).to(config['device'], dtype=torch.float)

            # run model
            zs, recons = model(tab1, tab2)
            if info == 'dataset':
                adj_mx = model.build_graph_dataset(zs_train, zs)
                delta_z, delta_h = model.compute_social_pooling_delta_z_dataset(zs_train, interval_train, zs, interval, adj_mx)
            else:
                adj_mx = model.build_graph_batch(zs)
                delta_z, delta_h = model.compute_social_pooling_delta_z_batch(zs, interval, adj_mx)

            # loss
            loss = 0
            if config['lambda_recon'] > 0:
                loss_recon = 0.5 * (model.compute_recon_loss(tab1, recons[0]) + model.compute_recon_loss(tab2, recons[1]))
                loss += config['lambda_recon'] * loss_recon
            else:
                loss_recon = torch.tensor(0.)
            if config['lambda_dir'] > 0:
                if config['model_name'] in ['LSP']:
                    loss_dir = model.compute_direction_loss(delta_z, delta_h)
                else:
                    loss_dir = model.compute_direction_loss(zs)
                loss += config['lambda_dir'] * loss_dir
            else:
                loss_dir = torch.tensor(0.)

            loss_all_dict['all'] += loss.item()
            loss_all_dict['recon'] += loss_recon.item()
            loss_all_dict['dir'] += loss_dir.item()

        for key in loss_all_dict.keys():
            loss_all_dict[key] /= (iter + 1)


    return loss_all_dict


for trial in range(50):
    print('Start trial:', trial)

    ###############
    # Tensorboard #
    ###############
    # from torch.utils.tensorboard import SummaryWriter
    # writer = SummaryWriter()

    ##################
    # Train Settings #
    ##################
    config['trial'] = trial
    localtime = time.localtime(time.time())
    ckpt_label = f"trial{trial}_{localtime.tm_year}_{localtime.tm_mon}_{localtime.tm_mday}_{localtime.tm_hour}_{localtime.tm_min}"
    config['ckpt_path'] = os.path.join('./ckpt/', config['dataset_name'], config['model_name'], ckpt_label)
    os.makedirs(config['ckpt_path'])
    save_config_yaml(config['ckpt_path'], config)
    # define dataset
    Data = LongitudinalData(
        data_dir=f"./data/{config['dataset_name']}/splits/trial{trial}/",
        batch_size=config['batch_size'],
        shuffle=config['shuffle'],        # Only for training set
        )
    trainDataLoader = Data.trainLoader
    valDataLoader = Data.valLoader
    testDataLoader = Data.testLoader
    # define model
    model = LSP(num_neighbours=config['num_neighbours'],
                dims=config['dims'],
                agg_method=config['agg_method'],
                gpu=config['device'],
                activation="leakyrelu",
                dropout=0.0,
                slope=0.2,
                batch_norm=True).to(config['device'])

    print(model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-5)
    start_epoch = -1
    train()

    #################
    # Test Settings #
    #################
    flag, config_load = load_config_yaml(os.path.join(config['ckpt_path'], 'config.yaml'))
    if flag:    # load yaml success
        print('load yaml config file')
        for key in config_load.keys():  # if yaml has, use yaml's param, else use config
            if key == 'phase' or key == 'gpu' or key == 'continue_train' or key == 'ckpt_name':
                continue
            if key in config.keys():
                config[key] = config_load[key]
            else:
                print('current config do not have yaml param')
    else:
        save_config_yaml(config['ckpt_path'], config)
    [optimizer, scheduler, model], start_epoch = load_checkpoint_by_key([optimizer, scheduler, model], config['ckpt_path'], ['optimizer', 'scheduler', 'model'], config['device'], config['ckpt_name'])
    stat = evaluate(phase='test', dataset='test', info='dataset')
    print(stat)

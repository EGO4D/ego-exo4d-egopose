import math
import json
import torch
import wandb
import random
import logging
import os.path
import argparse
import numpy as np
from utils import utils_logger
from utils import utils_option as option
from torch.utils.data import DataLoader
from data.select_dataset import define_Dataset
from models.select_model import define_Model

def main(json_path='options/train_egoexo.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)
    wandb.init(project=opt['wandb_name'],config=opt, mode = opt['wandb_mode'])
    paths = (path for key, path in opt['path'].items() if 'pretrained' not in key)
    if isinstance(paths, str):
        if not os.path.exists(paths):
            os.makedirs(paths)
    else:
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    if init_path_G is not None:
        opt['path']['pretrained_netG'] = init_path_G
    current_step = init_iter

    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    logger_name = 'train'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)
    logger.info(option.dict2str(opt))
    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    '''
    # ----------------------------------------
    # Step--2 (create dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = define_Dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            train_loader = DataLoader(train_set,
                                      batch_size=dataset_opt['dataloader_batch_size'],
                                      shuffle=dataset_opt['dataloader_shuffle'],
                                      num_workers=dataset_opt['dataloader_num_workers'],
                                      drop_last=True,
                                      pin_memory=True
                                      )
        elif phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True
                                    )
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    
    model = define_Model(opt)

    if opt['merge_bn'] and current_step > opt['merge_bn_startpoint']:
        logger.info('^_^ -----merging bnorm----- ^_^')
        model.merge_bnorm_test()

    logger.info(model.info_network())
    model.init_train()
    logger.info(model.info_params())

    '''
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''

    test_step = 0

    for epoch in range(100000):  # keep running
        for i, train_data in enumerate(train_loader):

            current_step += 1
            # -------------------------------
            # 1) feed patch pairs
            # -------------------------------
            
            model.feed_data(train_data)
            
            # -------------------------------
            # 2) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 3) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)
            wandb_dict = model.log_dict
            wandb_dict['train_step']=current_step
            wandb.log(wandb_dict)

            # -------------------------------
            # merge bnorm
            # -------------------------------
            if opt['merge_bn'] and opt['merge_bn_startpoint'] == current_step:
                logger.info('^_^ -----merging bnorm----- ^_^')
                model.merge_bnorm_train()
                model.print_network()

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0:
                logger.info('Saving the model.')
                model.save(current_step)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0:
                pos_error = []
                vel_error = []
                tasks = []
                test_step+=1
                inference_dict = {}
                gt_dict = {}
                
                for index, test_data in enumerate(test_loader):

                    logger.info("testing the sample {}/{}".format(index, len(test_loader)))
                    model.feed_data(test_data, test=True)

                    model.test()

                    body_parms_pred = model.current_prediction()
                    body_parms_gt = model.current_gt()

                    predicted_position = body_parms_pred['position']
                    gt_position = body_parms_gt['position']

                    data = model.visible[0]*torch.sqrt(torch.sum(torch.square(gt_position-predicted_position),axis=-1))
                    pos_error_ = data.sum()/(data!=0).sum()

                    gt_velocity = (gt_position[1:,...] - gt_position[:-1,...])*10
                    predicted_velocity = (predicted_position[1:,...] - predicted_position[:-1,...])*10

                    data_vel = model.visible[0]*torch.mean(torch.sqrt(torch.sum(torch.square(gt_velocity-predicted_velocity),axis=-1)))
                    vel_error_  = data_vel.sum()/(data_vel!=0).sum()

                    if model.visible.max() !=0:
                        pos_error.append(pos_error_)
                        vel_error.append(vel_error_)
                        tasks.append(str(test_data['task'].numpy()[0])[0])

                    visible = model.visible.squeeze(0).unsqueeze(2).repeat(1,1,3)
                    visible[visible!=1]=torch.nan
                    gt_nan = visible*gt_position
                    t_ = np.array(test_data['t']).squeeze(1).tolist()
                    preds_ = dict(zip(t_,predicted_position.tolist()))
                    gt_ = dict(zip(t_,gt_nan.tolist()))
                    inference_dict[test_data['take_uid'][0]]={"take_name":test_data['take_name'][0],"body":preds_}
                    gt_dict[test_data['take_uid'][0]]={"take_name":test_data['take_name'][0],"body":gt_}


                activities ={'1':'Cooking','2':'Health','3':'Campsite','4':'Bike repair','5':'Music','6':'Basketball','7':'Bouldering','8':'Soccer','9':'Dance'}
                for task_num in range(0,10):
                    task_ids = [i for i, j in enumerate(tasks) if j == str(task_num)]
                    if len(task_ids)>0:
                        pos_ = torch.stack(pos_error)[task_ids].cpu().numpy()
                        vel_ = torch.stack(vel_error)[task_ids].cpu().numpy()
                        logger.info('<epoch:{:3d}, iter:{:8,d}, Task: {}, Samples: {}, MPJPE[cm]: {:<.5f}, MPJVE [m/s]: {:<.5f}\n'.format(epoch, current_step,activities[str(task_num)],len(task_ids), (pos_.mean())*100, (vel_.mean())))
                        

                
                pos_error = sum(pos_error)/len(pos_error)
                vel_error = sum(vel_error)/len(vel_error)
                wandb.log({'MPJPE':pos_error*100,'MPJVE':vel_error,'test_step':test_step})
                # testing log
                logger.info('<epoch:{:3d}, iter:{:8,d}, Average positional error [cm]: {:<.5f}, Average velocity error [m/s]: {:<.5f}\n'.format(epoch, current_step,pos_error*100, vel_error))


    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()

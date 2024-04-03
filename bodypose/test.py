import json
import torch
import os.path
import logging
import argparse
import numpy as np
from utils import utils_logger
from torch.utils.data import DataLoader
from utils import utils_option as option
from data.select_dataset import define_Dataset
from models.select_model import define_Model


def main(json_path='options/test_egoexo.json'):

    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    opt = option.parse(parser.parse_args().opt, is_train=True)

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
    logger_name = 'test'
    utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
    logger = logging.getLogger(logger_name)


    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for train and test
    # ----------------------------------------
    dataset_type = opt['datasets']['test']['dataset_type']
    for phase, dataset_opt in opt['datasets'].items():

        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                     shuffle=False, num_workers=0,
                                     drop_last=False, pin_memory=True)
        elif phase == 'train':
            continue
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

    model.init_test()
    pos_error = []
    vel_error = []
    tasks = []
    inference_dict = {}
    gt_dict = {}
    
    for index, test_data in enumerate(test_loader):

        logger.info("testing the sample {}/{}".format(index, len(test_loader)))
        model.feed_data(test_data)

        model.test()

        body_parms_pred = model.current_prediction()
        body_parms_gt = model.current_gt()
        predicted_position = body_parms_pred['position']
        gt_position = body_parms_gt['position']

        visible = model.visible.squeeze(0).unsqueeze(2).repeat(1,1,3)
        visible[visible!=1]=torch.nan
        gt_nan = visible*gt_position

        data = model.visible[0]*torch.sqrt(torch.sum(torch.square(gt_nan-predicted_position),axis=-1))
        pos_error_ = torch.nanmean(data)



        gt_velocity = (gt_position[1:,...] - gt_position[:-1,...])*10
        gt_nan_velocity = (gt_nan[1:,...] - gt_nan[:-1,...])*10
        predicted_velocity = (predicted_position[1:,...] - predicted_position[:-1,...])*10
        
        data_vel = torch.sqrt(torch.sum(torch.square(gt_nan_velocity-predicted_velocity),axis=-1))
        vel_error_ = torch.nanmean(data_vel)


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
    pred_path = os.path.join(opt['path']['images'],dataset_opt['split']+'_pred.json')
    gt_path = os.path.join(opt['path']['images'],dataset_opt['split']+'_gt.json')

    with open(pred_path, 'w') as fp:
        json.dump(inference_dict, fp)
    with open(gt_path, 'w') as fp:
        json.dump(gt_dict, fp)


    activities ={'1':'Cooking','2':'Health','3':'Campsite','4':'Bike repair','5':'Music','6':'Basketball','7':'Bouldering','8':'Soccer','9':'Dance'}
    for task_num in range(0,10):
        task_ids = [i for i, j in enumerate(tasks) if j == str(task_num)]
        if len(task_ids)>0:
            pos_ = torch.stack(pos_error)[task_ids].cpu().numpy()
            vel_ = torch.stack(vel_error)[task_ids].cpu().numpy()
            logger.info('Task: {}, Samples: {}, MPJPE[cm]: {:<.5f}, MPJVE [cm/s]: {:<.5f}\n'.format(activities[str(task_num)],len(task_ids), (pos_.mean())*100, (vel_.mean())))
            

    
    pos_error = sum(pos_error)/len(pos_error)
    vel_error = sum(vel_error)/len(vel_error)

    # testing log
    logger.info('Average positional error [cm]: {:<.5f}, Average velocity error [cm/s]: {:<.5f}\n'.format(pos_error*100, vel_error))



if __name__ == '__main__':
    main()

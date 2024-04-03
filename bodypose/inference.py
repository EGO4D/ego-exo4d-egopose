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


def main(json_path='options/inference_egoexo.json'):

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
        test_set = define_Dataset(dataset_opt)
        test_loader = DataLoader(test_set, batch_size=dataset_opt['dataloader_batch_size'],
                                    shuffle=False, num_workers=0,
                                    drop_last=False, pin_memory=True)

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
        model.feed_data(test_data,inference=True)

        model.test(inference=True)

        body_parms_pred = model.current_prediction()
        predicted_position = body_parms_pred['position']

        t_ = np.array(test_data['t']).squeeze(1).tolist()
        preds_ = dict(zip(t_,predicted_position.tolist()))
        inference_dict[test_data['take_uid'][0]]={"take_name":test_data['take_name'][0],"body":preds_}
    pred_path = os.path.join(opt['path']['images'],dataset_opt['split']+'_pred.json')

    with open(pred_path, 'w') as fp:
        json.dump(inference_dict, fp)
    logger.info("Done with inference")

if __name__ == '__main__':
    main()

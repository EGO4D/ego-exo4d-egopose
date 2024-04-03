def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()

    if dataset_type in ['egoexo']:
        from data.dataset_egoexo import Dataset_EgoExo as D
    elif dataset_type in ['egoexo_inference']:
        from data.dataset_egoexo import Dataset_EgoExo_inference as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset

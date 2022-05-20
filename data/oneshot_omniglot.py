import torch
from PIL import Image

# Parameters
nrun = 20 # number of classification runs
fname_label = 'class_labels.txt' # where class labels are stored for each run

class OneShotClfDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, transform):
        self.transform = transform
        self.supports = data_dict['supports']
        self.querys = data_dict['querys']
        self.labels = data_dict['labels']
        self.dataset_size = len(self.labels)
    
    def __getitem__(self, index):
        # support
        sup = self.supports[index]
        sup = [self.transform(img) for img in sup]
        sup = torch.stack(sup, dim=0)
        # query
        qry = self.querys[index]
        qry = [self.transform(img) for img in qry]
        qry = torch.stack(qry, dim=0)
        # label
        lbl = torch.stack(self.labels[index], dim=0)
        return sup, qry, lbl
    
    def __len__(self):
        return self.dataset_size
    
def get_dataloader(transform, batch_size, shuffle=False):
    data_dict_path = './data/omniglot/python/one-shot-classification/data-dict.pt'

    try:
        data_dict = torch.load(data_dict_path)
    except FileNotFoundError:
        make_os_omniglot_dataset(data_dict_path)
        data_dict = torch.load(data_dict_path)
    
    dataset = OneShotClfDataset(data_dict, transform)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                            batch_size=batch_size,
                                            shuffle=shuffle,)

    return dataloader

def make_os_omniglot_dataset(data_dict_path):
    '''make torch dataset from files
    '''
    source_data_path = './data/omniglot/python/one-shot-classification/'
    f_load = lambda path: Image.open(path, mode="r").convert("L")
    supports, querys, labels = [], [], []
    
    print('===> Processing raw one shot clf data')
    for r in range(1, 21):
        rs = str(r)
        if len(rs)==1: rs = '0' + rs
        r_path = source_data_path + f'run{rs}'
        
        # get file names
        with open(r_path+'/'+fname_label) as f:
            content = f.read().splitlines()
        pairs = [line.split() for line in content]
        test_files = [pair[0] for pair in pairs]
        train_files = [pair[1] for pair in pairs]

        # answers_files = copy.copy(train_files)
        test_files.sort()
        train_files.sort()	
        # ntrain = len(train_files)
        # ntest = len(test_files)

        # load the images (and, if needed, extract features)
        train_items = [f_load(source_data_path + f) for f in train_files]
        test_items  = [f_load(source_data_path + f) for f in test_files]
        label = [torch.tensor([
                    int(pair[0].split('/')[2][4:6]), # test
                    int(pair[1].split('/')[2][5:7]), # class 
                ]) for pair in pairs]
        
        supports.append(train_items)
        querys.append(test_items)
        labels.append(label)
    
    # organize and save
    # supports = torch.concat(supports, dim=0)
    # querys = torch.concat(querys, dim=0)
    # labels = torch.concat(labels, dim=0)

    data_dict = {
        'supports': supports,
        'querys': querys,
        'labels': labels
    } 

    torch.save(data_dict, data_dict_path)
import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()



    def initialize(self, opt):
        pass

    def __len__(self):
        return 0

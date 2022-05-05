import torch

from models import ssp
from models.ssp import DecoderParam
class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic dataset based on the generative model
    """
    def __init__(self, dataset, return_latent:bool=True):
        self.imgs = dataset['imgs']
        print(f"===> dataset size: {self.imgs.shape}")
        # self.zs = dataset['zs']
    
    def __getitem__(self, index: int):
        # if self.return_latent:
        #     return self.imgs[index], self.zs[index]
        # else:
        return self.imgs[index], torch.tensor(0)

    def __len__(self):
        return self.imgs.shape[0]

    
def make_dataset(args, dataset_size, dataset_path):
    print("===> Start making synthetic dataset")
    # init generative model
    it_per_sample = 100
    gen = ssp.GenerativeModel(max_strks=int(args.strokes_per_img), 
                                pts_per_strk=args.points_per_stroke, 
                                res=args.img_res, 
                                z_where_type=args.z_where_type,
                                use_canvas=False, 
                                transform_z_what=True,
                                input_dependent_param=True,
                                prior_dist='Independent',
                                maxnorm=True,
                                sgl_strk_tanh=True,
                                add_strk_tanh=True,
                                fixed_prior=True,
                                spline_decoder=True,
                                render_method=args.render_method,
                                dependent_prior=False,
                                linear_sum=True,
                                generate_data=True,
                            )
    # Make and save the dataset
    all_xs, all_zs = [], []
    num_data = 0
    while num_data < dataset_size:
        n_strokes = torch.randint(low=1,high=6,size=()
                                  ).item()
        bs = [1, it_per_sample, n_strokes]

        # Sample render param
        gen.sigma = 0.026 + 0.0001 * torch.randn(bs)
        gen.sgl_strk_tanh_slope = 0.7 + 0.0001 * torch.randn(bs)
        gen.add_strk_tanh_slope = 0.46 + 0.0001 * torch.randn(bs[:2])

        # Sample obs, latent
        _, img = gen.sample(bs=bs)
        all_xs.append(img)
        # all_zs.append(zs)
        num_data += it_per_sample

        if num_data % 5000 == 0:
            print(f"Have generatived {num_data}/{dataset_size}")

    # Aggregate data
    imgs = torch.cat(all_xs,1).squeeze(0)
    # all_zs ...
    dataset = {"imgs": imgs}
    torch.save(dataset, dataset_path)
    print("===> Done making synthetic dataset")
        
def get_data_loader(args, make_new_dataset=False):
    # Try load from save, if not exist then make one
    datasets = []
    for train in [True, False]:
        name = "train" if train else "test"
        dataset_path = f"./data/synthetic_dataset/{name}.pt"
        try:
            if make_new_dataset:
                raise FileNotFoundError
            data_dict = torch.load(dataset_path)
        except FileNotFoundError:
            dataset_size = 60000 if train else 10000
            make_dataset(args, dataset_size, dataset_path)
            data_dict = torch.load(dataset_path)
        dataset = SyntheticDataset(data_dict)
        datasets.append(dataset)

    return datasets
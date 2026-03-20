# dataset.py  ---------------------------------------------------------------
import torch, pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import paths


class MyDataset(Dataset):

    def __init__(self, data_dict, length):
        self.data, self.length = data_dict, length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


class Instance:
    """
    Read syn_full.csv and produce tensors for the Simulator.
    Split is determined by train.json / val.json / test.json SKU lists.
    """

    def __init__(self, problem_params, dataset_params, state_params):
        self.problem_params = problem_params
        self.dataset_params = dataset_params
        self.state_params = state_params
        self.max_T = problem_params['max_period_num']
        self.max_L = problem_params['max_lead_time']

        df = pd.read_csv(dataset_params['file_location'])

        train_skus = paths.load_json(paths.TRAIN_SKUS_FILE, 'train.json')
        val_skus = paths.load_json(paths.VAL_SKUS_FILE, 'val.json')
        test_skus = paths.load_json(paths.TEST_SKUS_FILE, 'test.json')

        # Build index mapping: filter df to only include SKUs in the splits
        all_split_skus = train_skus + val_skus + test_skus
        df = df[df['sku_id'].isin(all_split_skus)].reset_index(drop=True)
        sku_to_idx = {int(row['sku_id']): i for i, row in df.iterrows()}

        self.train_indices = [sku_to_idx[s] for s in train_skus if s in sku_to_idx]
        self.val_indices = [sku_to_idx[s] for s in val_skus if s in sku_to_idx]
        self.test_indices = [sku_to_idx[s] for s in test_skus if s in sku_to_idx]

        self.n_train = len(self.train_indices)
        self.n_val = len(self.val_indices)
        self.n_test = len(self.test_indices)
        self.N = len(df)
        print(f'train/val/test numbers: {self.n_train}/{self.n_val}/{self.n_test} (total {self.N})')

        # Build tensors from all rows
        self.review_period = self.get_col(df, dataset_params['review_period_col'][0]).long()
        self.period_num = self.get_col(df, dataset_params['period_num_col'][0]).long()
        self.holding_cost = torch.full((self.N, 1), float(problem_params['holding_cost']))
        self.backlog_cost = torch.full((self.N, 1), float(problem_params['backlog_cost']))

        # Sequence
        self.demands = self.seq2tensor(df, dataset_params['demand_seq_col'][0], self.max_T).float()
        self.lead_times = self.seq2tensor(df, dataset_params['lead_time_col'][0], self.max_T).long()
        self.features = self.featseq2tensor_normalized(df, dataset_params['features_col'][0], self.max_T).float()
        self.features_org = self.featseq2tensor(df, dataset_params['features_col'][0], self.max_T).float()
        self.reg_input = self.featseq2tensor(df, dataset_params['reg_input_col'][0], self.max_T).float()

        # Initial inventory pipeline
        stock0 = self.seq2tensor(df, dataset_params['initial_inventory_col'][0], self.max_T)[:, :, 0]
        # Need max lead-time (defined in global.yaml) to be nonzero
        pipeline_zeros = torch.zeros(self.N, self.max_L)
        self.initial_inv_pipeline = torch.cat([stock0, pipeline_zeros], dim=1).float()

        # ignore_period_num = max lead time + review period (skip days before first order arrives)
        self.ignore_period_num = (torch.max(self.lead_times, dim=2)[0] + self.review_period).long()

    def get_col(self, df, col):
        return torch.tensor(df[col].values).unsqueeze(1).float()

    def seq2tensor(self, df, col, T):
        seqs = df[col].apply(lambda s: list(map(float, s.split(","))))
        return torch.stack([F.pad(torch.tensor(s), (0, T - len(s))) for s in seqs]).unsqueeze(1)

    def featseq2tensor(self, df, col, T):
        feats = df[col].apply(lambda cell: [list(map(float, f.split(','))) for f in cell.split(';')])
        return torch.stack([
            torch.stack([F.pad(torch.tensor(f), (0, T - len(f))) for f in feat_list])
            for feat_list in feats
        ])

    def featseq2tensor_normalized(self, df, col, T):
        # normalize feature sequence over time
        features = torch.tensor(
            df[col].apply(lambda cell: [list(map(float, f.split(','))) for f in cell.split(';')]))
        min_vals = features.amin(dim=2, keepdim=True)
        max_vals = features.amax(dim=2, keepdim=True)
        feats_norm = (features - min_vals) / (max_vals - min_vals + 1e-8)

        return torch.stack([
            torch.stack([F.pad(f.clone(), (0, T - len(f))) for f in feat_list])
            for feat_list in feats_norm
        ])

    def create_data_dict(self):
        return {
            'initial_inventory_pipeline': self.initial_inv_pipeline,
            'demands': self.demands,
            'features': self.features,
            'features_org': self.features_org,
            'lead_times': self.lead_times,
            'review_period': self.review_period,
            'holding_cost': self.holding_cost,
            'backlog_cost': self.backlog_cost,
            'period_num': self.period_num,
            'ignore_period_num': self.ignore_period_num,
            'reg_input': self.reg_input,
        }


# ------------------------------------------------------------
def build_dataloaders(cfg, device="cpu", seed=None, batch_size=32):
    """
    Build train/val/test DataLoaders using JSON SKU splits.
    """
    inst = Instance(cfg.PROBLEM_PARAMS,
                    cfg.DATASET_PARAMS,
                    cfg.STATE_PARAMS)

    full = inst.create_data_dict()

    def subset(indices):
        return {k: v[indices].to(device) for k, v in full.items()}

    datasets = {
        'train': MyDataset(subset(inst.train_indices), inst.n_train),
        'val': MyDataset(subset(inst.val_indices), inst.n_val),
        'test': MyDataset(subset(inst.test_indices), inst.n_test),
    }

    generator = torch.Generator()
    actual_seed = seed if seed is not None else 42
    generator.manual_seed(actual_seed)

    loaders = {
        phase: DataLoader(ds,
                          batch_size=batch_size,
                          shuffle=(phase == 'train'),
                          generator=generator if phase == 'train' else None)
        for phase, ds in datasets.items()
    }
    return loaders

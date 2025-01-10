import json
from collections import defaultdict

from textDiff.data_utils import SimSEDataLoader


if __name__ == '__main__':
    data_output_dir = 'data/simse_treatment/'

    dataloader = SimSEDataLoader()
    full_datasets = dataloader.full_datasets
    split_datasets = dataloader.split_datasets

    for split in ['train', 'dev', 'test']:
        print(f"Split: {split}")
        out_fn = f"{data_output_dir}/simse_treatment_{split}.json"
        data = defaultdict(list)
        for ele in split_datasets[split]:
            data['class_transcript'].append(ele['text'])
            data['label'].append(ele['condition'])

        with open(out_fn, 'w') as f:
            json.dump(data, f)

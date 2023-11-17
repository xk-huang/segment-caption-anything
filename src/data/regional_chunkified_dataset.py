# Deprecated, Not compatible with HuggingFace Datasets
from torch.utils.data import Dataset, IterableDataset
from datasets import HFDataset, HFIterableDataset
from math import ceil


def prepare_regional_chunkified_dataset(dataset: Dataset, regional_chunk_size: int) -> Dataset:
    if not isinstance(dataset, (HFDataset, Dataset)):
        raise ValueError(f"Currently dataset must be a map-style Dataset, got {type(dataset)}")


class RegionalChunkifiedDataset(Dataset):
    def __init__(self, dataset: Dataset, regional_chunk_size: int):
        self.dataset = dataset
        self.regional_chunk_size = regional_chunk_size
        self.get_num_regions_and_num_samples_after_regional_chunkify()

    def get_num_regions_and_num_samples_after_regional_chunkify(self):
        self.num_regions = 0
        self.num_samples_after_regional_chunkify = 0
        self.chunkified_sample_ids_to_sample_idx = {}
        for sample_idx, example in enumerate(self.dataset):
            for chunkified_sample_idx in range(ceil(len(example["regions"]) / self.regional_chunk_size)):
                self.chunkified_sample_ids_to_sample_idx[
                    self.num_samples_after_regional_chunkify + chunkified_sample_idx
                ] = (sample_idx, chunkified_sample_idx)
            self.num_regions += len(example["regions"])
            self.num_samples_after_regional_chunkify += ceil(len(example["regions"]) / self.regional_chunk_size)

    def __len__(self):
        return self.num_samples_after_regional_chunkify

    def __getitem__(self, idx):
        sample_idx, chunkified_sample_idx = self.chunkified_sample_ids_to_sample_idx[idx]
        example = self.dataset[sample_idx]
        regions = example.pop("regions")

        return {
            **example,
            "regions": regions[
                chunkified_sample_idx
                * self.regional_chunk_size : (chunkified_sample_idx + 1)
                * self.regional_chunk_size
            ],
        }

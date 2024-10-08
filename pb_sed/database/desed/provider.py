from pb_sed.data_preparation.provider import DataProvider
import dataclasses
from pb_sed.paths import database_jsons_dir

def get_non_empty_datasets(data_provider):
    return list(filter(
        lambda key: data_provider.train_set[key] > 0,
        data_provider.train_set.keys()
    ))
@dataclasses.dataclass
class DESEDProvider(DataProvider):

    def __post_init__(self):
        super().__post_init__()
        self.train_transform.label_encoder.initialize_labels(
            dataset=self.db.get_dataset(get_non_empty_datasets(self)
            ),
            verbose=True
        )
        self.test_transform.label_encoder.initialize_labels()

    @classmethod
    def finalize_dogmatic_config(cls, config):
        config['json_path'] = str(database_jsons_dir / 'desed.json')
        config['validate_set'] = 'validation'
        super().finalize_dogmatic_config(config)
        num_events = 10
        config['train_fetcher']['min_label_diversity_in_batch'] = min(
            num_events, config['train_fetcher']['batch_size']
        )
        min_dataset_examples_in_batch = config['train_fetcher']['min_dataset_examples_in_batch']
        if min_dataset_examples_in_batch:
            dataset_lengths = {
                'train_weak': 1578,
                'train_unlabel_in_domain': 14412,
                'train_synthetic20': 2576,
                'train_synthetic21': 10000,
                'train_strong': 3470,
            }
            dataset_lengths = {
                key: config['train_set'].get(key, 0) * dataset_lengths[key]
                for key in dataset_lengths
            }
            total_dataset_length = sum(dataset_lengths.values())
            batch_size = config['train_fetcher']['batch_size']
            for key in min_dataset_examples_in_batch.keys():
                assert (
                    (min_dataset_examples_in_batch[key] / batch_size)
                    <= (dataset_lengths[key] / total_dataset_length)
                ), (batch_size, min_dataset_examples_in_batch, dataset_lengths)

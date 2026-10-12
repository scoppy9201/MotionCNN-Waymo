from pathlib import Path

import yaml


def create_tf_dataset(datapath, n_shards, shard_id):
    """Create a TFRecord dataset from all files in a directory."""
    source_dir = Path(datapath)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"TFRecord directory does not exist: {source_dir}")

    files = sorted(str(path) for path in source_dir.iterdir() if path.is_file())
    if not files:
        raise ValueError(f"No TFRecord files found in {source_dir}")

    import tensorflow as tf

    dataset = tf.data.TFRecordDataset(
        files, num_parallel_reads=1
    )
    if n_shards > 1:
        dataset = dataset.shard(n_shards, shard_id)
    return dataset


def get_config(path):
    """Load a YAML mapping using the safe loader."""
    with open(path, "r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return data

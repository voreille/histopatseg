import logging

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

logger = logging.getLogger(__name__)


def split_data_per_image(metadata_per_image, random_state=42):
    """Split the data into train and validation sets."""
    image_ids = metadata_per_image.index.values
    labels = metadata_per_image["class_name"].values
    patient_ids = metadata_per_image["patient_id"].values

    test_splitter = GroupShuffleSplit(n_splits=1,
                                      test_size=0.2,
                                      random_state=random_state)

    train_val_idx, test_idx = next(
        test_splitter.split(image_ids, labels, groups=patient_ids))

    train_val_image_ids = image_ids[train_val_idx]

    test_image_ids = image_ids[test_idx]

    train_val_labels = metadata_per_image.loc[train_val_image_ids,
                                              "class_name"].values
    train_val_patient_ids = metadata_per_image.loc[train_val_image_ids,
                                                   "patient_id"].values

    splitter = StratifiedGroupKFold(n_splits=5,
                                    shuffle=True,
                                    random_state=random_state)
    train_idx, val_idx = next(
        splitter.split(train_val_image_ids,
                       train_val_labels,
                       groups=train_val_patient_ids))
    train_image_ids = train_val_image_ids[train_idx]
    val_image_ids = train_val_image_ids[val_idx]

    train_patient_ids = metadata_per_image.loc[train_image_ids,
                                               "patient_id"].values
    val_patient_ids = metadata_per_image.loc[val_image_ids,
                                             "patient_id"].values
    test_patient_ids = metadata_per_image.loc[test_image_ids,
                                              "patient_id"].values

    are_splits_disjoint = set(train_patient_ids).isdisjoint(
        test_patient_ids) and set(train_patient_ids).isdisjoint(
            val_patient_ids) and set(val_patient_ids).isdisjoint(
                test_patient_ids)

    if not are_splits_disjoint:
        raise ValueError("Train, validation, and test sets have no overlap.")

    return train_image_ids, val_image_ids, test_image_ids


def split_data(tile_ids, metadata, random_state=42):
    """Split the data into train and validation sets."""
    tile_ids = np.array(tile_ids)
    initial_idx = np.arange(len(tile_ids))
    test_splitter = GroupShuffleSplit(n_splits=1,
                                      test_size=0.2,
                                      random_state=random_state)
    patient_ids = metadata.loc[tile_ids, "patient_id"].values
    labels = metadata.loc[tile_ids, "class_name"].values

    train_val_idx, test_idx = next(
        test_splitter.split(tile_ids, labels, groups=patient_ids))

    train_val_tile_ids = tile_ids[train_val_idx]
    train_val_idx = initial_idx[train_val_idx]

    test_tile_ids = tile_ids[test_idx]

    train_val_labels = metadata.loc[train_val_tile_ids, "class_name"].values
    train_val_patient_ids = metadata.loc[train_val_tile_ids,
                                         "patient_id"].values

    splitter = StratifiedGroupKFold(n_splits=5,
                                    shuffle=True,
                                    random_state=random_state)
    train_idx, val_idx = next(
        splitter.split(train_val_tile_ids,
                       train_val_labels,
                       groups=train_val_patient_ids))
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    train_tile_ids = tile_ids[train_idx]
    val_tile_ids = tile_ids[val_idx]

    train_patient_ids = metadata.loc[train_tile_ids, "patient_id"].values
    val_patient_ids = metadata.loc[val_tile_ids, "patient_id"].values
    test_patient_ids = metadata.loc[test_tile_ids, "patient_id"].values

    are_splits_disjoint = set(train_patient_ids).isdisjoint(
        test_patient_ids) and set(train_patient_ids).isdisjoint(
            val_patient_ids) and set(val_patient_ids).isdisjoint(
                test_patient_ids)

    if not are_splits_disjoint:
        raise ValueError("Train, validation, and test sets have no overlap.")

    logger.info(f"Train size: {len(train_tile_ids)}")
    logger.info(f"Validation size: {len(val_tile_ids)}")
    logger.info(f"Test size: {len(test_tile_ids)}")

    return train_idx, val_idx, test_idx


def split_data_old(embeddings, labels, patient_ids, random_state=42):
    """Split the data into train and validation sets."""
    test_splitter = GroupShuffleSplit(n_splits=1,
                                      test_size=0.2,
                                      random_state=random_state)
    train_val_idx, test_idx = next(
        test_splitter.split(embeddings, labels, groups=patient_ids))

    train_val_embeddings = embeddings[train_val_idx]
    train_val_labels = labels[train_val_idx]
    train_val_patient_ids = patient_ids[train_val_idx]

    test_embeddings = embeddings[test_idx]
    test_labels = labels[test_idx]
    test_patient_ids = patient_ids[test_idx]

    splitter = StratifiedGroupKFold(n_splits=5,
                                    shuffle=True,
                                    random_state=random_state)
    train_idx, val_idx = next(
        splitter.split(train_val_embeddings,
                       train_val_labels,
                       groups=train_val_patient_ids))

    train_embeddings = train_val_embeddings[train_idx]
    train_labels = train_val_labels[train_idx]
    train_patient_ids = train_val_patient_ids[train_idx]

    val_embeddings = train_val_embeddings[val_idx]
    val_labels = train_val_labels[val_idx]
    val_patient_ids = train_val_patient_ids[val_idx]

    are_splits_disjoint = set(train_patient_ids).isdisjoint(
        test_patient_ids) and set(train_patient_ids).isdisjoint(
            val_patient_ids) and set(val_patient_ids).isdisjoint(
                test_patient_ids)

    if not are_splits_disjoint:
        raise ValueError("Train, validation, and test sets have no overlap.")

    logger.info(f"Train size: {len(train_embeddings)}")
    logger.info(f"Validation size: {len(val_embeddings)}")
    logger.info(f"Test size: {len(test_embeddings)}")

    return (train_embeddings, train_labels, train_patient_ids, val_embeddings,
            val_labels, val_patient_ids, test_embeddings, test_labels,
            test_patient_ids)

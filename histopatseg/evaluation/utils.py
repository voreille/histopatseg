import pandas as pd


def aggregate_tile_embeddings(embeddings, tile_ids, metadata, group_by="original_filename"):
    """
    Aggregate tile-level embeddings into image-level embeddings by averaging.

    Parameters:
    -----------
    embeddings : numpy.ndarray
        Array of embeddings with shape (n_tiles, embedding_dim)
    tile_ids : list or numpy.ndarray
        List of tile IDs corresponding to the embeddings
    metadata : pandas.DataFrame
        DataFrame containing metadata for the tiles, indexed by tile_id
    group_by : str, default='original_filename'
        Column in metadata to group by (typically 'original_filename')

    Returns:
    --------
    tuple: (aggregated_embeddings, aggregated_metadata)
        - aggregated_embeddings: numpy.ndarray with shape (n_images, embedding_dim)
        - aggregated_metadata: pandas.DataFrame with image-level metadata
    """
    # Create a DataFrame with embeddings and tile_ids
    embedding_df = pd.DataFrame(
        embeddings,  # The embedding values
        index=tile_ids,  # Use tile_ids as the index
    )

    # Merge with metadata (metadata should be indexed by tile_id)
    aligned_metadata = metadata.reindex(tile_ids)
    merged_df = embedding_df.join(aligned_metadata)

    # Verify the group_by column exists
    if group_by not in merged_df.columns:
        raise ValueError(f"Column '{group_by}' not found in metadata. Cannot group embeddings.")

    # Identify metadata columns (non-numeric columns after the embeddings)
    embedding_cols = embedding_df.columns
    metadata_cols = [col for col in merged_df.columns if col not in embedding_cols]

    # Build aggregation dictionary
    agg_dict = {
        # Average all embedding columns
        **{col: "mean" for col in embedding_cols},
        # For metadata columns: take first value for categorical, count for one column
        **{col: "first" for col in metadata_cols},
    }

    # Use the last metadata column for counting tiles
    count_col = metadata_cols[-1] if metadata_cols else None
    if count_col:
        agg_dict[count_col] = "count"

    # Group by the specified column and calculate aggregations
    aggregated_df = merged_df.groupby(group_by).agg(agg_dict)

    # Rename the count column
    if count_col:
        aggregated_df.rename(columns={count_col: "tile_count"}, inplace=True)

    # Extract aggregated embeddings and metadata
    aggregated_embeddings = aggregated_df[embedding_cols].values
    aggregated_metadata = aggregated_df.drop(columns=embedding_cols)

    print(
        f"Aggregated {len(embedding_df)} individual tile embeddings into {len(aggregated_df)} {group_by}-level embeddings"
    )

    return aggregated_embeddings, aggregated_metadata

"""Operations on pandas data frame."""


def split_features(df, groups):
    """Split the columns of a df according to their given group.

    Parameters:
    -----------
    df : pandas.DataFrame
        The data frame to be splitted.
    groups : pandas.Series
        Series with the features' names or indexs as index and the group as
        values.

    Returns:
    --------
    dict
        Dictionnary with groups as keys and sub data frame as values.

    """
    sub_df = dict()

    for group_id in groups.unique():
        # Get the names of the features to drop
        features_to_drop = groups[groups != group_id].index
        # Get the data frame without htose features
        sub_df[group_id] = df.drop(features_to_drop, 1)

    return sub_df


if __name__ == '__main__':
    from database import TB
    print(split_features(TB['20000'], TB.features_types['20000']))

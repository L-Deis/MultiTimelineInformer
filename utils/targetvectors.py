import pandas as pd
import numpy as np


def generate_antibiotics_vector(df_antibiotics,
                                df_id,
                                gap_minutes=24 * 60,
                                margin_minutes=30,
                                freq='1min'):
    """
    Efficiently generates a binary time series indicating whether a patient was under antibiotic
    treatment at each minute of their stay.

    Parameters
    ----------
    df_antibiotics : pd.DataFrame
        DataFrame containing antibiotic administration events.
        Must contain the following columns:
            - 'stay_id': Unique identifier for the stay.
            - 'administration_time': Timestamp of each antibiotic administration.

    df_id : pd.DataFrame
        DataFrame defining the full time range of each stay.
        Must contain:
            - 'stay_id': Unique identifier for the stay.
            - 'date': Timestamp entries indicating the range of stay.
              These are used to determine the min and max datetime per stay.

    gap_minutes : int, default=1440 (24*60)
        Maximum number of minutes allowed between two consecutive administrations
        to consider them part of the same antibiotic treatment.

    margin_minutes : int, default=30
        Number of minutes before the first administration in a treatment to include
        in the antibiotics window.

    freq : str, default='1min'
        Frequency of the output time vector. Must be a valid Pandas offset alias
        like '1min', '5min', etc.

    Returns
    -------
    pd.DataFrame
        A concatenated DataFrame with the following columns:
            - 'stay_id': hospital stay identifier
            - 'date': Timestamp at the given frequency
            - 'antibiotics': 1 if antibiotics are considered active at that time, 0 otherwise

    Notes
    -----
    - This function assumes administration times are within the bounds of stay times.
    - TODO The output is fully expanded (non-sparse); maybe use `pd.SparseDtype` for long time ranges.
    - TODO check thaty the time vector is based on stay_id

    """

    all_results = []

    # for stay_id, stay_antibios_df in df_antibiotics.groupby('stay_id'):
    for stay_id, stay_ids_df in df_id.groupby('stay_id'):
        stay_antibios_df = df_antibiotics[df_antibiotics['stay_id'] == stay_id]

        stay_start = stay_ids_df['date'].min()
        stay_end = stay_ids_df['date'].max()

        full_time_range = pd.date_range(start=stay_start, end=stay_end, freq=freq)
        n = len(full_time_range)

        # Preallocate numpy array for antibiotic flags
        antibiotics_vector = np.zeros(n, dtype=np.uint8)

        # Build index mapping for fast lookup
        time_to_index = pd.Series(index=full_time_range, data=np.arange(n))

        # Prepare and clean up administration times
        stay_antibios_df = stay_antibios_df.copy()
        stay_antibios_df['administration_time'] = pd.to_datetime(stay_antibios_df['administration_time'])
        stay_antibios_df = stay_antibios_df.sort_values(by='administration_time')
        stay_antibios_df['administration_time'] = stay_antibios_df['administration_time'].dt.floor(freq)

        # early exit for empty vectors
        if stay_antibios_df.empty:
            result = pd.DataFrame({
                'stay_id': stay_id,
                'date': full_time_range,
                'antibiotics': antibiotics_vector
            })
            all_results.append(result)
            continue

        # Group administrations into treatments based on time gaps
        time_diffs = stay_antibios_df['administration_time'].diff().dt.total_seconds().div(60)
        group_ids = (time_diffs > gap_minutes).cumsum()

        for _, group in stay_antibios_df.groupby(group_ids):
            start_time = group['administration_time'].iloc[0] - pd.Timedelta(minutes=margin_minutes)
            end_time = group['administration_time'].iloc[-1]

            # Clip to stay window
            start_time = max(start_time, stay_start)
            end_time = min(end_time, stay_end)

            # Get index range using closest matches
            try:
                start_idx = time_to_index.get(start_time)
                end_idx = time_to_index.get(end_time)
                antibiotics_vector[start_idx:end_idx + 1] = 1
            except KeyError:
                print("There was an antibiotics administration outside of a known stay. It has been skipped.")
                # TODO check if this only happens where expected
                #  (discarded stays or after the patient has gone home for example)
                continue

        result = pd.DataFrame({
            'stay_id': stay_id,
            'date': full_time_range,
            'antibiotics': antibiotics_vector
        })
        all_results.append(result)

    if not all_results:
        # return empty df in case of empty inputs
        return pd.DataFrame(columns=['stay_id', 'date', 'antibiotics'])

    return pd.concat(all_results).sort_values(by=['stay_id', 'date'])['antibiotics'].values


def generate_antibiotics_vector_old(df_antibiotics,
                                    df_id,
                                    df_mappings,
                                    border1,
                                    border2,
                                    gap_minutes=24 * 60,
                                    margin_minutes=30,
                                    freq='1min'):
    """"
    The antibiotics vector needs to be a vector of 1s and 0s, 1 if the patient received antibiotics at that time, 0 otherwise,
    the time used needs to match the one from the same stay, i.e. same range (start to finish) of time as the patient stay
    The vector of antibiotics is filled with 1s in between two antibiotics administrations first and last time,
    except if the time between two administrations is more than 24 hours, then it is considered two different administrations
    """
    # Create new df that matches the range, for each stay_id, create a full time range from first administration to last administration, per stay_id
    full_range_dfs = []
    for stay_id, stay_antibios_df in df_antibiotics.groupby('stay_id'):
        # Find matching vital date start and end for corresponding stay_id
        staymatch_df_id = df_id[df_id['stay_id'] == stay_id]
        if staymatch_df_id.empty:
            # Skip this stay_id as there's no matching stay info
            continue

        stay_start = staymatch_df_id['date'].min()
        stay_end = staymatch_df_id['date'].max()

        # Generate a full time index from min to max timestamp
        full_time_range = pd.date_range(
            start=stay_start,
            end=stay_end,
            freq='1min'  # 1 minute frequency #TODO: Make frequency a parameter
        )

        # Create a new dataframe with the full time range
        administration_complete_df = pd.DataFrame({"date": full_time_range, "antibiotics": np.nan})
        administration_complete_df['stay_id'] = stay_id

        # In stay_antibios_df, modulo it out to our current freq to make sure it will always match a time in the full_time_range
        stay_antibios_df['administration_time'] = stay_antibios_df['administration_time'].dt.floor(
            '1min')  # TODO: Make frequency a parameter

        # Find the start and end time of each administrations (and considering gaps), fill with 1s in the new dataframe based on this
        prev_time = stay_antibios_df['administration_time'].iloc[0]

        if not pd.isnull(prev_time):
            for index, row in stay_antibios_df.iterrows():
                # Find first and last time of administration
                curr_time = row['administration_time']
                # Check if the gap is less than GAP_CONST
                if (curr_time - prev_time).seconds / 60 < gap_minutes:
                    # Then fill in the administration_complete_df with 1s between prev_time and curr_time
                    # Always also fill it with the MARGIN_CONST minutes before an actual point too
                    administration_complete_df.loc[
                        (administration_complete_df['date'] >= prev_time - pd.Timedelta(minutes=margin_minutes)) & (
                                administration_complete_df['date'] <= curr_time), 'antibiotics'] = 1
                else:
                    # Then fill in the administration_complete_df with 1s at curr_time, still with the margin
                    administration_complete_df.loc[
                        (administration_complete_df['date'] >= curr_time - pd.Timedelta(minutes=margin_minutes)) & (
                                administration_complete_df['date'] <= curr_time), 'antibiotics'] = 1

                # Update prev_time
                prev_time = curr_time

        # Fill the NaN values with 0
        administration_complete_df = administration_complete_df.fillna(0)

        # Append the new dataframe to the list
        full_range_dfs.append(administration_complete_df)

    if not full_range_dfs:
        return pd.Series(dtype=np.uint8)  # or an empty DataFrame if needed

    # Concatenate all the dataframes
    df_antibiotics = pd.concat(full_range_dfs)

    # Sort again because im scared
    df_antibiotics = df_antibiotics.sort_values(by=['stay_id', 'date'])

    # #DEBUG: Print df_antibiotics columns and head
    # print("Antibiotics columns",df_antibiotics.columns)
    # print("Antibiotics head",df_antibiotics.head())

    # #DEBUG: Iterate over all patients and print the number of 1s in the antibiotics vector
    # for stay_id, stay_antibios_df in df_antibiotics.groupby('stay_id'):
    #     print(f"Stay_id: {stay_id}, Number of 1s in antibiotics vector: {stay_antibios_df['antibiotics'].sum()}")
    #     #And total number of rows
    #     print(f"Total number of rows: {len(stay_antibios_df)}")

    return df_antibiotics['antibiotics'].values
    #.values[border1:border2]
    #DEBUG: I should not do this, since I already border my data_id on which i base my timeframe

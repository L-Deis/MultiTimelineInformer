import pandas as pd
import numpy as np


def generate_antibiotics_vector(df_antibiotics,
                                df_id,
                                gap_minutes=24 * 60,
                                margin_minutes=240,
                                freq='1min'):
    """
    Generates a binary time series indicating whether a patient was under antibiotic
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
        if stay_antibios_df.dropna().empty:
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

            start_time = start_time.floor(freq)
            end_time = end_time.floor(freq)

            # Clip to stay window
            # start_time = max(start_time, stay_start)
            # end_time = min(end_time, stay_end)
            if (start_time < stay_start):
                start_time = stay_start.floor(freq)
            if (start_time > stay_end):
                continue
            if (end_time > stay_end):
                end_time = stay_end.floor(freq)
            if (end_time < stay_start):
                continue

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


def generate_infections_vector(df_infections, 
                                df_id, 
                                margin_minutes=240,
                                freq='1min'):
    """
    Generates a binary time series indicating whether a patient had an infection at each minute of their stay.

    Parameters
    ----------
    df_infections : DataFrame with 'stay_id', 'start', 'end'
    df_id : DataFrame with 'stay_id', 'date' (can have duplicates)
    margin_minutes : int, pre-infection activation window

    Returns
    -------
    np.ndarray : Flattened binary infection vector
    """
    # df_id = df_id.copy()
    # df_id['row_idx'] = np.arange(len(df_id))

    all_results = []

    for stay_id, stay_ids_df in df_id.groupby('stay_id'):
        stay_infections_df = df_infections[df_infections['stay_id'] == stay_id]

        stay_start = stay_ids_df['date'].min()
        stay_end = stay_ids_df['date'].max()

        full_time_range = pd.date_range(start=stay_start, end=stay_end, freq=freq)
        n = len(full_time_range)

        # Get stay start offset (modulo) from the freq
        # stay_start_offset = (stay_start) % pd.Timedelta(freq)

        # Preallocate numpy array for infection flags
        infection_vector = np.zeros(n, dtype=np.uint8)

        # early exit for empty vectors
        if len(stay_infections_df) == 0:
            result = pd.DataFrame({
                'stay_id': stay_id,
                'date': full_time_range,
                'infections': infection_vector
            })
            all_results.append(result)
            continue

        # Build index mapping for fast lookup
        time_to_index = pd.Series(index=full_time_range, data=np.arange(n))

        # Prepare and clean up infection times
        stay_infections_df = stay_infections_df.copy()

        # Fill missing end values with infinity
        stay_infections_df['date_end'] = stay_infections_df['date_end'].fillna(pd.Timestamp.max)

        # Drop the rows with NA in start
        stay_infections_df = stay_infections_df.dropna()

        stay_infections_df['date_start'] = pd.to_datetime(stay_infections_df['date_start'])
        stay_infections_df['date_end'] = pd.to_datetime(stay_infections_df['date_end']) 
        stay_infections_df = stay_infections_df.sort_values(by='date_start')
        stay_infections_df['date_start'] = stay_infections_df['date_start'].dt.floor(freq)
        stay_infections_df['date_end'] = stay_infections_df['date_end'].dt.floor(freq)

        # For each row/infection, input the infection window
        for _, row in stay_infections_df.iterrows():
            start_time = row['date_start'] - pd.Timedelta(minutes=margin_minutes)
            end_time = row['date_end']

            start_time = start_time.floor(freq)
            end_time = end_time.floor(freq)

            # Clip to stay window
            if (start_time < stay_start):
                start_time = stay_start.floor(freq)
            if (start_time > stay_end):
                continue
            if (end_time > stay_end):
                end_time = stay_end.floor(freq)
            if (end_time < stay_start):
                continue

            # Get index range using closest matches
            try:
                start_idx = time_to_index.get(start_time)
                end_idx = time_to_index.get(end_time)
                infection_vector[start_idx:end_idx + 1] = 1
            except KeyError:
                print("There was an infection outside of a known stay. It has been skipped.")
                continue

        result = pd.DataFrame({
            'stay_id': stay_id,
            'date': full_time_range,
            'infections': infection_vector
        })
        all_results.append(result)

    if not all_results:
        # return empty df in case of empty inputs
        return pd.DataFrame(columns=['stay_id', 'date', 'infections'])

    return pd.concat(all_results).sort_values(by=['stay_id', 'date'])['infections'].values
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
                               margin_minutes=30,
                               freq='1min'):
    """
    Generates a binary vector indicating infection presence during ICU stays.

    Parameters
    ----------
    df_infections : pd.DataFrame
        Must contain:
            - 'stay_id': Unique identifier for the stay.
            - 'start': Infection start time in minutes since start of stay.
            - 'end': Infection end time in minutes since start of stay (or NaN for ongoing).

    df_id : pd.DataFrame
        Must contain:
            - 'stay_id': Unique identifier.
            - 'date': Timestamps covering full stay.

    margin_minutes : int, default=30
        Number of minutes before infection start to activate infection flag.

    freq : str, default='1min'
        Temporal resolution.

    Returns
    -------
    np.ndarray
        Flattened, sorted binary infection vector across all stays.
    """
    all_results = []

    # Group by stay
    for stay_id, stay_df in df_id.groupby('stay_id'):
        stay_start = stay_df['date'].min()
        stay_end = stay_df['date'].max()
        time_range = pd.date_range(start=stay_start, end=stay_end, freq=freq)
        n = len(time_range)

        infection_vector = np.zeros(n, dtype=np.uint8)
        time_to_index = pd.Series(index=time_range, data=np.arange(n))

        # Get infections for current stay
        stay_infections = df_infections[df_infections['stay_id'] == stay_id].copy()
        if stay_infections.empty:
            result = pd.DataFrame({
                'stay_id': stay_id,
                'date': time_range,
                'infection': infection_vector
            })
            all_results.append(result)
            continue

        # Convert offsets to actual timestamps
        for _, row in stay_infections.iterrows():
            # TODO round to freq
            offset_start = row['start'] - margin_minutes
            offset_end = row['end'] if pd.notnull(row['end']) else (time_range[-1] - stay_start).total_seconds() / 60

            # Clamp start and end
            offset_start = max(offset_start, 0)
            offset_end = min(offset_end, (time_range[-1] - stay_start).total_seconds() / 60)

            # Convert to timestamps
            start_time = (stay_start + pd.Timedelta(minutes=offset_start)).floor(freq)
            end_time = (stay_start + pd.Timedelta(minutes=offset_end)).floor(freq)

            try:
                start_idx = time_to_index.get(start_time)
                end_idx = time_to_index.get(end_time)
                if pd.notnull(start_idx) and pd.notnull(end_idx):
                    infection_vector[start_idx:end_idx + 1] = 1
            except KeyError:
                print(f"Infection window out of bounds for stay_id {stay_id}. Skipped.")
                continue

        result = pd.DataFrame({
            'stay_id': stay_id,
            'date': time_range,
            'infection': infection_vector
        })
        all_results.append(result)

    if not all_results:
        return np.array([], dtype=np.uint8)

    final_df = pd.concat(all_results).sort_values(by=['stay_id', 'date'])
    return final_df['infection'].values


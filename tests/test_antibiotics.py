import pandas as pd
import numpy as np
import time

from pandas.testing import assert_frame_equal

from utils.targetvectors import generate_antibiotics_vector, generate_antibiotics_vector_old


def test_matching_outputs():
    # Simulate input
    stay_ids = [1, 2]
    df_antibiotics = pd.DataFrame({
        'stay_id': [1, 1, 2, 2],
        'administration_time': pd.to_datetime([
            '2023-01-01 08:00', '2023-01-01 16:00',
            '2023-01-02 10:00', '2023-01-02 18:30'
        ])
    })

    df_id = pd.DataFrame({
        'stay_id': [1] * 5 + [2] * 5,
        'date': pd.to_datetime([
            '2023-01-01 07:00', '2023-01-01 08:00', '2023-01-01 12:00',
            '2023-01-01 16:00', '2023-01-01 20:00',
            '2023-01-02 08:00', '2023-01-02 10:00', '2023-01-02 14:00',
            '2023-01-02 18:30', '2023-01-02 21:00'
        ])
    })

    # Dummy mapping for legacy function (not used in logic)
    df_mappings = pd.DataFrame()
    border1, border2 = None, None

    new_vec = generate_antibiotics_vector(df_antibiotics, df_id)
    old_vec = generate_antibiotics_vector_old(df_antibiotics, df_id, df_mappings, border1, border2)

    if np.array_equal(new_vec, old_vec):
        print("✅ Outputs match between both implementations.")
    else:
        print("❌ Outputs DO NOT match between both implementations.")
        print(f"new_vec shape: {new_vec.shape}")
        print(f"old_vec shape: {old_vec.shape}")

        print("\ndifferences:")
        for i, (n, o) in enumerate(zip(new_vec, old_vec)):
            if n != o:
                print(f"Row {i}: new={n}, old={o}")

        print("\nequals:")
        for i, (n, o) in enumerate(zip(new_vec, old_vec)):
            if n == o:
                print(f"Row {i}: value={n}")


def test_missing_stays_and_empty_input():
    df_antibiotics = pd.DataFrame({
        'stay_id': [99],  # ID not present in df_id
        'administration_time': pd.to_datetime(['2023-01-01 12:00'])
    })

    df_id = pd.DataFrame({
        'stay_id': [1],
        'date': pd.to_datetime(['2023-01-01 10:00'])
    })

    df_mappings = pd.DataFrame()
    border1, border2 = None, None

    new_df = generate_antibiotics_vector(df_antibiotics, df_id)
    old_df = generate_antibiotics_vector_old(df_antibiotics, df_id, df_mappings, border1, border2)

    assert (new_df == 0).all()
    assert (old_df == 0).all()
    print("✅ Handled missing stay_id and empty vectors correctly.")

def test_missing_antibiotics_data():
    df_antibiotics = pd.DataFrame({
        'stay_id': [99],  # ID not present in df_id
        'administration_time': pd.to_datetime(['2023-01-01 12:00'])
    })

    df_id = pd.DataFrame({
        'stay_id': [1],
        'date': pd.to_datetime(['2023-01-01 10:00'])
    })

    df_mappings = pd.DataFrame()
    border1, border2 = None, None

    new_df = generate_antibiotics_vector(df_antibiotics, df_id)
    old_df = generate_antibiotics_vector_old(df_antibiotics, df_id, df_mappings, border1, border2)

    assert (new_df == 0).all()
    assert (old_df == 0).all()
    print("✅ Handled missing stay_id and empty vectors correctly.")


def test_varying_lengths_of_stays():
    np.random.seed(42)

    n_stays = 5
    df_id = []
    df_antibiotics = []

    for i in range(n_stays):
        stay_id = i + 1
        start = pd.Timestamp('2023-01-01') + pd.Timedelta(days=i)
        end = start + pd.Timedelta(hours=np.random.randint(2, 48))  # stays from 2h to 2 days
        dates = pd.date_range(start=start, end=end, freq='1min')

        df_id.extend([{'stay_id': stay_id, 'date': d} for d in dates])

        admin_times = np.random.choice(dates, size=np.random.randint(1, 5), replace=False)
        for t in admin_times:
            df_antibiotics.append({'stay_id': stay_id, 'administration_time': t})

    df_id = pd.DataFrame(df_id)
    df_antibiotics = pd.DataFrame(df_antibiotics)
    df_mappings = pd.DataFrame()
    border1, border2 = None, None

    new_df = generate_antibiotics_vector(df_antibiotics, df_id)
    old_df = generate_antibiotics_vector_old(df_antibiotics, df_id, df_mappings, border1, border2)

    # Match shape
    assert len(new_df) == len(old_df)
    print("✅ Successfully handled varying stay lengths.")


def test_performance():
    # Simulate large input
    n_stays = 100
    all_ids = []
    all_dates = []
    all_antibiotics = []

    for stay_id in range(1, n_stays + 1):
        start = pd.Timestamp('2023-01-01') + pd.Timedelta(days=np.random.randint(0, 30))
        end = start + pd.Timedelta(days=1)
        dates = pd.date_range(start=start, end=end, freq='1min')
        all_dates.extend([{'stay_id': stay_id, 'date': d} for d in dates])

        n_admins = np.random.randint(2, 10)
        admin_times = np.random.choice(dates, n_admins, replace=False)
        for at in admin_times:
            all_antibiotics.append({'stay_id': stay_id, 'administration_time': at})

    df_id = pd.DataFrame(all_dates)
    df_antibiotics = pd.DataFrame(all_antibiotics)
    df_mappings = pd.DataFrame()
    border1, border2 = None, None

    t1 = time.time()
    _ = generate_antibiotics_vector(df_antibiotics, df_id)
    t2 = time.time()
    print(f"New version time: {t2 - t1:.2f}s")

    t3 = time.time()
    _ = generate_antibiotics_vector_old(df_antibiotics, df_id, df_mappings, border1, border2)
    t4 = time.time()
    print(f"Old version time: {t4 - t3:.2f}s")


if __name__ == "__main__":
    test_matching_outputs()
    test_missing_stays_and_empty_input()
    test_varying_lengths_of_stays()
    test_performance()

import os
import numpy as np
import pandas as pd
import gc
import pickle

from torch.utils.data import Dataset

from utils.tools import StandardScaler
from utils.timefeatures import time_features
from utils.targetvectors import generate_infections_vector

import warnings

warnings.filterwarnings('ignore')


class Dataset_eICU(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S',
                 data_path={
                     "vitals": "vitals.csv",
                     "patients": "patient.csv",
                     "infections": "infection.csv",
                 },
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None, use_preprocessed=False,
                 data_compress=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.set_flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.use_preprocessed = use_preprocessed
        self.data_compress = data_compress
        self.__read_data__()

    def _get_preprocessed_path(self):
        """Get the path for the preprocessed data file based on the dataset type."""
        return os.path.join(self.root_path, f"preprocessed_{self.set_flag}.pkl")

    def _save_preprocessed_data(self):
        print("Saving preprocessed data.")
        """Save the preprocessed data to a pickle file."""
        preprocessed_data = {
            'data_x': self.data_x,
            'data_y': self.data_y,
            'data_stamp': self.data_stamp,
            'data_id': self.data_id,
            'data_static': self.data_static,
            'data_infections': self.data_infections,
            'scaler': self.scaler
        }

        save_path = self._get_preprocessed_path()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(preprocessed_data, f)
        print(f"Saved preprocessed data to {save_path}")

    def _load_preprocessed_data(self):
        print("Try to loading preprocessed data...")
        """Load preprocessed data from a pickle file."""
        load_path = self._get_preprocessed_path()
        if not os.path.exists(load_path):
            print(f"No preprocessed data found at {load_path}")
            return False

        try:
            with open(load_path, 'rb') as f:
                preprocessed_data = pickle.load(f)

            self.data_x = preprocessed_data['data_x']
            self.data_y = preprocessed_data['data_y']
            self.data_stamp = preprocessed_data['data_stamp']
            self.data_id = preprocessed_data['data_id']
            self.data_static = preprocessed_data['data_static']
            self.data_infections = preprocessed_data['data_infections']
            self.scaler = preprocessed_data['scaler']

            self._compress_data()

            print(f"Loaded preprocessed data from {load_path}")
            return True
        except Exception as e:
            print(f"Error loading preprocessed data: {e}")
            return False

    def _compress_data(self):
        # if datafreq is not 1min, skip every x row
        if self.data_compress == "skip" and pd.Timedelta(self.freq) != pd.Timedelta(minutes=1):
            # TODO this is simple downsampling with no regard for missing data
            #  every Nth element is taken, this can create gaps that are much worse than what exists in the basic data
            self.data_infections = self.data_infections[::int(pd.Timedelta(self.freq) / pd.Timedelta(minutes=1))]
            self.data_x = self.data_x[::int(pd.Timedelta(self.freq) / pd.Timedelta(minutes=1))]
            self.data_y = self.data_y[::int(pd.Timedelta(self.freq) / pd.Timedelta(minutes=1))]
            self.data_stamp = self.data_stamp[::int(pd.Timedelta(self.freq) / pd.Timedelta(minutes=1))]
            self.data_id = self.data_id[::int(pd.Timedelta(self.freq) / pd.Timedelta(minutes=1))]
        if self.data_compress == "mean" and pd.Timedelta(self.freq) != pd.Timedelta(minutes=1):
            factor = int(pd.Timedelta(self.freq) / pd.Timedelta(minutes=1))
            # take the mean of the factor//2 points before and after the timestep
            # TODO: Implement summing up the data by getting the mean from a timewindow

    def __read_data__(self):
        if self.use_preprocessed and self._load_preprocessed_data():
            return

        # --- VITALS ---
        print("DATALOADER: Start Loading Vitals...")
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(
            self.root_path,
            self.data_path["vitals"]),
            usecols=['minutes_since_admission', 'HR', 'respiratory_rate', 'oxygen_saturation', 'SYS', 'DIA', 'BP_mean',
                    'stay_id'],
                # 'date_time', 'HR', 'Ademhaling_frequentie', 'Saturatie', 'SYS', 'DIA', 'Bloeddruk_gemiddeld',
                    #  'stay_id'],  # Don't load mdn to save memory
            # nrows=10000,  #DEBUG: Read only the first 1000 lines
        )

        # Remove every row where minutes_since_admission is negative
        df_raw = df_raw[df_raw['minutes_since_admission'] >= 0]
        # Sort by ascending stay_id and minutes_since_admission
        df_raw = df_raw.sort_values(by=['stay_id', 'minutes_since_admission'])

        # Create date column from minutes_since_admission
        df_raw['date'] = pd.to_datetime(df_raw['minutes_since_admission'], unit='m', errors='coerce')
        # floor date to the nearest freq
        df_raw['date'] = df_raw['date'].dt.floor(self.freq)
        df_raw.drop(columns=['minutes_since_admission'], inplace=True)

        # --- eICU/MEWS Specific pre-processing ---
        # Print all columns names
        print("Columns names", df_raw.columns)
        # Ensure the datatype of the known number columns is int
        column_types = {
            'stay_id': np.uint32,
            'HR': np.uint16,
            'respiratory_rate': np.uint16,
            'oxygen_saturation': np.uint16,
            'SYS': np.uint16,
            'DIA': np.uint16,
            'BP_mean': np.uint16,
            # fallback: np.float32
        }

        for col, dtype in column_types.items():
            print(col)
            # Fill NaNs with 0
            df_raw[col] = df_raw[col].fillna(0)

            # Remove negative values (set them to 0)
            df_raw[col] = df_raw[col].map(lambda x: x if x >= 0 else 0)

            # Round values and cast to desired type
            df_raw[col] = df_raw[col].round().astype(dtype)

        # Fallback for all remaining numeric columns
        for col in df_raw.select_dtypes(include='number').columns:
            if col not in column_types:
                df_raw[col] = df_raw[col].fillna(0).astype(np.float32)

        # Ensure the data is sorted per date_time per stay_id
        df_raw = df_raw.sort_values(by=['stay_id', 'date'])

        # For each stay_id (patient), find the start date and end date, if there is a gap in time between two cells, create the missing row and fill it with 0
        full_range_dfs = []
        for stay_id, patient_df in df_raw.groupby('stay_id'):
            # Generate a full time index from min to max timestamp
            full_time_range = pd.date_range(
                start=patient_df.date.min(),
                end=patient_df.date.max(),
                freq=self.freq,  # TODO: Check if freq is correct
            )

            # Create a new dataframe with the full time range
            patient_complete_df = pd.DataFrame({"date": full_time_range})
            patient_complete_df['stay_id'] = stay_id  # Assign the stay_id to all rows

            # Merge back with the original data (to fill the new table)
            patient_complete_df = patient_complete_df.merge(patient_df, on=['stay_id', 'date'], how='left')

            # Fill the NaN values with 0
            patient_complete_df = patient_complete_df.fillna(0)
            #TODO: Also add a mode to fill the NaN values with the closest value

            # Append the new dataframe to the list
            full_range_dfs.append(patient_complete_df)

        # Concatenate all the dataframes
        df_raw = pd.concat(full_range_dfs)

        # Sort again because im scared
        df_raw = df_raw.sort_values(by=['stay_id', 'date'])

        # --- eICU Specific pre-processing end ---

        '''
        df_raw.columns: ['stay_id', 'date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('stay_id')
            cols.remove('date')
        df_raw = df_raw[['stay_id', 'date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]  # train: 70%, val: 10%, test: 20%
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            # df_raw columns without stay_id and date_time
            cols_data = list(df_raw.columns)
            cols_data.remove('stay_id')
            cols_data.remove('date')
            df_data = df_raw[cols_data]
            df_id = df_raw[['stay_id', 'date']]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            df_id = df_raw[['stay_id', 'date']]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        stay_ids_to_keep = df_raw['stay_id'].unique()

        # Del df to save memory
        del df_raw
        gc.collect()

        # Save ID data
        self.data_id = df_id.values[border1:border2]

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        # --- VITALS END ---

        # --- STATIC ---
        print("DATALOADER: Start Loading Admissions...")
        df_patients = pd.read_csv(os.path.join(self.root_path,
                                                 self.data_path["patients"]))
        # Filter out patients that are not in the stay_ids_to_keep
        df_patients = df_patients[df_patients['stay_id'].isin(stay_ids_to_keep)]

        # create new column is_man based on PatientGeslacht (M=1, V=0)
        def is_man(gender):
            if gender == 'M' or gender == 'Male':
                return 1
            else:
                return 0

        df_patients["is_man"] = df_patients["gender"].map(is_man)
        # Rename columns: 'TrajectSpecialismeAgbCode' -> 'dbc_dept', 'TrajectDiagnoseCode' -> 'dbc_diag'
        cols_to_keep = [
            'stay_id',
            'age',
            'is_man',
        ]
        df_patients = df_patients[cols_to_keep]

        # Make sure all numbers are ints
        df_patients['stay_id'] = pd.to_numeric(df_patients['stay_id'], errors='coerce').fillna(0).astype(np.uint32)
        df_patients['age'] = pd.to_numeric(df_patients['age'], errors='coerce').fillna(0).astype(np.uint16)
        df_patients['is_man'] = pd.to_numeric(df_patients['is_man'], errors='coerce').fillna(0).astype(np.uint8)

        # For each categorical feature in static_data, it needs to be mapped to [0,1,2,...,n-1] where n is the number of unique categories
        N_NUM = 1  #TODO: Pass the argument from config # Number of numerical features, placed before each categorical feature, every following feature is categorical
        # -> Skip 'stay_id' and n_num afterwards;
        # Get columns names
        cols_static = list(df_patients.columns)
        cols_static.remove('stay_id')
        cols_static = cols_static[N_NUM:]
        for col in cols_static:
            # Get unique values
            unique_values = df_patients[col].unique()
            # Get n unique and print it
            n_unique = len(unique_values)
            print(f"Unique categories for {col}: {n_unique}")
            # Map the unique values to a list of integers
            mapping = dict(zip(unique_values, range(n_unique)))
            # Map the column to the new values
            df_patients[col] = df_patients[col].map(mapping)

        # self.data_static = Dict: key=[stay_id], value=[age, is_man, dbc_dept, dbc_diag]
        self.data_static = df_patients.set_index('stay_id').T.to_dict('list')

        # del df_patients to save memory
        del df_patients
        gc.collect()
        # --- STATIC END ---

        # --- ANTIBIOTICS ---
        print("DATALOADER: Start Loading Infections...")
        df_infections = pd.read_csv(os.path.join(self.root_path,
                                                  self.data_path["infections"]))
        # Filter out infections that are not in the stay_ids_to_keep
        df_infections = df_infections[df_infections['stay_id'].isin(stay_ids_to_keep)]

        # Keep only stay_ids that are in df_id
        df_infections = df_infections[df_infections['stay_id'].isin(df_id['stay_id'])]

        # Keep only the columns that are needed: ['mdn', 'stay_id', 'description', 'status', 'administration_time']
        df_infections = df_infections[['stay_id', 'start_offset', 'end_offset']]

        # Sort by ascending stay_id and start_offset
        df_infections = df_infections.sort_values(by=['stay_id', 'start_offset'])

        # Frist, if end_offset is negative, drop the row, keep also the nulls
        df_infections = df_infections[df_infections['end_offset'] >= 0 | df_infections['end_offset'].isna()]
        # End, if start_offset is negative, set it to 0
        df_infections['start_offset'] = df_infections['start_offset'].apply(lambda x: 0 if x < 0 else x)

        # Convert start_offset and end_offset to date_start and date_end
        df_infections['date_start'] = pd.to_datetime(df_infections['start_offset'], unit='m', errors='coerce')
        df_infections['date_end'] = pd.to_datetime(df_infections['end_offset'], unit='m', errors='coerce')
        # floor date_start and date_end to the nearest freq
        df_infections['date_start'] = df_infections['date_start'].dt.floor(self.freq)
        df_infections['date_end'] = df_infections['date_end'].dt.floor(self.freq)
        df_infections.drop(columns=['start_offset', 'end_offset'], inplace=True)

        self.data_infections = generate_infections_vector(df_infections, df_id,
                                                           freq=self.freq)  # TODO: Check if freq is correct

        #DEBUG: Save the infections data to a csv file
        self.data_infections.to_csv(os.path.join(self.root_path, "infections.csv"), index=False)
        #DEBUG: Print meann of the infections data
        print("!!! Mean of infections data: ", self.data_infections.mean())

        # del df_antibiotics to save memory
        del df_infections
        gc.collect()

        # --- DEBUG ---
        # PRINT SHAPE OF EACH DATA
        print("DATALOADER: Shapes of each data")
        print(f"data_x: {self.data_x.shape}")
        print(f"data_y: {self.data_y.shape}")
        print(f"data_stamp: {self.data_stamp.shape}")
        print(f"data_id: {self.data_id.shape}")
        # print(f"data_static: {self.data_static.shape}")
        print(f"data_infections: {self.data_infections.shape}")
        # --- DEBUG END ---

        # After processing all data, save it if using preprocessed mode
        if self.use_preprocessed:
            self._save_preprocessed_data()

        self._compress_data()

        # --- DEBUG ---
        # PRINT SHAPE OF EACH DATA
        print("DATALOADER: Shapes of each data after compression")
        print(f"data_x: {self.data_x.shape}")
        print(f"data_y: {self.data_y.shape}")
        print(f"data_stamp: {self.data_stamp.shape}")
        print(f"data_id: {self.data_id.shape}")
        # print(f"data_static: {self.data_static.shape}")
        print(f"data_infections: {self.data_infections.shape}")
        # --- DEBUG END ---

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x_id = self.data_id[s_begin:s_end]
        seq_y_id = self.data_id[r_begin:r_end]

        # Get the static data for the stay_id
        seq_static = self.data_static[seq_y_id[-1][0]]

        # Get the antibiotics data, matching the y range
        seq_infections = self.data_infections[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_id, seq_y_id, seq_static, seq_infections

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_eICU_special(Dataset):
    # Columns in static_patient_info.csv.gz:
    # ['stay_id', 'gender', 'admit_dx', 'hosp_admit_source', 'unit_type', 'unit_admit_source', 'weight_kg', 'discharge_offset', 'age', 'infection_tag']
    # Columns in balanced_24h_windows.csv.gz:
    # ['stay_id', 'ts', 'offset_min', 'temp_c_mean', 'temp_c_std', 'temp_range30', 'temp_c_slope30h1', 'HR_bpm_mean', 'HR_bpm_std', 'HR_sdnn30', 'HR_delta_prev', 'RR_bpm_mean', 'RR_bpm_std', 'RR_range30', 'MAP_mmHg_mean', 'MAP_mmHg_std', 'MAP_slope30h1', 'spo2_pct_mean', 'spo2_pct_std', 'spo2_cv30', 'spo2_pct_min', 'infected', 'window_has_infection', 'infection_tag', 'window_has_temp']
    # Shape of vitals dataset: (989856, 25) -> 20,622 stays (10311/10311) keep 20K for training, and then for val: 322, test: 300
    # Shape of static dataset: (195339, 10)
    def __init__(self, root_path, flag='train', size=None,
                 features='S',
                 data_path={
                     "vitals": "balanced_24h_windows.csv.gz",
                     "static": "static_patient_info.csv.gz",
                 },
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None, use_preprocessed=False,
                 data_compress=None):
        # size [seq_len, label_len, pred_len]
        # info
        # if size == None:
        #     self.seq_len = 24 * 4 * 4
        #     self.label_len = 24 * 4
        #     self.pred_len = 24 * 4
        # else:
        #     self.seq_len = size[0]
        #     self.label_len = size[1]
        #     self.pred_len = size[2]
        self.seq_len = 12 * 2 # 12 hours , 1 point every 30 minutes
        self.label_len = 12 * 2 # label and seq overlap
        self.pred_len = 12 * 2 #also pred the next 12 hours
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.set_flag = flag

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.use_preprocessed = False
        self.data_compress = False
        self.__read_data__()

    def __read_data__(self):
        # --- VITALS ---
        print("DATALOADER: Start Loading Vitals...")
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(
            self.root_path,
            self.data_path["vitals"]),
            usecols=['stay_id', 'offset_min', 'temp_c_mean', 'temp_c_std', 'temp_range30', 'temp_c_slope30h1', 'HR_bpm_mean', 'HR_bpm_std', 'HR_sdnn30', 'HR_delta_prev', 'RR_bpm_mean', 'RR_bpm_std', 'RR_range30', 'MAP_mmHg_mean', 'MAP_mmHg_std', 'MAP_slope30h1', 'spo2_pct_mean', 'spo2_pct_std', 'spo2_cv30', 'spo2_pct_min', 'infected', 'window_has_infection']
            # nrows=10000,  #DEBUG: Read only the first 1000 lines
        )
        # Sort by ascending stay_id and minutes_since_admission
        df_raw = df_raw.sort_values(by=['window_has_infection', 'stay_id', 'offset_min'])

        # TODO: Dont forgot to drop window_has_infection later
        df_raw.drop(columns=['window_has_infection'], inplace=True)

        # Make offset_min start at 0 for each stay_id
        df_raw['offset_min'] = df_raw.groupby('stay_id')['offset_min'].transform(lambda x: x - x.min())
        #rename offset_min to date
        df_raw.rename(columns={'offset_min': 'date'}, inplace=True)

        # Create a mask df with the same shape as df_raw, with True for the cells that are not NaN
        df_mask = df_raw.notna()

        # Zero fill all NaN
        df_raw = df_raw.fillna(0)

        '''
        df_raw.columns: ['stay_id', 'date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns);
        if self.cols:
            cols = self.cols.copy()
            # cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            # cols.remove(self.target)
            cols.remove('stay_id')
            cols.remove('date')
        # df_raw = df_raw[['stay_id', 'date'] + cols + [self.target]]
        df_raw = df_raw[['stay_id', 'date'] + cols]

        # num_train = int(len(df_raw) * 0.7)
        # num_test = int(len(df_raw) * 0.2)
        # num_vali = len(df_raw) - num_train - num_test
        num_train = 20000
        num_vali = 322
        num_test = 300

        # border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        # border2s = [num_train, num_train + num_vali, len(df_raw)]  # train: 70%, val: 10%, test: 20%
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]

        #for this one instead border is instantly just a set of ids true false, not a from to
        #if train then border = 10k true then 311 false then 10k then 311 false
        #if test then border = 10k false then 150 true then 161 false then 10k false then 150 true then 161 false
        #if val then border = 10k false then 150 false then 161 true then 10k false then 150 false then 161 true
        if self.set_type == 0:
            border = [True for i in range(num_train//2)]*48 + [False for i in range(num_vali//2)]*48 + [True for i in range(num_test//2)]*48
            border = border * 2
        elif self.set_type == 1:
            border = [False for i in range(num_train//2)]*48 + [True for i in range(num_vali//2)]*48 + [False for i in range(num_test//2)]*48
            border = border * 2
        elif self.set_type == 2:
            border = [False for i in range(num_train//2)]*48 + [False for i in range(num_vali//2)]*48 + [True for i in range(num_test//2)]*48
            border = border * 2

        if self.features == 'M' or self.features == 'MS':
            # df_raw columns without stay_id and date_time
            cols_data = list(df_raw.columns)
            cols_data.remove('stay_id')
            cols_data.remove('date')
            df_data = df_raw[cols_data]
            df_id = df_raw[['stay_id', 'date']]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            df_id = df_raw[['stay_id', 'date']]

        if self.scale:
            # train_data = df_data[border1s[0]:border2s[0]]
            # ! Dont scale the last vector (infected) !
            infected_data = df_data[border].iloc[:, -1]
            train_data = df_data[border].iloc[:, :-1]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(train_data.values)
            # Add the infected data back to the scaled data
            data = np.concatenate((data, infected_data.values.reshape(-1, 1)), axis=1)
        else:
            data = df_data[border].values

        # df_stamp = df_raw[['date']][border1:border2]
        df_stamp = df_raw[['date']][border]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        # Del df to save memory
        del df_raw
        gc.collect()

        # Save ID data
        self.data_id = df_id.values[border]

        self.data_x = data
        # self.data_x = data[border]
        #df mask remove the two first columns (stay_id and date) to keep only the features
        df_mask = df_mask[border].iloc[:, 2:]
        # Convert the mask to a numpy array
        self.data_mask = df_mask.values
        if self.inverse:
            self.data_y = df_data.values[border]
        else:
            # self.data_y = data[border]
            self.data_y = data
        self.data_stamp = data_stamp
        # --- VITALS END ---

        # --- STATIC ---
        print("DATALOADER: Start Loading Admissions...")
        df_static = pd.read_csv(os.path.join(self.root_path, self.data_path["static"]),
                                usecols=['stay_id', 'age', 'weight_kg', 'gender', 'admit_dx', 'hosp_admit_source', 'unit_type', 'unit_admit_source'],
                                )
        
        cols_order = ['stay_id', 'age', 'weight_kg', 'gender', 'admit_dx', 'hosp_admit_source', 'unit_type', 'unit_admit_source']
        #Order the columns
        df_static = df_static[cols_order]

        # Zero fill all NaN
        df_static = df_static.fillna(0)

        N_NUM = 2 #Number of numerical features, placed before each categorical feature, every following feature is categorical

        # Map every other features to a number from 0 to nunique per feature
        # For each categorical feature in static_data, it needs to be mapped to [0,1,2,...,n-1] where n is the number of unique categories
        # Get columns names
        cols_static = list(df_static.columns)
        cols_static.remove('stay_id')
        cols_static = cols_static[N_NUM:]

        # for each string column, keep only the first 6 chars (or less if shorter)
        for col in cols_static:
            df_static[col] = df_static[col].astype(str).str[:6]
        # for each string column, keep only the first 6 chars (or less if shorter)

        for col in cols_static:
            # Get unique values
            unique_values = df_static[col].unique()
            # Get n unique and print it
            n_unique = len(unique_values)
            print(f"Unique categories for {col}: {n_unique}")
            # Map the unique values to a list of integers
            mapping = dict(zip(unique_values, range(n_unique)))
            # Map the column to the new values
            df_static[col] = df_static[col].map(mapping)

        # Make sure all numbers are ints
        # df_static['stay_id'] = pd.to_numeric(df_static['stay_id'], errors='coerce').fillna(0).astype(np.uint32)
        df_static['age'] = pd.to_numeric(df_static['age'], errors='coerce').fillna(0).astype(np.uint16)
        df_static['weight_kg'] = pd.to_numeric(df_static['weight_kg'], errors='coerce').fillna(0).astype(np.uint16)

        self.data_static = df_static.set_index('stay_id').T.to_dict('list')
        #make sure the keys in data_static are the same type as the keys in data_id
        self.data_static = {int(k): v for k, v in self.data_static.items()}

        # del df_patients to save memory
        del df_static
        gc.collect()
        # --- STATIC END ---

        # --- DEBUG ---
        # PRINT SHAPE OF EACH DATA
        print("DATALOADER: Shapes of each data")
        print(f"data_x: {self.data_x.shape}")
        print(f"data_y: {self.data_y.shape}")
        print(f"data_stamp: {self.data_stamp.shape}")
        print(f"data_id: {self.data_id.shape}")
        # print(f"data_static: {self.data_static.shape}")
        print(f"data_mask: {self.data_mask.shape}")
        # --- DEBUG END ---

        # # After processing all data, save it if using preprocessed mode
        # if self.use_preprocessed:
        #     self._save_preprocessed_data()

        # self._compress_data()

    def __getitem__(self, index):
        # One item is one full stay, aka also one window of 24h and then it moves into another 24h
        s_begin = index * 48 #24 hours * 2 (30 minutes)
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_y_mask = self.data_mask[r_begin + self.label_len:r_end]

        seq_x_id = self.data_id[s_begin:s_end]
        seq_y_id = self.data_id[r_begin:r_end]

        # #DEBUG, Print all keys in data_static
        # print("Keys in data_static: ", self.data_static.keys())
        # #DEBUG, Print all keys in seq_y_id
        # print("Keys in seq_y_id: ", seq_y_id[-1][0])

        # #DEBUG print if the key was found
        if seq_y_id[-1][0] in self.data_static:
            # print("Key found in data_static: ", seq_y_id[-1][0])
            # Get the static data for the stay_id
            seq_static = self.data_static[int(seq_y_id[-1][0])]
            seq_static = np.array(seq_static)
        else:
            # print("Key not found in data_static: ", seq_y_id[-1][0])
            # If the key is not found, set seq_static to None or some default value
            first_key = next(iter(self.data_static))
            seq_static = np.zeros_like(self.data_static[first_key])

        # Get the antibiotics data, matching the y range
        # seq_infections = self.data_infections[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_id, seq_y_id, seq_static, seq_y_mask

    def __len__(self):
        # return len(self.data_x) - self.seq_len - self.pred_len + 1
        return len(self.data_x) // 48

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
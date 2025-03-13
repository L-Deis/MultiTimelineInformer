import os
import numpy as np
import pandas as pd
import gc

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_MEWS(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path={"vitals":"vitals.csv", "admissions":"admissions.csv", "mappings":"mapping.csv", "antibiotics":"antibiotics.csv"}, 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.path_vitals = data_path["vitals"]
        self.path_admissions = data_path["admissions"]
        self.path_mapping = data_path["mappings"]
        self.path_antibiotics = data_path["antibiotics"]
        self.__read_data__()

    def __read_data__(self):
        # --- VITALS ---
        print("DATALODER: Start Loading Vitals...")
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(
            self.root_path,
            self.path_vitals),
            usecols=['date_time', 'HR', 'Ademhaling_frequentie', 'Saturatie', 'SYS', 'DIA', 'Bloeddruk_gemiddeld', 'stay_id'], #Don't load mdn to save memory
            nrows=1000, #DEBUG: Read only the first 1000 lines
        )
        
        #Get head
        # print("Head of df_raw",df_raw.head())

        #DEBUG: Load mappings df, to keep only the stay_id in df_raw that are in df_mappings
        df_mappings = pd.read_csv(os.path.join(self.root_path,
                                            self.path_mapping),
                                            usecols=['stay_id']) #Only load stay_id to save memory
        #Create a list of all the stay_id that are in df_mappings
        stay_ids = df_mappings['stay_id'].unique()
        #Filter the df_raw to keep only the stay_id that are in stay_ids
        df_raw = df_raw[df_raw['stay_id'].isin(stay_ids)]

        #inner join df_mappings and df_admissions on double key 'mdn' and 'stay_id'
        # df_mappings = pd.merge(df_mappings, df_raw, how='inner', left_on=['mdn', 'stay_id'], right_on=['mdn', 'stay_id'])

        
        # --- MEWS Specific pre-processing ---
        #Print all columns names
        print("Columns names",df_raw.columns)
        #Drop mdn column
        # df_raw.drop(columns=['mdn'], inplace=True)
        #Rename date_time to date
        df_raw.rename(columns={'date_time': 'date'}, inplace=True)
        #Ensure the datatype of each column is float, except date_time which is datetime
        for col in df_raw.columns:
            if col == 'date':
                df_raw[col] = pd.to_datetime(df_raw[col])
            elif col in ['stay_id']:
                #Turn the NaN to 0
                df_raw[col].fillna(0, inplace=True)
                #Convert the column to int
                df_raw[col] = df_raw[col].astype(np.uint32)                
            elif col in ['HR', 'Ademhaling_frequentie', 'Saturatie', 'SYS', 'DIA', 'Bloeddruk_gemiddeld']:
                #Turn the NaN to 0
                df_raw[col].fillna(0, inplace=True)
                #Convert the column to int
                df_raw[col] = df_raw[col].astype(np.uint16)
            else:
                #Turn the NaN to 0
                df_raw[col].fillna(0, inplace=True)
                #Convert the column to float
                df_raw[col] = df_raw[col].astype(np.float32)
        #Ensure the data is sorted per date_time per stay_id
        df_raw = df_raw.sort_values(by=['stay_id', 'date'])

        #For each stay_id (patient), find the start date and end date, if there is a gap in time between two cells, create the missing row and fill it with 0
        full_range_dfs = []
        for stay_id, patient_df in df_raw.groupby('stay_id'):
            # Generate a full time index from min to max timestamp
            full_time_range = pd.date_range(
                start=patient_df.date.min(), 
                end=patient_df.date.max(), 
                freq='1min' # 1 minute frequency #TODO: Make frequency a parameter
            )

            #Create a new dataframe with the full time range
            patient_complete_df = pd.DataFrame({"date": full_time_range})
            patient_complete_df['stay_id'] = stay_id #Assign the stay_id to all rows

            #Merge back with the original data (to fill the new table)
            patient_complete_df = patient_complete_df.merge(patient_df, on=['stay_id','date'], how='left')

            #Fill the NaN values with 0
            patient_complete_df = patient_complete_df.fillna(0)

            #Append the new dataframe to the list
            full_range_dfs.append(patient_complete_df)

        #Concatenate all the dataframes
        df_raw = pd.concat(full_range_dfs)

        #Sort again because im scared
        df_raw = df_raw.sort_values(by=['stay_id', 'date'])

        # --- MEWS Specific pre-processing end ---

        '''
        df_raw.columns: ['stay_id', 'date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('stay_id'); cols.remove('date')
        df_raw = df_raw[['stay_id', 'date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features=='M' or self.features=='MS':
            #df_raw columns without stay_id and date_time
            cols_data = list(df_raw.columns); cols_data.remove('stay_id'); cols_data.remove('date')
            df_data = df_raw[cols_data]
            df_id = df_raw[['stay_id', 'date']]
        elif self.features=='S':
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

        #Del df to save memory
        del df_raw
        gc.collect()

        #Save ID data
        self.data_id = df_id.values[border1:border2]

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        # --- VITALS END ---

        # --- STATIC ---
        print("DATALODER: Start Loading Admissions...")
        df_admissions = pd.read_csv(os.path.join(self.root_path,
                                            self.path_admissions))
        df_mappings = pd.read_csv(os.path.join(self.root_path,
                                            self.path_mapping),
                                            usecols=['stay_id', 'admission_id', 'mdn'])
        #inner join df_mappings and df_admissions on double key 'mdn' and 'admission_id'
        df_admissions = pd.merge(df_mappings, df_admissions, how='inner', left_on=['mdn', 'admission_id'], right_on=['mdn', 'admission_id'])

        #create new column is_man based on PatientGeslacht (M=1, V=0)
        def is_man(gender):
            if gender == 'M':
                return 1
            else:
                return 0
        
        df_admissions["is_man"] = df_admissions["PatientGeslacht"].map(is_man)
        #Rename columns: 'TrajectSpecialismeAgbCode' -> 'dbc_dept', 'TrajectDiagnoseCode' -> 'dbc_diag'
        df_admissions = df_admissions.rename(columns={'TrajectSpecialismeAgbCode': 'dbc_dept', 'TrajectDiagnoseCode': 'dbc_diag'})
        cols_to_keep = ['stay_id', 'age', 'is_man', 'dbc_dept', 'dbc_diag']
        df_admissions = df_admissions[cols_to_keep]

        #Make sure everything is an int
        df_admissions['stay_id'] = pd.to_numeric(df_admissions['stay_id'], errors='coerce').fillna(0).astype(np.uint32)
        df_admissions['age'] = pd.to_numeric(df_admissions['age'], errors='coerce').fillna(0).astype(np.uint16)
        df_admissions['is_man'] = pd.to_numeric(df_admissions['is_man'], errors='coerce').fillna(0).astype(np.uint8)
        df_admissions['dbc_dept'] = pd.to_numeric(df_admissions['dbc_dept'], errors='coerce').fillna(0).astype(np.uint16)
        df_admissions['dbc_diag'] = pd.to_numeric(df_admissions['dbc_diag'], errors='coerce').fillna(0).astype(np.uint16)

        # For each categorical feature in static_data, it needs to be mapped to [0,1,2,...,n-1] where n is the number of unique categories
        N_NUM = 1 #Number of numerical features, placed before each categorical feature, every following feature is categorical
        #-> Skip 'stay_id' and n_num afterwards;
        #Get columns names
        cols_static = list(df_admissions.columns)
        cols_static.remove('stay_id')
        cols_static = cols_static[N_NUM:]
        for col in cols_static:
            #Get unique values
            unique_values = df_admissions[col].unique()
            #Get n unique and print it
            n_unique = len(unique_values)
            print(f"Unique categories for {col}: {n_unique}")
            #Map the unique values to a list of integers
            mapping = dict(zip(unique_values, range(n_unique)))
            #Map the column to the new values
            df_admissions[col] = df_admissions[col].map(mapping)

        # self.data_static = Dict: key=[stay_id], value=[age, is_man, dbc_dept, dbc_diag]
        self.data_static = df_admissions.set_index('stay_id').T.to_dict('list')

        #del df_admissions to save memory
        del df_admissions
        gc.collect()

        # --- STATIC END ---

        # --- ANTIBIOTICS ---
        print("DATALODER: Start Loading Antibiotics...")
        df_antibiotics = pd.read_csv(os.path.join(self.root_path,
                                            self.path_antibiotics))

        #Size of the gap in antibiotics allowed to be considered two different antibiotics administrations
        GAP_CONST = 48*60 #2 days in minutes
        MARGIN_CONST = 4*60 #4 hours in minutes

        #No stay_id omg i forgot this will be my curse forever
        #DEBUG: Lukas forgot, so for now, get back stay_id from matching df_antibiotics with df_mapping on mdn
        #It also keeps only the mdn/stay_id that are in df_mappings, just like for the vitals.
        df_antibiotics = df_mappings.merge(df_antibiotics, how='left', left_on=['mdn'], right_on=['mdn'])

        #Keep only stay_ids that are in df_id
        df_antibiotics = df_antibiotics[df_antibiotics['stay_id'].isin(df_id['stay_id'])]

        #Keep only the columns that are needed: ['mdn', 'stay_id', 'description', 'status', 'administration_time']
        df_antibiotics = df_antibiotics[['stay_id', 'description', 'status', 'administration_time']]

        #Make sure the administration_time is a datetime
        df_antibiotics['administration_time'] = pd.to_datetime(df_antibiotics['administration_time'])
        #Sort antibiotics by stay_id and date
        df_antibiotics = df_antibiotics.sort_values(by=['stay_id', 'administration_time'])

        """"
        The antibiotics vector needs to be a vector of 1s and 0s, 1 if the patient received antibiotics at that time, 0 otherwise,
        the time used needs to match the one from the same stay, i.e. same range (start to finish) of time as the patient stay
        The vector of antibiotics is filled with 1s in between two antibiotics administrations first and last time, 
        except if the time between two administrations is more than 24 hours, then it is considered two different administrations
        """
        #Create new df that matches the range, for each stay_id, create a full time range from first administration to last administration, per stay_id
        full_range_dfs = []
        for stay_id, stay_antibios_df in df_antibiotics.groupby('stay_id'):
            #Find matching vital date start and end for corresponding stay_id
            staymatch_df_id = df_id[df_id['stay_id'] == stay_id]

            stay_start = staymatch_df_id['date'].min()
            stay_end = staymatch_df_id['date'].max()

            # Generate a full time index from min to max timestamp
            full_time_range = pd.date_range(
                start=stay_start,
                end=stay_end,
                freq='1min' # 1 minute frequency #TODO: Make frequency a parameter
            )

            #Create a new dataframe with the full time range
            administration_complete_df = pd.DataFrame({"date": full_time_range, "antibiotics": np.nan})
            administration_complete_df['stay_id'] = stay_id

            #In stay_antibios_df, modulo it out to our current freq to make sure it will always match a time in the full_time_range
            stay_antibios_df['administration_time'] = stay_antibios_df['administration_time'].dt.floor('1min') #TODO: Make frequency a parameter

            #Find the start and end time of each administrations (and considering gaps), fill with 1s in the new dataframe based on this
            prev_time = stay_antibios_df['administration_time'].iloc[0]

            if not pd.isnull(prev_time):
                for index, row in stay_antibios_df.iterrows():
                    #Find first and last time of administration
                    curr_time = row['administration_time']
                    #Check if the gap is less than GAP_CONST
                    if (curr_time - prev_time).seconds/60 < GAP_CONST:
                        #Then fill in the administration_complete_df with 1s between prev_time and curr_time
                        #Always also fill it with the MARGIN_CONST minutes before an actual point too
                        administration_complete_df.loc[(administration_complete_df['date'] >= prev_time - pd.Timedelta(minutes=MARGIN_CONST)) & (administration_complete_df['date'] <= curr_time), 'antibiotics'] = 1
                    else:
                        #Then fill in the administration_complete_df with 1s at curr_time, still with the margin
                        administration_complete_df.loc[(administration_complete_df['date'] >= curr_time - pd.Timedelta(minutes=MARGIN_CONST)) & (administration_complete_df['date'] <= curr_time), 'antibiotics'] = 1
                        
                    #Update prev_time
                    prev_time = curr_time

            #Fill the NaN values with 0
            administration_complete_df = administration_complete_df.fillna(0)

            #Append the new dataframe to the list
            full_range_dfs.append(administration_complete_df)

        #Concatenate all the dataframes
        df_antibiotics = pd.concat(full_range_dfs)

        #Sort again because im scared
        df_antibiotics = df_antibiotics.sort_values(by=['stay_id', 'date'])

        # #DEBUG: Print df_antibiotics columns and head
        # print("Antibiotics columns",df_antibiotics.columns)
        # print("Antibiotics head",df_antibiotics.head())

        # #DEBUG: Iterate over all patients and print the number of 1s in the antibiotics vector
        # for stay_id, stay_antibios_df in df_antibiotics.groupby('stay_id'):
        #     print(f"Stay_id: {stay_id}, Number of 1s in antibiotics vector: {stay_antibios_df['antibiotics'].sum()}")
        #     #And total number of rows
        #     print(f"Total number of rows: {len(stay_antibios_df)}")

        self.data_antibiotics = df_antibiotics['antibiotics'].values[border1:border2]

        #del df_antibiotics to save memory
        del df_antibiotics
        gc.collect()

        # --- ANTIBIOTICS END ---
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        seq_x_id = self.data_id[s_begin:s_end]
        seq_y_id = self.data_id[r_begin:r_end]

        #Get the static data for the stay_id
        seq_static = self.data_static[seq_y_id[-1][0]]

        #Get the antibiotics data, matching the y range
        seq_antibiotics = self.data_antibiotics[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_id, seq_y_id, seq_static, seq_antibiotics
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

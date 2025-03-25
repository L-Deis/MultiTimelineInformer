import torch
import pandas as pd
import numpy as np
from functools import partial
from utils.timefeatures import time_features
from collections import Counter


# TODO for this to work we need to work with shuffle = false, and sort on category_id, check!
#  so sorting on: person_id and then date?
# TODO check if dimensions are accurate (annotated below but also: is one input a list of timestamps)
def categorical_collate(batches, timeenc, freq):
    """
    Collate function that ensures all timesteps within each input and output are consistent with the same person_id.
    Handles cases where inputs switch between person_ids and salvages the part that matches the output.
    Args:
        batch: A list of tuples, where each tuple contains (inputs, outputs).
               - inputs: Tensor of shape (seq_len, features) for model input.
               - outputs: Tensor of shape (pred_len, features) for ground truth.
        person_id_index: The index in the input and output tensors that represents the person_id.
    Returns:
        A batch of tensors (inputs, outputs) with consistent person_ids,
        partially zero-padded inputs for salvageable sequences
        or all zeros for inconsistent data.
    """
    try:

        if len(batches) == 0:  # Early exit for empty batch
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor(
                []), torch.tensor([]), torch.tensor([])

        # Initialize containers for valid inputs and outputs
        valid_inputs = []
        valid_outputs = []
        valid_ids = []
        valid_statics = []
        valid_antibiotics = []

        for seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_id, seq_y_id, seq_static, seq_antibiotics in batches:
            # Dimensions of inputs and outputs
            # inputs: (seq_len, features)
            # outputs: (pred_len, features)

            # Extract person_ids for inputs and outputs
            input_person_ids = seq_x_id  # Shape: (seq_x_len,2)
            output_person_ids = seq_y_id  # Shape: (seq_y_len,2)

            #DEBUG: Print the shapes of each seq
            # print("--- SEQ SHAPES ---")
            # print(f"seq_x: {seq_x.shape}")
            # print(f"seq_y: {seq_y.shape}")
            # print(f"seq_x_mark: {seq_x_mark.shape}")
            # print(f"seq_y_mark: {seq_y_mark.shape}")
            # print(f"input_person_ids: {input_person_ids.shape}")
            # print(f"output_person_ids: {output_person_ids.shape}")

            # Check if person_id is consistent within inputs and outputs
            input_consistent = np.all(input_person_ids[:, 0] == input_person_ids[0, 0])  # True if all input IDs match
            output_consistent = np.all(output_person_ids[:, 0] == output_person_ids[0, 0])  # True if all output IDs match
            input_output_match = input_person_ids[-1, 0] == output_person_ids[0, 0]

            if input_consistent and output_consistent and input_output_match:
                # Case 1: Fully consistent input and output
                valid_inputs.append((seq_x, seq_x_mark))
                valid_outputs.append((seq_y, seq_y_mark))
                valid_ids.append((seq_x_id, seq_y_id))
                valid_statics.append(seq_static)
                valid_antibiotics.append(seq_antibiotics)
            elif output_consistent and input_output_match:
                # Case 2: Outputs are consistent, but inputs transition between person_ids
                # The input does contain at least 1 datapoint for the person_id in the output
                # Replace inconsistent parts of inputs with zeros

                # Identify the person_id from outputs
                target_person_id = output_person_ids[0, 0]

                # Create a mask for valid input rows
                valid_mask = input_person_ids[:, 0] == target_person_id  # Shape: (seq_len,)

                # Zero out rows in inputs where person_id doesn't match outputs
                salvageable_inputs = seq_x.copy()  # Clone to avoid modifying original data
                salvageable_inputs[~valid_mask] = 0  # Replace invalid rows with zeros

                #Fill invalid rows from seq_y_mark with new fake datetime values (every 1 min) and then use time features to get the mark
                timeline = seq_x_id[:, 1].copy()
                timeline = pd.DataFrame({'date': pd.to_datetime(timeline)})

                #Find the first valid timestamp
                # first_valid = timeline[valid_mask][0] #deosnt work for pandas
                first_valid = timeline.loc[valid_mask, 'date'].iloc[0]  #works for pandas
                #Get index of first valid timestamp
                first_valid_index = np.where(valid_mask)[0][0]

                #1 minutes in pandas datetime
                delta_time = pd.Timedelta(minutes=1)  #TODO: Make it a parameter that can change based on arguments
                #Fix every invalid timestamps based on their index, 1 min offset each, based on the first valid timestamp
                math_vec = np.arange(-first_valid_index, 0)
                timeline.loc[~valid_mask, 'date'] = first_valid + delta_time * math_vec

                #Get the time features
                salvageable_inputs_mark = time_features(timeline, timeenc, freq)

                #Salvageable ids
                salvageable_x_id = input_person_ids.copy()
                salvageable_x_id[~valid_mask, 0] = seq_y_id[0, 0]  #Replace invalid ids with the target id
                salvageable_x_id[~valid_mask, 1] = timeline.loc[
                    ~valid_mask, 'date'].values  #Replace invalid timestamps with the fixed ones

                #Append the mark to the inputs
                valid_inputs.append((salvageable_inputs, salvageable_inputs_mark))
                valid_outputs.append((seq_y, seq_y_mark))
                valid_ids.append((salvageable_x_id, seq_y_id))
                valid_statics.append(seq_static)
                valid_antibiotics.append(seq_antibiotics)
            else:
                # Case 3: output has inconsistencies
                # Unsalvageable data; the difference between predicted output and real output will never match
                # Remove patient from this batch
                pass

        if not valid_inputs:
            # Return tensors with zero-size or a custom exception
            return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor(
                []), torch.tensor([]), torch.tensor([])

        # Stack the valid inputs and outputs into tensors
        # Reconstruction of the original shape
        valid_seq_x = torch.stack(
            [torch.tensor(seq[0]) if isinstance(seq[0], np.ndarray) else seq[0] for seq in valid_inputs])
        valid_seq_x_mark = torch.stack(
            [torch.tensor(seq[1]) if isinstance(seq[1], np.ndarray) else seq[1] for seq in valid_inputs])
        valid_seq_y = torch.stack(
            [torch.tensor(seq[0]) if isinstance(seq[0], np.ndarray) else seq[0] for seq in valid_outputs])
        valid_seq_y_mark = torch.stack(
            [torch.tensor(seq[1]) if isinstance(seq[1], np.ndarray) else seq[1] for seq in valid_outputs])
        valid_seq_x_id = np.stack([seq[0] for seq in valid_ids])
        valid_seq_y_id = np.stack([seq[1] for seq in valid_ids])
        # valid_statics = np.stack(valid_statics)
        # valid_antibiotics = np.stack(valid_antibiotics)
        # valid_statics = torch.stack([torch.tensor(seq) if isinstance(seq, np.ndarray) else seq for seq in valid_statics])
        valid_statics = torch.stack([
            seq if isinstance(seq, torch.Tensor) else torch.tensor(seq)
            for seq in valid_statics
        ])

        # ___________________ANTIBIOTICS________________________
        # Check and correct antibiotics length
        # Convert to tensors
        abx_tensors = [
            abx if isinstance(abx, torch.Tensor) else torch.tensor(abx)
            for abx in valid_antibiotics
        ]

        # Determine most common length (mode)
        lengths = [abx.shape[0] for abx in abx_tensors]
        length_counts = Counter(lengths)
        expected_len = length_counts.most_common(1)[0][0]

        # Fix length if necessary
        fixed_antibiotics = []
        for i, abx in enumerate(abx_tensors):
            current_len = abx.shape[0]
            if current_len == expected_len:
                fixed_antibiotics.append(abx)
            elif current_len > expected_len:
                # Trim from the start, keep the last `expected_len` entries
                trimmed = abx[-expected_len:]
                fixed_antibiotics.append(trimmed)
                print("WARNING: There were inconsistent antibiotics vectors in a batch!")
                print("The antibiotics vector has been fixed as far as possible.")
                print("The antibiotics vector was TOO LONG")
                # TODO find out why and how that happens
                print("\n--- Debug: First Sample Snapshot ---")
                print("valid_seq_x[0]:", valid_seq_x[0])
                print("valid_seq_x_mark[0]:", valid_seq_x_mark[0])
                print("valid_seq_y[0]:", valid_seq_y[0])
                print("valid_seq_y_mark[0]:", valid_seq_y_mark[0])
                print("valid_seq_x_id[0]:", valid_seq_x_id[0])
                print("valid_seq_y_id[0]:", valid_seq_y_id[0])
                print("valid_statics[0]:", valid_statics[0])
                print("valid_antibiotics[0]:", valid_antibiotics[0])
                print("--- 🔎 End Debug ---\n")

            elif current_len < expected_len:
                # Pad at the end with the last available value
                pad_len = expected_len - current_len
                pad_value = abx[-1].item() if current_len > 0 else 0.0  # fallback if empty
                pad_tensor = torch.full((pad_len,), pad_value, dtype=abx.dtype)
                padded = torch.cat([abx, pad_tensor])
                fixed_antibiotics.append(padded)
                print("WARNING: There were inconsistent antibiotics vectors in a batch!")
                print("The antibiotics vector has been fixed as far as possible.")
                print("The antibiotics vector was TOO SHORT")
                # TODO find out why and how that happens
                print("\n--- Debug: First Sample Snapshot ---")
                print("valid_seq_x[0]:", valid_seq_x[0])
                print("valid_seq_x_mark[0]:", valid_seq_x_mark[0])
                print("valid_seq_y[0]:", valid_seq_y[0])
                print("valid_seq_y_mark[0]:", valid_seq_y_mark[0])
                print("valid_seq_x_id[0]:", valid_seq_x_id[0])
                print("valid_seq_y_id[0]:", valid_seq_y_id[0])
                print("valid_statics[0]:", valid_statics[0])
                print("valid_antibiotics[0]:", valid_antibiotics[0])
                print("--- 🔎 End Debug ---\n")

        # Step 4: Stack
        valid_antibiotics = torch.stack(fixed_antibiotics)

        return valid_seq_x, valid_seq_y, valid_seq_x_mark, valid_seq_y_mark, valid_seq_x_id, valid_seq_y_id, valid_statics, valid_antibiotics
    except Exception as error:
        print("WARNING: The collate_fn encountered an error and the corresponding batch has been dropped!")
        print(error)
        # TODO figure out how and why that can happen
        return torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor([]), torch.tensor(
            []), torch.tensor([]), torch.tensor([])

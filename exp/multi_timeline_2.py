import torch
from functools import partial

# TODO for this to work we need to work with shuffle = false, and sort on category_id, check!
#  so sorting on: person_id and then date?
# TODO check if dimensions are accurate (annotated below but also: is one input a list of timestamps)
def categorical_collate(batch, person_id_index):
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
    if len(batch) == 0:  # Early exit for empty batch
        return torch.tensor([]), torch.tensor([])

    # Initialize containers for valid inputs and outputs
    valid_inputs = []
    valid_outputs = []

    for inputs, outputs in batch:
        # Dimensions of inputs and outputs
        # inputs: (seq_len, features)
        # outputs: (pred_len, features)

        # Extract person_ids for inputs and outputs
        input_person_ids = inputs[:, person_id_index]  # Shape: (seq_len,)
        output_person_ids = outputs[:, person_id_index]  # Shape: (pred_len,)

        # Check if person_id is consistent within inputs and outputs
        input_consistent = torch.all(input_person_ids == input_person_ids[0])  # True if all input IDs match
        output_consistent = torch.all(output_person_ids == output_person_ids[0])  # True if all output IDs match
        input_output_match = input_person_ids[-1] == output_person_ids[0]

        if input_consistent and output_consistent and input_output_match:
            # Case 1: Fully consistent input and output
            valid_inputs.append(inputs)
            valid_outputs.append(outputs)
        elif output_consistent and input_output_match:
            # Case 2: Outputs are consistent, but inputs transition between person_ids
            # The input does contain at least 1 datapoint for the person_id in the output
            # Replace inconsistent parts of inputs with zeros

            # Identify the person_id from outputs
            target_person_id = output_person_ids[0]

            # Create a mask for valid input rows
            valid_mask = input_person_ids == target_person_id  # Shape: (seq_len,)

            # Zero out rows in inputs where person_id doesn't match outputs
            salvageable_inputs = inputs.clone()  # Clone to avoid modifying original data
            salvageable_inputs[~valid_mask] = 0  # Replace invalid rows with zeros

            valid_inputs.append(salvageable_inputs)
            valid_outputs.append(outputs)
        else:
            # Case 3: Fully inconsistent input or output
            # Replace both inputs and outputs with zero-padded placeholders
            seq_len, input_features = inputs.shape
            pred_len, output_features = outputs.shape

            padded_inputs = torch.zeros((seq_len, input_features))  # Shape: (seq_len, features)
            padded_outputs = torch.zeros((pred_len, output_features))  # Shape: (pred_len, features)

            valid_inputs.append(padded_inputs)
            valid_outputs.append(padded_outputs)

    # Stack the valid inputs and outputs into tensors
    # valid_inputs: (batch_size, seq_len, features)
    # valid_outputs: (batch_size, pred_len, features)
    valid_inputs = torch.stack(valid_inputs)
    valid_outputs = torch.stack(valid_outputs)

    return valid_inputs, valid_outputs
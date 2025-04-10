import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataFrame(object):

    """Minimal pd.DataFrame analog for handling n-dimensional numpy matrices with additional
    support for shuffling, batching, and train/test splitting.

    Args:
        columns: List of names corresponding to the matrices in data.
        data: List of n-dimensional data matrices ordered in correspondence with columns.
            All matrices must have the same leading dimension. Data can also be fed a list of
            instances of np.memmap, in which case RAM usage can be limited to the size of a
            single batch.
    """

    def __init__(self, columns, data):
        if not columns or not data:
             raise ValueError("Columns and data cannot be empty")
        if len(columns) != len(data):
             raise ValueError(f'Columns length ({len(columns)}) does not match data length ({len(data)})')

        # Check if data contains numpy arrays or list of arrays (ragged)
        is_ragged = any(isinstance(d, np.ndarray) and d.dtype == object for d in data)

        if is_ragged:
            # If ragged, length check might be tricky. Assume first dimension consistency.
            try:
                lengths = [len(mat) for mat in data]
                if len(set(lengths)) != 1:
                    raise ValueError(f'All data elements must have same first dimension length. Found lengths: {lengths}')
                self.length = lengths[0]
            except TypeError:
                 raise TypeError("Cannot determine length of all elements in data. Ensure they are list-like or numpy arrays.")

        else:
            # If not ragged, check first dimension shape consistency
            shapes = [mat.shape for mat in data if isinstance(mat, np.ndarray)]
            if not shapes:
                 raise ValueError("Data contains no numpy arrays to determine shape")

            lengths = [s[0] for s in shapes]
            if len(set(lengths)) != 1:
                 raise ValueError(f'All numpy arrays in data must have same first dimension. Found shapes: {shapes}')
            self.length = lengths[0]


        self.columns = columns
        self.data = data # Store potentially ragged data
        # Create dict view, handling potential non-numpy data carefully if needed
        self.dict = dict(zip(self.columns, self.data))
        self.idx = np.arange(self.length)

    def shapes(self):
        # Report shapes, indicating raggedness if applicable
        """Returns a pandas Series showing the shape of each data column."""
        shape_dict = {}
        for col, mat in zip(self.columns, self.data):
             if isinstance(mat, np.ndarray):
                  if mat.dtype == object:
                       shape_dict[col] = f"ragged, N={len(mat)}"
                  else:
                       shape_dict[col] = mat.shape
             elif isinstance(mat, list):
                  shape_dict[col] = f"list, N={len(mat)}"
             else:
                  shape_dict[col] = type(mat)
        return pd.Series(shape_dict)


    def dtypes(self):
        # Report dtypes
        """Returns a pandas Series showing the dtype of each data column."""
        dtype_dict = {}
        for col, mat in zip(self.columns, self.data):
             if isinstance(mat, np.ndarray):
                  dtype_dict[col] = mat.dtype
             else:
                  dtype_dict[col] = type(mat)
        return pd.Series(dtype_dict)

    def shuffle(self):
        """Shuffles the row order in-place for subsequent batching or iteration."""
        np.random.shuffle(self.idx)

    def train_test_split(self, train_size, random_state=np.random.randint(1000), stratify=None):
        if self.length == 0:
             print("Warning: DataFrame is empty, cannot perform train/test split.")
             # Return two empty DataFrames with the same columns
             return DataFrame(copy.copy(self.columns), [[] for _ in self.data]), \
                    DataFrame(copy.copy(self.columns), [[] for _ in self.data])

        stratify_data = self.dict[stratify][self.idx] if stratify else None

        train_idx, test_idx = train_test_split(
            self.idx,
            train_size=train_size,
            random_state=random_state,
            stratify=stratify_data
        )

        # Define a helper to safely index potentially ragged data
        def safe_index(mat, indices):
            if isinstance(mat, np.ndarray) and mat.dtype == object:
                 # For ragged array, index element by element
                 return np.array([mat[i] for i in indices], dtype=object)
            elif isinstance(mat, np.ndarray):
                 # Standard numpy indexing
                 return mat[indices]
            elif isinstance(mat, list):
                 # Standard list indexing
                 return [mat[i] for i in indices]
            else:
                 # Handle other types if necessary, or raise error
                 raise TypeError(f"Unsupported data type for indexing: {type(mat)}")


        train_df = DataFrame(copy.copy(self.columns), [safe_index(mat, train_idx) for mat in self.data])
        test_df = DataFrame(copy.copy(self.columns), [safe_index(mat, test_idx) for mat in self.data])
        return train_df, test_df

    def batch_generator(self, batch_size, shuffle=True, num_epochs=10000, allow_smaller_final_batch=False):
        if self.length == 0:
             print("Warning: DataFrame is empty, batch generator yields nothing.")
             return # Stop iteration immediately

        epoch_num = 0
        while epoch_num < num_epochs:
            if shuffle:
                self.shuffle()

            for i in range(0, self.length, batch_size): # Corrected range step
                batch_idx = self.idx[i: min(i + batch_size, self.length)] # Handle end of data

                # Skip if final batch is smaller than requested and not allowed
                if not allow_smaller_final_batch and len(batch_idx) < batch_size and i + batch_size < self.length :
                     # This condition was slightly off, it should allow the *very last* batch
                     # Let's rethink: only skip if it's *not* the final batch and too small
                     # Simpler: Check length *after* slicing
                     pass # Let the check below handle it

                if len(batch_idx) == 0: continue # Skip empty index list

                if not allow_smaller_final_batch and len(batch_idx) < batch_size:
                    # If we are here, it must be the *final* batch and it's smaller.
                    # If allow_smaller_final_batch is False, we break the loop *before* yielding.
                    break


                # Define safe indexing helper again (or make it a method)
                def safe_index(mat, indices):
                    if isinstance(mat, np.ndarray) and mat.dtype == object:
                        return np.array([mat[i] for i in indices], dtype=object)
                    elif isinstance(mat, np.ndarray):
                        # Perform copy during indexing for batches to avoid view issues
                        return mat[indices].copy()
                    elif isinstance(mat, list):
                        return [mat[i] for i in indices] # List comprehension creates copies
                    else:
                        raise TypeError(f"Unsupported data type for indexing: {type(mat)}")

                yield DataFrame(
                    columns=copy.copy(self.columns),
                    data=[safe_index(mat, batch_idx) for mat in self.data]
                )

            epoch_num += 1

    def iterrows(self):
        if self.length == 0: return # Handle empty case
        for i in self.idx:
            yield self[i] # Use __getitem__

    def mask(self, mask):
        """Returns a new DataFrame containing only rows where mask is True."""
        if len(mask) != self.length:
            raise ValueError(f"Mask length ({len(mask)}) must match DataFrame length ({self.length})")
        masked_idx = self.idx[mask]
        # Use safe indexing helper
        def safe_index(mat, indices):
            if isinstance(mat, np.ndarray) and mat.dtype == object:
                return np.array([mat[i] for i in indices], dtype=object)
            elif isinstance(mat, np.ndarray):
                return mat[indices]
            elif isinstance(mat, list):
                return [mat[i] for i in indices]
            else:
                raise TypeError(f"Unsupported data type for indexing: {type(mat)}")
        return DataFrame(copy.copy(self.columns), [safe_index(mat, masked_idx) for mat in self.data])


    def concat(self, other_df):
        """Concatenates another DataFrame vertically."""
        if self.columns != other_df.columns:
            raise ValueError("DataFrames must have the same columns to concatenate.")

        new_data = []
        for col in self.columns:
            self_mat = self[col]
            other_mat = other_df[col]

            # Handle concatenation based on type
            if isinstance(self_mat, np.ndarray) and isinstance(other_mat, np.ndarray):
                 # Special handling if one or both are object arrays (ragged)
                 if self_mat.dtype == object or other_mat.dtype == object:
                      new_data.append(np.concatenate([self_mat, other_mat], axis=0).astype(object))
                 else:
                      new_data.append(np.concatenate([self_mat, other_mat], axis=0))
            elif isinstance(self_mat, list) and isinstance(other_mat, list):
                 new_data.append(self_mat + other_mat)
            else:
                 raise TypeError(f"Cannot concatenate types {type(self_mat)} and {type(other_mat)} for column '{col}'")

        return DataFrame(copy.copy(self.columns), new_data)


    def items(self):
        return self.dict.items()

    def __iter__(self):
        # Iterate over column names, like pandas
        return iter(self.columns)

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if isinstance(key, str):
            # Return column data
            if key not in self.dict:
                 raise KeyError(f"Column '{key}' not found.")
            return self.dict[key]

        elif isinstance(key, int):
             # Return row data as a dictionary (like a Series conceptually)
             if key < 0 or key >= self.length:
                  raise IndexError(f"Index {key} is out of bounds for length {self.length}")
             actual_index = self.idx[key] # Use shuffled index if applicable
             row_data = {}
             for col, mat in zip(self.columns, self.data):
                  try:
                       row_data[col] = mat[actual_index]
                  except IndexError:
                       # This might happen with ragged arrays if not handled carefully
                       raise IndexError(f"Failed to access index {actual_index} for column '{col}'.")
                  except TypeError:
                       # Handle cases where mat might not be indexable (e.g., scalar?)
                       raise TypeError(f"Data in column '{col}' is not indexable.")
             return pd.Series(row_data) # Return as pandas Series for convenience

        elif isinstance(key, slice):
            # Return a slice of the DataFrame
            sliced_idx = self.idx[key]
            def safe_index(mat, indices):
                 if isinstance(mat, np.ndarray) and mat.dtype == object: return np.array([mat[i] for i in indices], dtype=object)
                 elif isinstance(mat, np.ndarray): return mat[indices]
                 elif isinstance(mat, list): return [mat[i] for i in indices]
                 else: raise TypeError(f"Unsupported data type for indexing: {type(mat)}")
            return DataFrame(copy.copy(self.columns), [safe_index(mat, sliced_idx) for mat in self.data])

        elif isinstance(key, (list, np.ndarray)):
             # Return a subset of rows based on list/array of indices or boolean mask
             if isinstance(key, np.ndarray) and key.dtype == bool:
                 # Boolean mask
                 if len(key) != self.length: raise IndexError("Boolean mask length must match DataFrame length.")
                 return self.mask(key)
             else:
                 # List/array of integer indices
                 indices = np.array(key) # Convert list to array
                 if not np.issubdtype(indices.dtype, np.integer): raise IndexError("Index list/array must contain integers.")
                 # Use int indices on self.idx to respect shuffling
                 actual_indices = self.idx[indices]
                 def safe_index(mat, indices):
                     if isinstance(mat, np.ndarray) and mat.dtype == object: return np.array([mat[i] for i in indices], dtype=object)
                     elif isinstance(mat, np.ndarray): return mat[indices]
                     elif isinstance(mat, list): return [mat[i] for i in indices]
                     else: raise TypeError(f"Unsupported data type for indexing: {type(mat)}")
                 return DataFrame(copy.copy(self.columns), [safe_index(mat, actual_indices) for mat in self.data])

        else:
            raise TypeError(f"Unsupported key type for __getitem__: {type(key)}")


    def __setitem__(self, key, value):
        """Adds or replaces a column."""
        if not isinstance(key, str):
            raise ValueError("Key for setting item must be a string (column name).")

        # Check length consistency of the value being assigned
        try:
             if len(value) != self.length:
                  raise ValueError(f"Value length ({len(value)}) must match DataFrame length ({self.length}).")
        except TypeError:
             raise TypeError("Value being assigned must have a defined length.")


        if key not in self.columns:
            self.columns.append(key)
            self.data.append(value)
        else:
            # Replace existing column data
            col_index = self.columns.index(key)
            self.data[col_index] = value

        # Update the dictionary view
        self.dict[key] = value

    def __repr__(self):
        return f"DataFrame(columns={self.columns}, length={self.length})"
import os
from xml.etree import ElementTree
import numpy as np
import drawing
import argparse

BASE_RAW_DIR = 'data/raw'
OUTPUT_DIR = 'data/processed'
BLACKLIST_FILE = 'data/blacklist.npy'


def get_stroke_sequence(filename):
    """
    Parses an XML file containing stroke data, processes it, and returns normalized offsets.
    Args:
        filename (str): Path to the lineStrokes XML file.
    Returns:
        np.ndarray or None: Array of stroke offsets [dx, dy, eos] truncated to MAX_STROKE_LEN,
                          or None if parsing/processing fails.
    """
    try:
        tree = ElementTree.parse(filename).getroot()
        stroke_sets = [i for i in tree if i.tag == 'StrokeSet']
        if not stroke_sets:
            # print(f"Warning: No 'StrokeSet' found in {filename}")
            return None
        strokes = stroke_sets[0]

        coords = []
        for stroke in strokes:
            num_points = len(stroke)
            for i, point in enumerate(stroke):
                try:
                    x = int(point.attrib['x'])
                    y = -1 * int(point.attrib['y']) # Invert y-axis
                    eos = int(i == num_points - 1)
                    coords.append([x, y, eos])
                except (KeyError, ValueError):
                    # print(f"Warning: Skipping invalid point data in {filename}, stroke {strokes.index(stroke)}")
                    continue

        if not coords:
            # print(f"Warning: No valid coordinates extracted from {filename}")
            return None

        coords = np.array(coords, dtype=np.float32) # Shape (N, 3)

        # --- Apply processing steps ---

        # 1. Align: Pass only XY coordinates, then re-add EOS
        original_eos = coords[:, 2:3] # Shape (N, 1)
        aligned_xy = drawing.align(coords[:, :2]) # Pass (N, 2) to align
        # Check if alignment returned valid data
        if aligned_xy is None or len(aligned_xy) != len(coords):
             # print(f"Warning: Alignment failed or changed length for {filename}")
             return None # Treat as error if length changes
        coords = np.concatenate([aligned_xy, original_eos], axis=1) # Recombine to (N, 3)

        # 2. Denoise: Pass (N, 3) array
        coords = drawing.denoise(coords)
        if coords is None or len(coords) == 0:
            # print(f"Warning: Coordinates became empty after denoising for {filename}")
            return None

        # 3. Convert to Offsets
        offsets = drawing.coords_to_offsets(coords)
        if offsets is None or len(offsets) == 0:
             # print(f"Warning: Offsets became empty after conversion for {filename}")
             return None

        # 4. Truncate
        offsets = offsets[:drawing.MAX_STROKE_LEN]

        # 5. Normalize
        offsets = drawing.normalize(offsets)

        return offsets

    except ElementTree.ParseError:
        # print(f"Warning: Failed to parse XML file {filename}")
        return None
    except Exception as e:
        # print(f"Warning: Unexpected error processing strokes for {filename}: {e}")
        import traceback
        # traceback.print_exc() # Uncomment for detailed trace
        return None


def get_ascii_sequences(filename):
    """
    Reads an ASCII transcription file and returns encoded character sequences.
    Args:
        filename (str): Path to the ASCII transcription file.
    Returns:
        list or None: A list of encoded numpy arrays (int32), truncated to MAX_CHAR_LEN,
                     or None if reading/parsing fails.
    """
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        sections = content.replace(r'%%%%%%%%%%%', '\n').split('\n')
        try:
            csr_index = sections.index('CSR:')
            lines = sections[csr_index + 2:]
        except ValueError:
            # print(f"Warning: 'CSR:' marker not found in {filename}")
            return None

        processed_lines = []
        for line in lines:
            stripped_line = line.strip()
            if stripped_line:

                encoded_line = drawing.encode_ascii(stripped_line)
                truncated_line = encoded_line[:drawing.MAX_CHAR_LEN]
                processed_lines.append(truncated_line)

        # Return None if no valid lines were processed (e.g., CSR section exists but is empty)
        return processed_lines if processed_lines else None

    except FileNotFoundError:
        # print(f"Warning: ASCII file not found: {filename}")
        return None
    except Exception as e:
        # print(f"Warning: Error processing ASCII file {filename}: {e}")
        return None


def collect_data(base_raw_dir):
    """
    Walks the raw data directory, matches transcriptions with strokes,
    and collects file paths, encoded transcriptions, and writer IDs.
    Args:
        base_raw_dir (str): Path to the base directory containing 'ascii', 'lineStrokes', etc.
    Returns:
        tuple: (stroke_fnames, transcriptions, writer_ids) where each is a list.
    """
    ascii_dir = os.path.join(base_raw_dir, 'ascii')
    line_strokes_base_dir = os.path.join(base_raw_dir, 'lineStrokes')
    original_base_dir = os.path.join(base_raw_dir, 'original')

    if not os.path.isdir(ascii_dir):
        raise NotADirectoryError(f"ASCII directory not found: {ascii_dir}")

    fnames = []
    for dirpath, dirnames, filenames in os.walk(ascii_dir):
        for filename in filenames:
            if filename.endswith('.txt') and not filename.startswith('.'):
                fnames.append(os.path.join(dirpath, filename))

    # Load blacklist
    blacklist = set()
    if os.path.exists(BLACKLIST_FILE):
        try:
            blacklist_data = np.load(BLACKLIST_FILE, allow_pickle=True)
            blacklist = set(blacklist_data)
            print(f"Loaded {len(blacklist)} entries from blacklist file: {BLACKLIST_FILE}")
        except Exception as e:
            print(f"Warning: Failed to load or parse blacklist file {BLACKLIST_FILE}: {e}")
    else:
        print(f"Info: Blacklist file not found: {BLACKLIST_FILE}. No samples will be excluded.")


    stroke_fnames, transcriptions, writer_ids = [], [], []
    processed_count = 0
    skipped_blacklist = 0
    skipped_missing_files = 0
    skipped_processing_error = 0
    skipped_mismatch = 0

    for i, ascii_fname in enumerate(fnames):
        if (i + 1) % 50 == 0 or i == 0: # Print progress every 50 files and for the first one
             print(f"Processing ASCII file {i+1}/{len(fnames)}: {ascii_fname}")

        if 'z01-000z.txt' in ascii_fname: continue # Skip known problematic file

        relative_path = os.path.relpath(os.path.dirname(ascii_fname), ascii_dir)
        line_stroke_dir = os.path.join(line_strokes_base_dir, relative_path)
        base_fname = os.path.splitext(os.path.basename(ascii_fname))[0]
        line_stroke_fname_prefix = base_fname + '-'
        original_dir = os.path.join(original_base_dir, relative_path)
        last_letter_match = os.path.splitext(base_fname)[0][-1]
        last_letter = last_letter_match if last_letter_match.isalpha() else ''
        original_xml = os.path.join(original_dir, 'strokes' + last_letter + '.xml')

        if not os.path.isdir(line_stroke_dir):
            skipped_missing_files += 1
            continue

        writer_id = 0
        if os.path.exists(original_xml):
            try:
                tree = ElementTree.parse(original_xml).getroot()
                general = tree.find('General')
                if general is not None and len(general) > 0 and 'writerID' in general[0].attrib:
                    writer_id = int(general[0].attrib['writerID'])
            except (ElementTree.ParseError, ValueError, IndexError): pass # Keep default writer_id=0

        ascii_sequences = get_ascii_sequences(ascii_fname)
        if ascii_sequences is None: # Check for None specifically
            skipped_processing_error += 1
            continue

        try:
            line_stroke_files_in_dir = sorted([
                f for f in os.listdir(line_stroke_dir)
                if f.startswith(line_stroke_fname_prefix) and f.endswith('.xml')
            ])
        except FileNotFoundError:
             skipped_missing_files += len(ascii_sequences)
             continue

        if not line_stroke_files_in_dir:
            skipped_missing_files += len(ascii_sequences)
            continue

        if len(ascii_sequences) != len(line_stroke_files_in_dir):
            skipped_mismatch += len(ascii_sequences) # Count as mismatch skip
            continue

        for ascii_seq, line_stroke_fname in zip(ascii_sequences, line_stroke_files_in_dir):
            if line_stroke_fname in blacklist:
                skipped_blacklist += 1
                continue
            stroke_fnames.append(os.path.join(line_stroke_dir, line_stroke_fname))
            transcriptions.append(ascii_seq)
            writer_ids.append(writer_id)
            processed_count += 1

    print("-" * 30)
    print(f"Finished collecting data.")
    print(f"Total ASCII files processed: {len(fnames)}")
    print(f"Successfully matched items found: {processed_count}")
    print(f"Skipped (blacklist): {skipped_blacklist}")
    print(f"Skipped (missing files): {skipped_missing_files}")
    print(f"Skipped (ASCII/XML mismatch): {skipped_mismatch}")
    print(f"Skipped (processing error): {skipped_processing_error}")
    total_skipped = skipped_blacklist + skipped_missing_files + skipped_mismatch + skipped_processing_error
    print(f"Total items skipped: {total_skipped}")
    print("-" * 30)

    return stroke_fnames, transcriptions, writer_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prepare IAM On-Line Handwriting Data for RNN training.")
    parser.add_argument('--raw_dir', type=str, default=BASE_RAW_DIR, help="Base directory containing raw data subdirectories (ascii, lineStrokes, original).")
    parser.add_argument('--out_dir', type=str, default=OUTPUT_DIR, help="Directory to save processed .npy files.")
    parser.add_argument('--debug', action='store_true', help="Enable debug prints during validation.")
    args = parser.parse_args()

    BASE_RAW_DIR = args.raw_dir
    OUTPUT_DIR = args.out_dir
    DEBUG_MODE = args.debug

    print('Traversing data directory...')
    stroke_fnames, transcriptions, writer_ids = collect_data(BASE_RAW_DIR)

    if not stroke_fnames:
        print("Error: No valid data collected. Please check raw data paths and format.")
        exit()

    print(f'\nProcessing {len(stroke_fnames)} collected items into numpy arrays...')

    x_list, x_len_list, c_list, c_len_list, w_id_list = [], [], [], [], []
    valid_indices = []
    invalid_reasons = {"stroke": 0, "norm": 0, "char": 0} # Count reasons for invalidity

    total_items = len(stroke_fnames)
    for i, (stroke_fname, c_i, w_id_i) in enumerate(zip(stroke_fnames, transcriptions, writer_ids)):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"Processing item {i+1}/{total_items}...")

        x_i = get_stroke_sequence(stroke_fname)

        # --- Validation ---
        is_valid = True
        reason = ""

        # 1. Check if stroke processing succeeded
        if x_i is None or len(x_i) == 0:
            is_valid = False
            reason = "stroke_processing_failed"
            invalid_reasons["stroke"] += 1

        # 2. Check stroke norm (only if stroke is valid so far)
        if is_valid:
             try:
                 norms = np.linalg.norm(x_i[:, :2], axis=1)
                 if np.any(norms > 60):
                     is_valid = False
                     reason = "large_stroke_norm"
                     invalid_reasons["norm"] += 1
             except IndexError: # Catch error if x_i has wrong shape
                 is_valid = False
                 reason = "stroke_shape_error"
                 invalid_reasons["stroke"] += 1


        # 3. Check character sequence length (only if still valid)
        if is_valid and (len(c_i) == 0 or len(c_i) > drawing.MAX_CHAR_LEN):
             is_valid = False
             reason = f"invalid_char_len_{len(c_i)}"
             invalid_reasons["char"] += 1

        # --- Collect valid data ---
        if is_valid:
            valid_indices.append(i)
            x_list.append(x_i)
            x_len_list.append(len(x_i))
            c_list.append(c_i)
            c_len_list.append(len(c_i))
            w_id_list.append(w_id_i)
        elif DEBUG_MODE:
             # Print reason only in debug mode
             print(f"  [Debug] Skipping item {i} ({os.path.basename(stroke_fname)}): Reason: {reason}")


    num_valid = len(valid_indices)
    num_invalid = total_items - num_valid
    print(f"\nProcessing complete. Found {num_valid} valid items, skipped {num_invalid} invalid items.")
    print(f"Invalid counts by reason: {invalid_reasons}")

    if num_valid == 0:
        print("Error: No valid data to save after processing.")
        exit()

    # --- Create and Save NumPy Arrays ---
    print('Creating final numpy arrays...')
    max_stroke_len_actual = max(x_len_list) if x_len_list else 0
    max_char_len_actual = max(c_len_list) if c_len_list else 0
    max_stroke_len_save = drawing.MAX_STROKE_LEN
    max_char_len_save = drawing.MAX_CHAR_LEN
    print(f"Actual Max Stroke Length: {max_stroke_len_actual} (Saving with {max_stroke_len_save})")
    print(f"Actual Max Char Length: {max_char_len_actual} (Saving with {max_char_len_save})")


    x_final = np.zeros([num_valid, max_stroke_len_save, 3], dtype=np.float32)
    x_len_final = np.array(x_len_list, dtype=np.int32)
    c_final = np.zeros([num_valid, max_char_len_save], dtype=np.int32)
    c_len_final = np.array(c_len_list, dtype=np.int32)
    w_id_final = np.array(w_id_list, dtype=np.int32)

    for idx in range(num_valid):
        x_i = x_list[idx]
        c_i = c_list[idx]
        x_final[idx, :len(x_i), :] = x_i
        c_final[idx, :len(c_i)] = c_i

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving processed data to {OUTPUT_DIR}...")

    np.save(os.path.join(OUTPUT_DIR, 'x.npy'), x_final)
    np.save(os.path.join(OUTPUT_DIR, 'x_len.npy'), x_len_final)
    np.save(os.path.join(OUTPUT_DIR, 'c.npy'), c_final)
    np.save(os.path.join(OUTPUT_DIR, 'c_len.npy'), c_len_final)
    np.save(os.path.join(OUTPUT_DIR, 'w_id.npy'), w_id_final)

    print("Successfully saved processed data.")
    print(f"Shapes saved:\n"
          f"  x: {x_final.shape}\n"
          f"  x_len: {x_len_final.shape}\n"
          f"  c: {c_final.shape}\n"
          f"  c_len: {c_len_final.shape}\n"
          f"  w_id: {w_id_final.shape}")
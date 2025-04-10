import os
import logging

import numpy as np
import svgwrite
import tensorflow as tf

import drawing
import lyrics
from rnn import HandwritingRNN


class Hand(object):
    """
    Handwriting synthesis class.
    Loads a pre-trained HandwritingRNN model and provides methods
    to generate handwriting for given text lines.
    Replicates TF1 demo priming logic.
    """

    def __init__(self, checkpoint_dir='checkpoints_tf2', config=None):
        """
        Initializes the Hand object by loading the TensorFlow model and ensuring it's built.

        Args:
            checkpoint_dir (str): Directory containing the saved model checkpoints.
            config (dict, optional): Dictionary specifying model hyperparameters.
                                     Defaults to a standard configuration if None.
        """
        tf.get_logger().setLevel('WARNING')

        default_config = { 'lstm_size': 400, 'output_mixture_components': 20, 'attention_mixture_components': 10, }
        if config is None: config = default_config
        self.config = config
        self.alphabet_size = len(drawing.alphabet)

        # --- Instantiate Model ---
        self.model = HandwritingRNN(
            lstm_size=config['lstm_size'], output_mixture_components=config['output_mixture_components'],
            attention_mixture_components=config['attention_mixture_components'], alphabet_size=self.alphabet_size )

        # --- Load Checkpoint ---
        self.checkpoint_dir = checkpoint_dir
        self.ckpt = tf.train.Checkpoint(model=self.model) # Checkpoint tracks the model
        latest_ckpt = tf.train.latest_checkpoint(self.checkpoint_dir)

        # --- Define Input Signature for Build ---
        # Use placeholder shapes (None for batch and time/char dims)
        self.input_signature_for_build = {
            'x': tf.TensorSpec(shape=[None, None, 3], dtype=tf.float32),
            'c': tf.TensorSpec(shape=[None, None], dtype=tf.int32),
            'x_len': tf.TensorSpec(shape=[None], dtype=tf.int32),
            'c_len': tf.TensorSpec(shape=[None], dtype=tf.int32)
        }

        if latest_ckpt:
            print(f"Restoring model from {latest_ckpt}")
            # Restore weights *before* building
            status = self.ckpt.restore(latest_ckpt).expect_partial()
            print("Model restored.")

            # *** Explicitly Build AFTER Loading Weights ***
            print("Building restored model layers...")
            try:
                self.model.build(self.input_signature_for_build)
                print("Model built successfully after restore.")
            except Exception as build_e:
                print(f"Error during post-restore build step: {build_e}")
                import traceback; traceback.print_exc()
                raise RuntimeError("Failed to build model after restoring checkpoint.") from build_e

        else: # No checkpoint found
            print("No checkpoint found. Building model from scratch...")
            try:
                self.model.build(self.input_signature_for_build)
                print("Model built successfully (from scratch).")
            except Exception as build_e:
                 print(f"Error during initial build step: {build_e}")
                 import traceback; traceback.print_exc()
                 raise RuntimeError("Failed to build model.") from build_e


    def write(self, filename, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None):
        """
        Generates handwriting for the given lines and saves it as an SVG file.
        """
        valid_char_set = set(drawing.alphabet); validated_lines = []
        if not isinstance(lines, list): print("Warning: 'lines' should be a list."); lines = []
        for line_num, line in enumerate(lines):
             if not isinstance(line, str): print(f"W: Line {line_num} not str."); continue
             if len(line) > 75: print(f"W: Line {line_num} > 75 chars.")
             valid_line = True
             for char_num, char in enumerate(line):
                 if char not in valid_char_set: raise ValueError(f"Invalid char '{char}' (ord={ord(char)}) at pos {char_num} in line {line_num}: \"{line}\". Valid: {''.join(drawing.alphabet)}")
             if valid_line: validated_lines.append(line)
        lines = validated_lines
        if not lines: print("E: No valid lines."); dwg = svgwrite.Drawing(filename=filename); dwg.viewbox(width=100, height=50); dwg.add(dwg.rect(insert=(0, 0), size=(100, 50), fill='white')); dwg.save(); print(f"Created empty SVG: {filename}"); return

        strokes = self._sample(lines, biases=biases, styles=styles)
        self._draw(strokes, lines, filename, stroke_colors=stroke_colors, stroke_widths=stroke_widths)
        print(f"Handwriting saved to {filename}")


    def _sample(self, lines, biases=None, styles=None):
        """Internal method using TF1 priming logic."""
        num_samples = len(lines)
        if num_samples == 0: return []

        max_len_line = max([len(line) for line in lines if line] + [0])
        max_tsteps = 40 * max_len_line if max_len_line > 0 else 100

        if biases is None: biases = [0.5] * num_samples
        biases = np.array(biases, dtype=np.float32)

        encoded_lines = [drawing.encode_ascii(line) for line in lines]

        # --- Variables for Priming Data ---
        # Initialize to None or empty arrays as appropriate
        x_prime = None
        x_prime_len = None
        alloc_x_prime_len = drawing.MAX_STROKE_LEN if hasattr(drawing, 'MAX_STROKE_LEN') else 1200 # Default alloc size like TF1
        alloc_c_len = drawing.MAX_CHAR_LEN if hasattr(drawing, 'MAX_CHAR_LEN') else 120 # Default alloc size like TF1

        # --- Lists to hold final character sequences ---
        final_chars_list = []
        final_chars_len_list = []

        # --- Handle Priming ---
        if styles is not None:
            if len(styles) != num_samples: raise ValueError(f"Styles ({len(styles)}) must match lines ({num_samples})")

            # --- Process Priming Strokes ---
            max_x_prime_len_actual = 0; x_prime_list = []
            for style in styles: # Check files first
                 style_stroke_file=f'styles/style-{style}-strokes.npy'; style_char_file=f'styles/style-{style}-chars.npy'
                 if not os.path.exists(style_stroke_file): raise FileNotFoundError(f"Missing: {style_stroke_file}")
                 if not os.path.exists(style_char_file): raise FileNotFoundError(f"Missing: {style_char_file}")
            for style in styles: # Load strokes
                style_stroke_file = f'styles/style-{style}-strokes.npy'; x_p = np.load(style_stroke_file).astype(np.float32)
                x_prime_list.append(x_p); max_x_prime_len_actual = max(max_x_prime_len_actual, len(x_p))

            # Determine allocation size and pad
            alloc_x_prime_len = max(max_x_prime_len_actual, alloc_x_prime_len)
            x_prime_padded_list = []; x_prime_len_list = []
            for x_p in x_prime_list:
                 pad_len = alloc_x_prime_len - len(x_p)
                 x_p_padded = np.pad(x_p, ((0, pad_len), (0, 0)), 'constant') if pad_len > 0 else x_p
                 x_prime_padded_list.append(x_p_padded); x_prime_len_list.append(len(x_p))
            x_prime = np.array(x_prime_padded_list, dtype=np.float32) # Assign to x_prime
            x_prime_len = np.array(x_prime_len_list, dtype=np.int32) # Assign to x_prime_len

            # --- Process Combined Characters ---
            space_code = drawing.encode_ascii(" ")[0:1]; max_c_len_combined = 0
            for i, (line_codes, style) in enumerate(zip(encoded_lines, styles)):
                style_char_file = f'styles/style-{style}-chars.npy'; c_p_data = np.load(style_char_file, allow_pickle=True)
                if c_p_data.dtype == object or isinstance(c_p_data.item(), (str, bytes)):
                    try:
                        if isinstance(c_p_data.item(), bytes): prime_text = c_p_data.item().decode('utf-8', 'ignore')
                        else: prime_text = str(c_p_data.item())
                        c_p_codes = drawing.encode_ascii(prime_text)
                        if len(c_p_codes) > 0 and c_p_codes[-1] == 0: c_p_codes = c_p_codes[:-1]
                    except Exception as e: print(f"Error decoding/encoding {style_char_file}: {e}"); c_p_codes = np.array([], dtype=np.int32)
                elif np.issubdtype(c_p_data.dtype, np.integer): c_p_codes = c_p_data.astype(np.int32)
                else: print(f"W: Unexpected dtype {c_p_data.dtype} in {style_char_file}"); c_p_codes = np.array([], dtype=np.int32)
                combined_codes = np.concatenate([c_p_codes, space_code, line_codes]).astype(np.int32)
                final_chars_list.append(combined_codes); final_chars_len_list.append(len(combined_codes)); max_c_len_combined = max(max_c_len_combined, len(combined_codes))
            alloc_c_len = max(max_c_len_combined, alloc_c_len) # Update allocation length

        else: # No priming
            final_chars_list = encoded_lines; final_chars_len_list = [len(codes) for codes in encoded_lines]
            if final_chars_len_list: max_c_len_combined = max(final_chars_len_list)
            alloc_c_len = max(max_c_len_combined, alloc_c_len)

        # --- Pad Final Character Sequences ---
        chars_padded_list = []
        for codes in final_chars_list:
            pad_len = alloc_c_len - len(codes) # Use alloc_c_len for padding target
            if pad_len >= 0: codes_padded = np.pad(codes, (0, pad_len), 'constant')
            else: print(f"W: Negative padding ({pad_len}). Truncating."); codes_padded = codes[:alloc_c_len]
            chars_padded_list.append(codes_padded)
        chars = np.array(chars_padded_list, dtype=np.int32)
        chars_len = np.array(final_chars_len_list, dtype=np.int32) # Actual lengths

        # --- Convert inputs to Tensors ---
        chars_tf = tf.convert_to_tensor(chars, dtype=tf.int32); chars_len_tf = tf.convert_to_tensor(chars_len, dtype=tf.int32)
        biases_tf = tf.convert_to_tensor(biases, dtype=tf.float32); max_tsteps_tf = tf.constant(max_tsteps, dtype=tf.int32)
        # *** Use corrected variable names 'x_prime' and 'x_prime_len' ***
        x_prime_tf = tf.convert_to_tensor(x_prime, dtype=tf.float32) if x_prime is not None else None
        x_prime_len_tf = tf.convert_to_tensor(x_prime_len, dtype=tf.int32) if x_prime_len is not None else None

        # --- Call model sample function ---
        print(f"Sampling {num_samples} sequences (max_tsteps={max_tsteps}). Priming={styles is not None}...")
        sampled_sequence_tf = self.model.sample(
            c=chars_tf, c_len=chars_len_tf, biases=biases_tf, max_tsteps=max_tsteps_tf,
            x_prime=x_prime_tf, x_prime_len=x_prime_len_tf )
        print("Sampling complete.")

        # --- Process output ---
        sampled_sequence_np = sampled_sequence_tf.numpy(); processed_samples = []
        for i in range(num_samples):
            sample = sampled_sequence_np[i]; valid_indices = np.where(np.any(sample != 0.0, axis=1))[0]
            if len(valid_indices) > 0: processed_samples.append(sample[:valid_indices[-1] + 1])
            else: processed_samples.append(np.zeros((0, 3), dtype=np.float32))
        return processed_samples


    def _draw(self, strokes, lines, filename, stroke_colors=None, stroke_widths=None):
        if not strokes or not lines: print("W: No strokes/lines to draw."); dwg = svgwrite.Drawing(filename=filename); dwg.viewbox(width=100, height=50); dwg.add(dwg.rect(insert=(0, 0), size=(100, 50), fill='white')); dwg.save(); print(f"Created empty SVG: {filename}"); return
        num_lines = len(lines); stroke_colors = stroke_colors or ['black']*num_lines; stroke_widths = stroke_widths or [2]*num_lines
        if len(strokes) != num_lines: print(f"W: Mismatch strokes ({len(strokes)}) vs lines ({num_lines}). Truncating."); min_len = min(len(strokes), num_lines); strokes=strokes[:min_len]; lines=lines[:min_len]; stroke_colors=stroke_colors[:min_len]; stroke_widths=stroke_widths[:min_len]; num_lines = min_len
        line_height = 60; view_width = 1000; view_height = line_height * (num_lines + 1)
        dwg = svgwrite.Drawing(filename=filename, profile='tiny'); dwg.viewbox(width=view_width, height=view_height); dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))
        for i, (offsets, line, color, width) in enumerate(zip(strokes, lines, stroke_colors, stroke_widths)):
            if not line or len(offsets) == 0: continue
            current_y_baseline = view_height - (i + 1) * line_height
            coords = drawing.offsets_to_coords(offsets); coords = drawing.denoise(coords); coords[:, :2] = drawing.align(coords[:, :2])
            coords[:, :2] *= 1.5; coords[:, 1] *= -1; min_coords = coords[:, :2].min(axis=0); coords[:, :2] -= min_coords
            line_width = coords[:, 0].max(); start_x = (view_width - line_width) / 2; coords[:, 0] += start_x; coords[:, 1] += current_y_baseline
            if len(coords) == 0: continue
            path_d = "M{:.2f},{:.2f}".format(coords[0, 0], coords[0, 1]); prev_eos = offsets[0, 2]
            for k in range(1, len(coords)):
                command = 'M' if prev_eos == 1.0 else 'L'; path_d += " {}{:.2f},{:.2f}".format(command, coords[k, 0], coords[k, 1])
                if k < len(offsets): prev_eos = offsets[k, 2]
                else: prev_eos = 1.0
            path = dwg.path(d=path_d, fill="none", stroke=color, stroke_width=width, stroke_linecap='round', stroke_linejoin='round'); dwg.add(path)
        try: dwg.save(pretty=True)
        except Exception as e: print(f"Error saving SVG file {filename}: {e}")


# --- Main execution block ---
if __name__ == '__main__':
    os.makedirs('styles', exist_ok=True); os.makedirs('img', exist_ok=True)
    dummy_style_indices = [1, 5, 7, 9, 12]
    for style_num in dummy_style_indices:
        dummy_strokes_file = f'styles/style-{style_num}-strokes.npy'; dummy_chars_file = f'styles/style-{style_num}-chars.npy'
        if not os.path.exists(dummy_strokes_file): print(f"Creating dummy strokes: {dummy_strokes_file}"); dummy_stroke = np.array([[0,0,1], [5,2,0], [3,-3,1]], dtype=np.float32); np.save(dummy_strokes_file, dummy_stroke)
        if not os.path.exists(dummy_chars_file): print(f"Creating dummy chars (text): {dummy_chars_file}"); dummy_text = "style "+str(style_num)+" text"; np.save(dummy_chars_file, np.array(dummy_text)) # Save raw text
    existing_style_num = -1
    for i in range(20):
         if os.path.exists(f'styles/style-{i}-strokes.npy') and os.path.exists(f'styles/style-{i}-chars.npy'): existing_style_num = i; break
    if existing_style_num == -1: print("W: No existing style files found. Using dummy style 9."); existing_style_num = 9
    print("Initializing Hand...");
    try: hand = Hand(checkpoint_dir='checkpoints_tf2'); print("Hand initialized.")
    except Exception as e: print(f"Failed Hand init: {e}"); exit()
    lines_usage = [ "Now this is a story all about how", "My life got flipped turned upside down", "And I'd like to take a minute, just sit right there", "I'll tell you how I became the prince of a town called Bel-Air", ]
    biases_usage = [.75] * len(lines_usage); style_usage = 9; styles_usage = [style_usage] * len(lines_usage)
    stroke_colors_usage = ['red', 'green', 'black', 'blue']; stroke_widths_usage = [1, 2, 1, 2]
    print("\nGenerating usage_demo.svg..."); hand.write(filename='img/usage_demo.svg', lines=lines_usage, biases=biases_usage, styles=styles_usage, stroke_colors=stroke_colors_usage, stroke_widths=stroke_widths_usage)
    if lyrics:
        if hasattr(lyrics, 'all_star'):
             print("\nGenerating all_star.svg..."); lines_as = lyrics.all_star.split("\n"); biases_as = [.75] * len(lines_as); style_as = 12
             styles_as = [style_as if os.path.exists(f'styles/style-{style_as}-strokes.npy') else existing_style_num] * len(lines_as)
             hand.write(filename='img/all_star.svg', lines=lines_as, biases=biases_as, styles=styles_as)
        else: print("Skipping all_star demo - lyrics.all_star not found.")
        if hasattr(lyrics, 'downtown'):
            print("\nGenerating downtown.svg..."); lines_dt = lyrics.downtown.split("\n"); biases_dt = [.75] * len(lines_dt)
            styles_indices_dt = np.cumsum(np.array([len(i) == 0 for i in lines_dt])).astype(int)
            available_styles = [s for s in dummy_style_indices if os.path.exists(f'styles/style-{s}-strokes.npy')] or [existing_style_num]
            styles_dt = [available_styles[idx % len(available_styles)] for idx in styles_indices_dt]
            hand.write(filename='img/downtown.svg', lines=lines_dt, biases=biases_dt, styles=styles_dt)
        else: print("Skipping downtown demo - lyrics.downtown not found.")
        if hasattr(lyrics, 'give_up'):
             print("\nGenerating give_up.svg..."); lines_gu = lyrics.give_up.split("\n"); biases_gu = .2 * np.flip(np.cumsum([len(i) == 0 for i in lines_gu]), 0) + 0.1; style_gu = 7
             styles_gu = [style_gu if os.path.exists(f'styles/style-{style_gu}-strokes.npy') else existing_style_num] * len(lines_gu)
             hand.write(filename='img/give_up.svg', lines=lines_gu, biases=biases_gu, styles=styles_gu)
        else: print("Skipping give_up demo - lyrics.give_up not found.")
    else: print("\nSkipping lyrics demos - 'lyrics' module not found or import failed.")
    print("\nDemo script finished.")
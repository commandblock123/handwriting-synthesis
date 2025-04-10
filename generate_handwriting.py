import argparse
from demo import Hand
import os
import svgwrite

def split_into_lines(text, max_chars_per_line):
    """Splits text into lines, respecting word boundaries."""
    words = text.split()
    lines = []
    current_line = ''
    for word in words:
        # Check if adding the word exceeds the limit
        # Add 1 for the space before the word (if current_line is not empty)
        next_line_len = len(current_line) + (1 if current_line else 0) + len(word)
        if current_line and next_line_len > max_chars_per_line:
            # Current line is full, start a new one
            lines.append(current_line)
            current_line = word
        else:
            # Add word to current line
            if current_line:
                current_line += ' '
            current_line += word
    # Add the last line if it's not empty
    if current_line:
        lines.append(current_line)
    return lines

def main():
    parser = argparse.ArgumentParser(description="Perform handwriting synthesis using TensorFlow 2.")
    parser.add_argument('-t', '--text', required=True, help='Text to synthesize')
    parser.add_argument('-s', '--style', type=int, default=None, help='Style number (requires style-<N>-strokes.npy and style-<N>-chars.npy). If omitted, uses unprimed sampling.')
    parser.add_argument('-b', '--bias', type=float, default=0.75, help='Bias value for sampling (controls variability/randomness)')
    parser.add_argument('--stroke_color', default='black', help='Stroke color (e.g., "black", "blue", "#FF0000")')
    parser.add_argument('--stroke_width', type=int, default=2, help='Stroke width in pixels')
    parser.add_argument('-o', '--output', required=True, help='Output SVG filename')
    parser.add_argument('--max_chars', type=int, default=65, help='Maximum characters per line before splitting')
    parser.add_argument('--checkpoint_dir', default='checkpoints_tf2', help='Directory containing model checkpoints')

    args = parser.parse_args()

    # Check if checkpoint directory exists
    if not os.path.isdir(args.checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {args.checkpoint_dir}")
        print("Please ensure the model is trained or checkpoints are placed correctly.")
        return

    # Check if style files exist if a style is specified
    if args.style is not None:
        style_stroke_file = os.path.join('styles', f'style-{args.style}-strokes.npy')
        style_char_file = os.path.join('styles', f'style-{args.style}-chars.npy')
        if not os.path.exists(style_stroke_file) or not os.path.exists(style_char_file):
            print(f"Error: Style files not found for style {args.style}.")
            missing_files = []
            if not os.path.exists(style_stroke_file):
                missing_files.append(style_stroke_file)
            if not os.path.exists(style_char_file):
                missing_files.append(style_char_file)
            print(f"Missing: {', '.join(missing_files)}")
            print("Primed sampling requires both files in the 'styles' directory.")
            return

    # Split text into lines
    lines = split_into_lines(args.text, args.max_chars)
    if not lines:
        print("Warning: Input text resulted in zero lines after splitting.")
        # Create an empty SVG
        dwg = svgwrite.Drawing(filename=args.output)
        dwg.viewbox(width=100, height=50)
        dwg.add(dwg.rect(insert=(0, 0), size=(100, 50), fill='white'))
        dwg.save()
        print(f"Created empty SVG: {args.output}")
        return


    print(f"Number of lines: {len(lines)}")
    print(f"Using style: {'Unprimed' if args.style is None else args.style}")
    print(f"Using bias: {args.bias}")

    # Initialize Hand (loads the model)
    try:
        hand = Hand(checkpoint_dir=args.checkpoint_dir)
    except ValueError as e:
        print(f"Error initializing Hand: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during Hand initialization: {e}")
        import traceback
        traceback.print_exc()
        return


    # Prepare arguments for hand.write
    num_lines = len(lines)
    biases = [args.bias] * num_lines
    styles = [args.style] * num_lines if args.style is not None else None
    stroke_colors = [args.stroke_color] * num_lines
    stroke_widths = [args.stroke_width] * num_lines

    # Generate handwriting
    print("Generating handwriting...")
    try:
        hand.write(
            filename=args.output,
            lines=lines,
            biases=biases,
            styles=styles,
            stroke_colors=stroke_colors,
            stroke_widths=stroke_widths
        )
        print(f"Successfully generated {args.output}")
    except Exception as e:
        print(f"An error occurred during handwriting generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
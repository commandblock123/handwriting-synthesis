import argparse
from demo import Hand

def split_into_lines(text, max_chars_per_line):
    words = text.split()
    lines = []
    current_line = ''
    for word in words:
        if len(current_line) + len(word) + 1 > max_chars_per_line:
            lines.append(current_line)
            current_line = word
        else:
            if current_line:
                current_line += ' '
            current_line += word
    if current_line:
        lines.append(current_line)
    return lines

def main():
    parser = argparse.ArgumentParser(description="Perform handwriting synthesis.")
    parser.add_argument('-text', required=True, help='Text to synthesize')
    parser.add_argument('-style', type=int, required=True, help='Style for all lines')
    parser.add_argument('-bias', type=float, required=True, help='Bias for all lines')
    parser.add_argument('-stroke_color', required=True, help='Stroke color for all lines')
    parser.add_argument('-stroke_width', type=int, required=True, help='Stroke width for all lines')
    parser.add_argument('-output', required=True, help='Output SVG filename')

    args = parser.parse_args()

    lines = split_into_lines(args.text, 75)

    hand = Hand()
    hand.write(
        filename=args.output,
        lines=lines,
        biases=[args.bias] * len(lines),
        styles=[args.style] * len(lines),
        stroke_colors=[args.stroke_color] * len(lines),
        stroke_widths=[args.stroke_width] * len(lines)
    )

if __name__ == '__main__':
    main()

import argparse
from demo import Hand

def main():
    parser = argparse.ArgumentParser(description="Perform handwriting synthesis.")
    parser.add_argument('-lines', nargs='+', required=True, help='List of lines')
    parser.add_argument('-style', type=int, required=True, help='Style for all lines')
    parser.add_argument('-bias', type=float, required=True, help='Bias for all lines')
    parser.add_argument('-stroke_color', required=True, help='Stroke color for all lines')
    parser.add_argument('-stroke_width', type=int, required=True, help='Stroke width for all lines')
    parser.add_argument('-output', required=True, help='Output SVG filename')

    args = parser.parse_args()

    hand = Hand()
    hand.write(
        filename=args.output,
        lines=args.lines,
        biases=[args.bias] * len(args.lines),
        styles=[args.style] * len(args.lines),
        stroke_colors=[args.stroke_color] * len(args.lines),
        stroke_widths=[args.stroke_width] * len(args.lines)
    )

if __name__ == '__main__':
    main()

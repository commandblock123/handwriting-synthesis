import subprocess
from flask import Flask, request, jsonify, send_file
import re

app = Flask(__name__)

allowed_colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green', 'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 'yellowgreen']

def change_characters(text):
    replacedText = text.replace("X", "x")
    replacedText = replacedText.replace("Z", "z")
    return replacedText

def is_valid_Input(text, style, bias, stroke_color, stroke_width):
    # 'Z' and 'X' were not in the dataset
    search = re.search(r"""[^abcdefghijklmnopqrstuvwxyz1234567890()!.,;#'"?:ABCDEFGHIJKLMNOPQRSTUVWY -]""", text)
    if search != None:
        return False, f"Text cannot contain the character '{search.group(0)}'."
    if style < 0 or style > 12 or not isinstance(style, int):
        return False, "Style must be an integer between 0 and 12."
    if bias < 0 or bias > 1:
        return False, "Bias must be a float between 0 and 1."
    if stroke_width <= 0 or not isinstance(stroke_width, int):
        return False, "Stroke width must be a positive integer."
    if stroke_color in allowed_colors:
        pass
    elif len(stroke_color) == 7 and stroke_color[0] == '#':
        try:
            int(stroke_color[1:], 16)
        except ValueError:
            return False, "Invalid hex color code."
    else:
        return False, "Invalid stroke color."
    
    return True, ""

@app.route('/synthesize_handwriting', methods=['POST'])
def synthesize_handwriting():
    try:
        # Parse request data
        data = request.json
        text = change_characters(data.get('text'))
        style = data.get('style')
        bias = data.get('bias')
        stroke_color = data.get('stroke_color')
        stroke_width = data.get('stroke_width')

        is_valid, message = is_valid_Input(text, style, bias, stroke_color, stroke_width)
        if is_valid:
            output_filename = "output.svg"

            # Run the handwriting synthesis script using os.system
            cmd = ["conda", "run", "-n", "python3.5", "python", "generate_handwriting.py"]
            args = ["-text", str(text), "-style", str(style), "-bias", str(bias), "-stroke_color", str(stroke_color), "-stroke_width", str(stroke_width), "-output", output_filename]

            process = subprocess.run(cmd + args, check=True)

            # Return the generated SVG file
            return send_file('{}'.format(output_filename), as_attachment=True)
        else:
            return jsonify({'error': f'Inputs are invalid. {message}'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route("/ping")
def ping():
    return "Pong!"

if __name__ == '__main__':
    #app.run(debug=True)
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)

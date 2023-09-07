import os
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

@app.route('/synthesize_handwriting', methods=['POST'])
def synthesize_handwriting():
    try:
        # Parse request data
        data = request.json
        lines = data.get('lines')
        style = data.get('style')
        bias = data.get('bias')
        stroke_color = data.get('stroke_color')
        stroke_width = data.get('stroke_width')
        output_filename = data.get('output_filename')

        # Run the handwriting synthesis script using os.system
        command = 'python generate_handwriting.py -lines {} -style {} -bias {} -stroke_color {} -stroke_width {} -output {}'.format(
            ' '.join(lines), style, bias, stroke_color, stroke_width, output_filename
        )

        os.system(command)

        # Return the generated SVG file
        return send_file('{}.svg'.format(output_filename), as_attachment=True, download_name='generated_handwriting.svg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

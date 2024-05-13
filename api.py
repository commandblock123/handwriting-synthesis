import os
from flask import Flask, request, jsonify, send_file

app = Flask(__name__)

@app.route('/synthesize_handwriting', methods=['POST'])
def synthesize_handwriting():
    try:
        # Parse request data
        data = request.json
        text = data.get('text')
        style = data.get('style')
        bias = data.get('bias')
        stroke_color = data.get('stroke_color')
        stroke_width = data.get('stroke_width')

        output_filename = "output.svg"

        # Run the handwriting synthesis script using os.system
        command = f"python generate_handwriting.py -text {text} -style {style} -bias {bias} -stroke_color {stroke_color} -stroke_width {stroke_width} -output {output_filename}"

        os.system(command)

        # Return the generated SVG file
        return send_file(f'{output_filename}.svg', as_attachment=True, download_name='generated_handwriting.svg')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    #app.run(debug=True)
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)

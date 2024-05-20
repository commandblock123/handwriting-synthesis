![](img/banner.svg)

# Handwriting Synthesis API

Implementation of the handwriting synthesis experiements in the paper <a href="https://arxiv.org/abs/1308.0850">Generating Sequences with Recurrent Neural Networks</a> by Alex Graves.

This repository provides a Dockerized implementation of the handwriting synthesis models, making it easy to use and integrate into applications.

## Getting Started

To get started, simply run the Docker container using the following command:
```bash
docker run -p 8080:8080 ghcr.io/commandblock123/handwriting-synthesis:v1.0.0-alpha
```
This will start the container and make the API available at http://localhost:8080.

## Usage

To use the Handwriting Synthesis API, send a POST request to http://localhost:8080/synthesize_handwriting with the following parameters:

text: The text to be synthesized into handwriting <br>
style: The style of handwriting to generate (0-12) <br>
bias: The bias of the handwriting (0-1) <br>
stroke_color: The color of the strokes <br>
stroke_width: The width of the strokes <br>
For example, using curl:

```bash
curl -X POST \
  http://localhost:8080/synthesize_handwriting \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello, World!", "style": 5, "bias": 0.75, "stroke_color": "black", "stroke_width": 1}'
```
This will generate a SVG file with the synthesized handwriting and return it as a response. You can also see an example of an API request in the [example_api_request.py](example_api_request.py) file.

## Demonstrations

![](img/style_0.svg)
![](img/style_1.svg)
![](img/style_2.svg)
![](img/style_3.svg)
![](img/style_4.svg)
![](img/style_5.svg)
![](img/style_6.svg)
![](img/style_7.svg)
![](img/style_8.svg)
![](img/style_9.svg)
![](img/style_10.svg)
![](img/style_11.svg)
![](img/style_12.svg)



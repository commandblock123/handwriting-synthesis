API Documentation
================

**Endpoints**

### `/synthesize_handwriting`

* **Method:** `POST`
* **Request Body:** JSON object with the following properties:
	+ `text`: string, the text to be synthesized into handwriting
	+ `style`: integer, the style of handwriting (between 0 and 12)
	+ `bias`: float, the bias of the handwriting (between 0 and 1)
	+ `stroke_color`: string, the color of the stroke (either a valid color name or a hex code)
	+ `stroke_width`: integer, the width of the stroke
* **Response:**
	+ `200 OK`: The generated SVG file will be sent as a response.
	+ `400 Bad Request`: If the input is invalid, an error message will be returned.
	+ `500 Internal Server Error`: If an unexpected error occurs, an error message will be returned.

### `/ping`

* **Method:** `GET`
* **Response:** `Pong!`

**Input Validation**

The following input validation rules apply:

* `text`: Must not contain any characters other than letters, numbers, and certain special characters (! . , ; # ' " ? : ( ) -).
* `style`: Must be an integer between 0 and 12.
* `bias`: Must be a float between 0 and 1.
* `stroke_color`: Must be a valid color name or a hex code.
* `stroke_width`: Must be a positive integer.

**Allowed Colors**

The following colors are allowed:

aliceblue, antiquewhite, aqua, aquamarine, azure, beige, bisque, black, blanchedalmond, blue, blueviolet, brown, burlywood, cadetblue, chartreuse, chocolate, coral, cornflowerblue, cornsilk, crimson, cyan, darkblue, darkcyan, darkgoldenrod, darkgray, darkgreen, darkgrey, darkkhaki, darkmagenta, darkolivegreen, darkorange, darkorchid, darkred, darksalmon, darkseagreen, darkslateblue, darkslategray, darkslategrey, darkturquoise, darkviolet, deeppink, deepskyblue, dimgray, dimgrey, dodgerblue, firebrick, floralwhite, forestgreen, fuchsia, gainsboro, ghostwhite, gold, goldenrod, gray, grey, green, greenyellow, honeydew, hotpink, indianred, indigo, ivory, khaki, lavender, lavenderblush, lawngreen, lemonchiffon, lightblue, lightcoral, lightcyan, lightgoldenrodyellow, lightgray, lightgreen, lightgrey, lightpink, lightsalmon, lightseagreen, lightskyblue, lightslategray, lightslategrey, lightsteelblue, lightyellow, lime, limegreen, linen, magenta, maroon, mediumaquamarine, mediumblue, mediumorchid, mediumpurple, mediumseagreen, mediumslateblue, mediumspringgreen, mediumturquoise, mediumvioletred, midnightblue, mintcream, mistyrose, moccasin, navajowhite, navy, oldlace, olive, olivedrab, orange, orangered, orchid, palegoldenrod, palegreen, paleturquoise, palevioletred, papayawhip, peachpuff, peru, pink, plum, powderblue, purple, red, rosybrown, royalblue, saddlebrown, salmon, sandybrown, seagreen, seashell, sienna, silver, skyblue, slateblue, slategray, slategrey, snow, springgreen, steelblue, tan, teal, thistle, tomato, turquoise, violet, wheat, white, whitesmoke, yellow, yellowgreen
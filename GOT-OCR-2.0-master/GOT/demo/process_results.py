import string

punctuation_dict = {
    "，": ",",
    "。": ".",

}


# import os
 
def svg_to_html(svg_content, output_filename):

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SVG Embedded in HTML</title>
    </head>
    <body>
        <svg width="2100" height="15000" xmlns="http://www.w3.org/2000/svg">
            {svg_content}
        </svg>
    </body>
    </html>
    """

    with open(output_filename, 'w') as file:
        file.write(html_content)
 

 
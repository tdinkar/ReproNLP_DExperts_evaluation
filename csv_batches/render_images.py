from jinja2 import Template
import csv
import pathlib
import subprocess
import os

absolute_path = os.path.abspath(__file__)
working_dir = os.path.dirname(absolute_path)
path_name = os.path.join(working_dir, 'batches/')
batch_names = 'batch'

def create_image(data, image_name):
    # Read the HTML template
    with open(working_dir+'/detoxHead2Head.html', 'r') as file:
        template_content = file.read()
    # Use Jinja2 template engine to substitute variables
    template = Template(template_content)
    html_content = template.render(data)
    # Save the modified HTML to a temporary file
    temp_html_path = 'temp.html'
    with open(temp_html_path, 'w') as temp_file:
        temp_file.write(html_content)
    # Use wkhtmltoimage to convert the HTML to PNG
    subprocess.run(['wkhtmltoimage', temp_html_path, image_name])
    # Delete the temporary HTML file
    os.remove(temp_html_path)

# create folder for images based on batch name
for i in range(1,33):
    batch_no = '_' + str(i)
    batch_file_name = path_name + batch_names + batch_no
    # make folder named on batch csv file to store images
    path = pathlib.Path(batch_file_name) 
    path.mkdir(parents=True, exist_ok=True) 
    # load csv batch file
    with open(batch_file_name + '.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for j, row in enumerate(reader):
            if j == 0:
                continue # skip header row
            else:
                image_name = 'sample_' + row[1] + '.png'
                image_path = batch_file_name + '/' + image_name
                data = {'sentprefix': row[3], 'senta': row[4], 'sentb': row[5]}
                create_image(data, image_path)

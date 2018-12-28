import pandas as pd
import math
from lxml import etree as ET
from xml.etree.ElementTree import parse, Element
import codecs
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', choices = {"train", "test"} , help='choose train or test.')
opt = parser.parse_args()

data_split = opt.dataset

labels = pd.read_csv("./data/labels_{}.csv".format(data_split))

output_dir = "annotations_{}".format(data_split)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

keys = {"previous_key": None, "current_key": None}

for index, row in labels.iterrows():

    keys["current_key"] = row["i"]
    image_id = int(row["i"])
    center_x = int(row["col_p"])
    center_y = int(row["row_p"])
    radius = int(math.ceil(row["radius_p"]))
    
    xmin_value = str(center_x - radius)
    xmax_value = str(center_x + radius)
    ymin_value = str(center_y - radius)
    ymax_value = str(center_y + radius)
    
    output_file = os.path.join(output_dir, 'image_{}.xml'.format(image_id))
    
    
    if keys["current_key"] != keys["previous_key"]:
        
        f=codecs.open(output_file,'w')
        root = ET.Element('annotation')
        filename = ET.SubElement(root, 'filename')
        filename.text='image_{}'.format(image_id)
        #path = ET.SubElement(root, 'path')
        #path.text='C:\\Users\\Spock\\craters\\images\\image_{}.jpg'.format(image_id)
        size = ET.SubElement(root, 'size')
        height = ET.SubElement(size, 'height')
        height.text = '224'
        width = ET.SubElement(size, 'width')
        width.text = '224'
        objet = ET.SubElement(root, 'object')
        name = ET.SubElement(objet, 'name')
        name.text = 'crater'
        bndbox = ET.SubElement(objet, 'bndbox')
        xmin = ET.SubElement(bndbox, 'xmin')
        xmin.text = xmin_value
        ymin = ET.SubElement(bndbox, 'ymin')
        ymin.text = ymin_value
        xmax = ET.SubElement(bndbox, 'xmax')
        xmax.text = xmax_value
        ymax = ET.SubElement(bndbox, 'ymax')
        ymax.text = ymax_value
        ET.tostring(root, pretty_print=True, xml_declaration=True)
        f.write(ET.tostring(root, pretty_print=True, xml_declaration=True).decode("utf-8") )
        f.close
        keys["previous_key"] = row["i"]
                      
    else:
        
        doc = parse(output_file)
        root = doc.getroot()
        e = Element('object')

        root.insert(2, e)

        name1 = Element('name')
        name1.text = 'crater'
        e.insert(1, name1)

        bnd1 = Element('bndbox')
        #name1.text = 'crater'
        e.insert(1, bnd1)

        xmin1 = Element('xmin')
        xmin1.text = xmin_value
        bnd1.insert(1, xmin1)

        ymin1 = Element('ymin')
        ymin1.text = ymin_value
        bnd1.insert(1, ymin1)

        xmax1 = Element('xmax')
        xmax1.text = xmax_value
        bnd1.insert(1, xmax1)

        ymax1 = Element('ymax')
        ymax1.text = ymax_value
        bnd1.insert(1, ymax1)

        doc.write(output_file, xml_declaration=True)
        keys["previous_key"] = row["i"]
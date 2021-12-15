from pathlib import Path
import os

for i in range(433):
    image_file_path = "raw_data/dataset1/images/Cars" + str(i) + ".png"
    xml_path = "raw_data/dataset1/annotations/Cars" + str(i) + ".xml"
    if os.path.exists(xml_path):
        file_exist = os.path.exists(image_file_path)
        if not file_exist:
            os.remove(xml_path)


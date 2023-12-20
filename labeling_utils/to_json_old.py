import json
from zipfile import ZipFile
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def to_json(fullpath):
    frames = []
    with ZipFile(fullpath, 'r') as zipObj:
        # Get a list of all archived file names from the zip
        listOfFileNames = zipObj.namelist()
        # Iterate over the file names
        bboxes = []
        out_of_sight = []
        for fileName in listOfFileNames:
            data = zipObj.read(fileName)
            # data = str(data)
            # print(data)
            tree = ET.ElementTree(ET.fromstring(data))
            track = tree.find('track')
            print(list(map(lambda x: x.attrib["frame"], track.findall("box"))))
            for box in track.findall("box"):
                attributes = box.attrib
                outside = attributes["outside"]
                if int(outside):
                    bbox = [0.0, 0.0, 0.0, 0.0]
                else:
                    bbox = [float(attributes['xtl']),
                            float(attributes["ytl"]),
                            float(attributes["xbr"]) - float(attributes["xtl"]),
                            float(attributes["ybr"]) - float(attributes["ytl"])]
                bboxes.append(",".join(list(map(str, bbox))))
                out_of_sight.append(str(outside))
                # frames.append([attributes['xtl'],
                #                attributes['ytl'],
                #                attributes['xbr'],
                #                attributes['ybr']])

                # print(box.attrib)
            # # Check filename endswith csv
            # if fileName.endswith('.csv'):
            #     # Extract a single file from zip
            #     zipObj.extract(fileName, 'temp_csv')
        # with open("f.json", 'w', encoding='utf-8') as f:
        #     json.dump(frames, f, indent=2)
        name = Path(fullpath).stem[5:8]
        with open(f"labels/bboxes/{name}.txt", 'w') as f:
            f.write("\n".join(bboxes))
        with open(f"labels/out_of_view/{name}.txt", 'w') as f:
            f.write(",".join(out_of_sight))


if __name__ == '__main__':
    file_name = sys.argv[1]
    if os.path.isdir(file_name):
        f = Path(file_name)
        for file in f.iterdir():
            to_json(str(file.absolute()))
    elif os.path.isfile(file_name):
        file = Path(file_name)
        to_json(str(file.absolute()))

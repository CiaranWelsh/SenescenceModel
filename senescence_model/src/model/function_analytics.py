import pandas, numpy
import os, glob
import xml.etree.ElementTree as ET
from shutil import move


def get_files(directory):
    return glob.glob('[0-9]*.xml')


def move_files_to(files, new_dir):
    os.makedirs(new_dir) if not os.path.isdir(new_dir) else None
    for i in files:
        if os.path.isfile(i):
            move(i, new_dir)


def parse_element(element, parsed=None):
    if parsed is None:
        parsed = dict()
    for key in element.keys():
        parsed[key] = element.attrib.get(key)
    if element.text:
        parsed[element.tag] = element.text
    for child in list(element):
        parse_element(child, parsed)
    return parsed


def read_one_xml(file):
    tree = ET.parse(file)
    root = tree.getroot()

    tdct = {}
    fdct = {}
    for element in root.findall(".//xagent"):
        dct = parse_element(element)
        if dct['name'] == 'Fibroblast':
            fdct[dct['id']] = dct
        elif dct['name'] == 'TissueBlock':
            tdct[dct['id']] = dct

    tdf = pandas.DataFrame(tdct).transpose()
    fdf = pandas.DataFrame(fdct).transpose()

    tdf = tdf.drop(['xagent', 'name'], axis=1)
    fdf = fdf.drop(['xagent', 'name'], axis=1)
    tdf = tdf.set_index(['id', 'x', 'y', 'z'])
    fdf = fdf.set_index(['id'])
    return tdf, fdf


def read_xml_data(files):
    tdf_list = []
    fdf_list = []
    for f in files:
        tdf, fdf = read_one_xml(f)
        tdf_list.append(tdf.stack())
        fdf_list.append(fdf.stack())

    tdf = pandas.concat(tdf_list, axis=1)
    fdf = pandas.concat(fdf_list, axis=1)


    return tdf, fdf


if __name__ == '__main__':
    model_dir = r'D:\Documents\SenescenceModel\senescence_model\src\model'

    xml_output_dir = r'D:\Documents\SenescenceModel\senescence_model\src\model\results\xml_output'

    files = get_files(model_dir)
    if files == []:
        files = get_files(xml_output_dir)

    tdf, fdf = read_xml_data(files)


    print(fdf)


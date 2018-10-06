import pandas, numpy
import os, glob
import xml.etree.ElementTree as ET
from shutil import copy
from shutil import Error as ShutilError
import matplotlib.pyplot as plt
import seaborn
from multiprocessing import Pool

def get_files(directory):
    return glob.glob(os.path.join(directory, '[0-9]*.xml'))


def move_files_to(files, new_dir):
    os.makedirs(new_dir) if not os.path.isdir(new_dir) else None
    if files == []:
        raise ValueError

    for i in files:
        copy(i, new_dir)
        os.remove(i)



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

    for i in tdf:
        tdf[i] = pandas.to_numeric(tdf[i])

    for i in fdf:
        fdf[i] = pandas.to_numeric(fdf[i])

    tdf = tdf.set_index(['id', 'x', 'y', 'z'])
    fdf = fdf.set_index(['id'])
    return tdf, fdf


def read_xml_data(files):
    tdf_list = []
    fdf_list = []
    p = Pool(6)
    for f in files:
        # p.map_async(read_one_xml, f)
        tdf, fdf = read_one_xml(f)
        tdf_list.append(tdf.stack())
        fdf_list.append(fdf.stack())

    tdf = pandas.concat(tdf_list, axis=1)
    fdf = pandas.concat(fdf_list, axis=1)


    return tdf, fdf


def plot_tissue_damage(tdf, id):
    if isinstance(id, int):
        id = [id]

    fig = plt.figure()
    for i in id:
        plt.plot(list(tdf.columns), tdf.loc[i].values[0])
        plt.xlabel('Simulation time')
        plt.ylabel('Amount of Damage')

    return fig


if __name__ == '__main__':
    model_dir = r'D:\Documents\SenescenceModel\senescence_model\src\model'

    xml_output_dir = r'D:\Documents\SenescenceModel\senescence_model\src\model\results\xml_output'

    try:
        files = get_files(model_dir)
        move_files_to(files, xml_output_dir)
    except ValueError:
        files = get_files(xml_output_dir)

    print(files)
    tdf, fdf = read_xml_data(files)

    # print(tdf.loc[0])


    fig = plot_tissue_damage(tdf, range(10))

    tdf_fig_fname = os.path.join(os.path.dirname(xml_output_dir), 'tissue_data.png')
    # plt.show()
    fig.savefig(tdf_fig_fname, bbox_inches='tight')





    tdf_fname = os.path.join(os.path.dirname(xml_output_dir), 'tissue_data.csv')
    fdf_fname = os.path.join(os.path.dirname(xml_output_dir), 'fibroblast_data.csv')

    tdf.to_csv(tdf_fname)
    fdf.to_csv(fdf_fname)










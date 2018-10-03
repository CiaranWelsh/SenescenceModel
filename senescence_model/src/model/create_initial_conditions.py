import numpy
from lxml import etree
import os
import os, glob

'''    <itno>0</itno>
    <environment></environment>
    <xagent>
        <name>A</name>
        <id>0</id>
        <x>-0.12465215</x>
        <y>-0.2607972</y>
        <z>-0.39554715</z>
        <fx>0.1592794</fx>
        <fy>-0.29218417</fy>
        <fz>-0.29697996</fz>
    </xagent>'''


def create_root():
    root = etree.Element('states')
    itno = etree.SubElement(root, 'itono')
    itno.text = '0'
    etree.SubElement(root, 'environment')
    return root


def add_agent(root, agent_attributes):
    agent = etree.SubElement(root, 'xagent')
    for k, v in agent_attributes.items():
        e = etree.SubElement(agent, k)
        e.text = str(v)
    return root


def add_fibroblast_agents(n, root, lower_bound=0, upper_bound=10):
    """
    construct tissue agent data
    """
    for agent in range(n):
        id = agent
        ## random position
        x, y, z = numpy.random.uniform(lower_bound, upper_bound, 3)
        doublings = 0
        damage = 0
        current_state = 0
        early_sen_time_counter = 0
        agent_args = {
            'id': id,
            'name': 'Fibroblast',
            'x': x,
            'y': y,
            'z': z,
            'doublings': doublings,
            'damage': damage,
            'current_State': current_state,
            'early_sen_time_counter': early_sen_time_counter,
        }
        add_agent(root, agent_args)

    return root



def add_tissue_agents(scale=1, grid_size=10):
    centers = []
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                bottom_left_back = numpy.array([x, y, z])
                bottom_left_front = numpy.array([x, y + scale, z])
                bottom_right_back = numpy.array([x + scale, y, z])
                bottom_right_front = numpy.array([x + scale, y + scale, z])
                front_left_back = numpy.array([x, y, z + scale])
                front_left_front = numpy.array([x, y + scale, z + scale])
                front_right_back = numpy.array([x + scale, y, z + scale])
                front_right_front = numpy.array([x + scale, y + scale, z + scale])
                x_vec, y_vec, z_vec = zip(
                    bottom_left_back,
                    bottom_left_front,
                    bottom_right_back,
                    bottom_right_front,
                    front_left_back,
                    front_left_front,
                    front_right_back,
                    front_right_front,
                )
                x_center = sum(x_vec) / 8.0
                y_center = sum(y_vec) / 8.0
                z_center = sum(z_vec) / 8.0
                centers.append((x_center, y_center, z_center))

    for i in range(len(centers)):
        id = i
        ## random position
        x, y, z = centers[i]
        damage = 0
        agent_args = {
            'id': id,
            'name': 'TissueBlock',
            'x': x,
            'y': y,
            'z': z,
            'damage': damage,
        }
        add_agent(root, agent_args)

    return root


def to_file(root, fname):
    with open(fname, 'w') as f:
        f.write(etree.tostring(root, pretty_print=True, encoding='unicode'))


if __name__ == '__main__':
    root = create_root()
    root = add_fibroblast_agents(10, root)
    root = add_tissue_agents(1, 3)

    fname = os.path.join(os.path.dirname(__file__), 'init.xml')

    to_file(root, fname)

    # add_agent(root, )

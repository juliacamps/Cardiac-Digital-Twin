# Define here the paths to your OneDrive and other personal paths that you need to redefine for your code to work
# The code will later use the file generated from this code to replace the paths in the code with your own custom ones.
import json


def set_path_mapping(path_mapping_json):
    # parse json:
    # path_dict = json.loads(path_mapping_json)
    # the result is a Python dictionary:
    # print(path_dict["data_path"]) # Test that your path is correct
    with open('../.custom_config/.your_path_mapping.txt', 'w') as f:
        f.write(path_mapping_json)


def get_path_mapping():
    mapping_filename = '../.custom_config/.your_path_mapping.txt'
    with open(mapping_filename, 'r') as f:
        path_mapping_json = f.read()
    # print(path_mapping_json)
    return json.loads(path_mapping_json)


def get_server_config():
    server_config_filename = '../.custom_config/.your_server_config.txt'
    with open(server_config_filename, 'r') as f:
        server_config_json = f.read()
    return json.loads(server_config_json)


if __name__ == '__main__':
    # path mappings in JSON:
    path_mapping_json = '{"data_path":"C:/Users/julmps/OneDrive - Nexus365/Personalisation_projects/meta_data/",' \
                        '"results_path":"C:/Users/julmps/OneDrive - Nexus365/Personalisation_projects/meta_data/results/"}'
    set_path_mapping(path_mapping_json)
    server_config_json = '{"python_path":"/p/project/icei-prace-2022-0003/camps1/miniconda3/envs/penv/bin/python3",' \
                         '"code_path":"/p/project/icei-prace-2022-0003/camps1/Cardiac_Personalisation/src/"}'


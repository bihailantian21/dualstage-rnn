import runpy


def read_config(data):
    try:
        config = runpy.run_path(data)['config_dict']
        return config
    except FileNotFoundError:
        print(f'File not found')

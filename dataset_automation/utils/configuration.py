import os, sys
import yaml, nomen

def getConfigFile(config):
    model_path = os.path.join(os.environ['experiments'], config['model'])
    # Parse config ----------------------------------------------------------------------------
    print("Reading parameters from yaml file")
    filename = os.path.join(model_path, 'config.yml')
    
    with open(filename, 'r') as config:
        try:
            dictionary = yaml.safe_load(config)
            print(dictionary)
        except yaml.YAMLError as exc:
            sys.exit('configuration file not found')
    
    cfg = nomen.Config(dictionary)
    cfg.parse_args()
    return cfg
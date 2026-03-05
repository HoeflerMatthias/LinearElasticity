import json
import os

def hash_params(params):
    return json.dumps(params, sort_keys=True)

def make_dirs(dirlist):
    for dir in dirlist:
        if not os.path.exists(dir):
            os.makedirs(dir)

def params_to_filename(params, keylist = [], separator = ""):
    
    def format_entry(obj, key, path = ""):
        code = ""
        if isinstance(obj, list):
            if key == 'layers':
                key_str = 'L'
            elif key == 'phases':
                key_str = 'P'
            else:
                key_str = key
            if isinstance(obj[0], str):
                code += path + key_str + ''.join([l + '_' for l in obj])
            else:
                code += path + key_str + ''.join(['%d_' % l for l in obj])
        elif isinstance(obj, float):
            if key not in ['lr1', 'lr2']:
                code += path + key + "%1.1e" % (obj)
        elif isinstance(obj, bool):
            if obj == True:
                code += path + key
        elif isinstance(obj, int):
            code += path + key + "%d" % (obj)
        elif isinstance(obj, str):
            code += path + obj
        return code
        
    filename = ""
    keylist_comp = [key.split('/') for key in keylist if '/' in key]
    keylist = [key for key in keylist if '/' not in key]
    for key_comp in keylist_comp:
        obj = params[key_comp[0]]
        for key in key_comp[1:]:
            obj = obj[key]
        filename += format_entry(obj, key_comp[-1], ''.join([s[:3] for s in key_comp[:-1]]))
    
    for key in params:
        if key in keylist or isinstance(params[key], dict) or len(keylist) == 0:
            
            obj = params[key]
            if isinstance(obj, dict):
                if key == 'fourier_params' and params['fourier'] == 1:
                    filename += separator + params_to_filename(params['fourier_params'], keylist)
                elif key == 'time_params':
                    filename += separator + params_to_filename(params['time_params'], keylist)
                elif key == 'slicing_params' and params['slicing'] == 1:
                    filename += separator + params_to_filename(params['slicing_params'], keylist)
                elif key == 'bc':
                    filename += separator + params_to_filename(params['bc'], keylist)
                elif key == 'model':
                    filename += separator + params_to_filename(params['model'], keylist)
                elif key == 'inverse_params':
                    filename += separator + params_to_filename(params['inverse_params'], keylist)
            else:
                filename += separator + format_entry(obj, key)
    return filename

def replace_item(obj, keylist, replace_value):
    if isinstance(keylist, list):
        if len(keylist) > 1:
            obj[keylist[0]] = replace_item(obj[keylist[0]], keylist[1:], replace_value)
        else:
            obj[keylist[0]] = replace_value
    else:
        obj[keylist] = replace_value

    return obj

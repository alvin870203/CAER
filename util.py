import os
import shutil
def logger(txt, log_info, mode = 'a'):
    
    # print(log_info)
    f = open(txt, mode)
    f.write(log_info + '\n')
    f.close()


def make_dir(_path, _is_del = True):
    
    if _path[-1] != '/':
        _path = _path + '/'
    
    
    if os.path.isdir(_path):
        if _is_del == True:
            shutil.rmtree(_path)
            os.mkdir(_path)
    else:
        if not os.path.exists(os.path.dirname(_path)):
            os.makedirs(os.path.dirname(_path))
    
    return _path
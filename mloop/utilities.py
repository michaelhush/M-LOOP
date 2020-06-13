'''
Module of common utility methods and attributes used by all the modules.
'''
from __future__ import absolute_import, division, print_function
__metaclass__ = type

import scipy.io as si
import pickle
import logging
import datetime
import sys
import os
import numpy as np
import numpy.random as nr
import base64
import mloop

python_version = sys.version_info[0]

#For libraries with different names in pythons 2 and 3
if python_version < 3:
    import Queue #@UnresolvedImport @UnusedImport
    empty_exception = Queue.Empty
else:
    import queue
    empty_exception = queue.Empty


default_interface_in_filename = 'exp_output'
default_interface_out_filename = 'exp_input'
default_interface_file_type = 'txt'

archive_foldername = './M-LOOP_archives/'
log_foldername = './M-LOOP_logs/'
default_log_filename = 'M-LOOP_'

filewrite_wait = 0.1

mloop_path = os.path.dirname(mloop.__file__)

#Set numpy to have no limit on printing to ensure all values are saved
np.set_printoptions(threshold=np.inf)

def config_logger(**kwargs):
    '''
    Wrapper for _config_logger.
    '''
    _ = _config_logger(**kwargs)

def _config_logger(log_filename = default_log_filename,
                  file_log_level=logging.DEBUG,
                  console_log_level=logging.INFO,
                  **kwargs):
    '''
    Configure and the root logger.
    
    Keyword Args:
        log_filename (Optional [string]) : Filename prefix for log. Default MLOOP run . If None, no file handler is created
        file_log_level (Optional[int]) : Level of log output for file, default is logging.DEBUG = 10
        console_log_level (Optional[int]) :Level of log output for console, defalut is logging.INFO = 20
    
    Returns:
        dictionary: Dict with extra keywords not used by the logging configuration.
    '''    
    log = logging.getLogger('mloop')
    
    if len(log.handlers) == 0:
        log.setLevel(min(file_log_level,console_log_level))
        if log_filename is not None:
            date_string = datetime_to_string(datetime.datetime.now())
            full_filename = log_filename + date_string + '.log'
            filename_with_path = os.path.join(log_foldername, full_filename)
            # Create folder if it doesn't exist, accounting for any parts of the
            # path that may have been included in log_filename.
            actual_log_foldername = os.path.dirname(filename_with_path)
            if not os.path.exists(actual_log_foldername):
                os.makedirs(actual_log_foldername)
            fh = logging.FileHandler(filename_with_path)
            fh.setLevel(file_log_level)
            fh.setFormatter(logging.Formatter('%(asctime)s %(name)-20s %(levelname)-8s %(message)s'))
            log.addHandler(fh)
        ch = logging.StreamHandler(stream = sys.stdout)
        ch.setLevel(console_log_level)
        ch.setFormatter(logging.Formatter('%(levelname)-8s %(message)s'))
        log.addHandler(ch)
        log.debug('MLOOP Logger configured.')
    
    return kwargs

def datetime_to_string(datetime):
    '''
    Method for changing a datetime into a standard string format used by all packages.
    '''
    return datetime.strftime('%Y-%m-%d_%H-%M')

def generate_filename_suffix(file_type, file_datetime=None, random_bytes=False):
    '''
    Method for generating a string with date and extension for end of file names.
    
    This method returns a string such as '_2020-06-13_04-20.txt' where the date
    and time specify when this function was called.
    
    Args:
        file_type (string): The extension to use at the end of the filename,
            e.g. 'txt'. Note that the period should NOT be included.
        file_datetime (Optional datetime.datetime): The date and time to use in
            the filename suffix, represented as an instance of the datetime
            class defined in the datetime module. If set to None, then this
            function will use the result returned by datetime.datetime.now().
            Default None.
        random_bytes (Optional bool): If set to True, six random bytes will be
            added to the filename suffix. This can be useful avoid duplication
            if multiple filenames are created with the same datetime.
        
    Returns:
        string: A string giving the suffix that can be appended to a filename
            prefix to give a full filename with timestamp and extension, such as
            '_2020-06-13_04-20.txt'. The date and time specify when this
            function was called.
    '''
    if file_datetime is None:
        file_datetime = datetime.datetime.now()
    date_string = datetime_to_string(file_datetime)
    filename_suffix = '_' + date_string 
    if random_bytes:
        random_string = base64.urlsafe_b64encode(nr.bytes(6)).decode()
        filename_suffix = filename_suffix + '_' + random_string
    filename_suffix = filename_suffix + '.' + file_type
    return filename_suffix

def dict_to_txt_file(tdict,filename):
    '''
    Method for writing a dict to a file with syntax similar to how files are input.
    
    Args:
        tdict (dict): Dictionary to be written to file.
        filename (string): Filename for file. 
    '''
    with open(filename,'w') as out_file:
        for key in tdict:
            out_file.write(str(key) + '=' + repr(tdict[key]).replace('\n', '').replace('\r', '') + '\n')

def txt_file_to_dict(filename):
    '''
    Method for taking a file and changing it to a dict. Every line in file is a new entry for the dictionary and each element should be written as::
    
        [key] = [value]
    
    White space does not matter.
    
    Args:
        filename (string): Filename of file.
        
    Returns:
        dict : Dictionary of values in file. 
    '''
    with open(filename,'r') as in_file:
        tdict_string = ''
        for line in in_file:
            temp = (line.partition('#')[0]).strip('\n').strip()
            if temp != '':
                tdict_string += temp+','    
    #Setting up words for parsing a dict, ignore eclipse warnings
    array = np.array    #@UnusedVariable
    inf = float('inf')  #@UnusedVariable
    nan = float('nan')  #@UnusedVariable
    tdict = eval('dict('+tdict_string+')')
    return tdict
    
def save_dict_to_file(dictionary,filename,file_type):
    '''
    Method for saving a dictionary to a file, of a given format. 
    
    Args:
        dictionary: The dictionary to be saved in the file.
        filename: The filename for the saved file
        file_type: The file_type for the saved file. Can be 'mat' for matlab, 'txt' for text, or 'pkl' for pickle.
    '''
    if file_type=='mat':
        si.savemat(filename,dictionary)
    elif file_type=='txt':
        dict_to_txt_file(dictionary,filename)
    elif file_type=='pkl':
        with open(filename,'wb') as out_file:
            pickle.dump(dictionary,out_file) 
    else:
        raise ValueError 
    
def get_dict_from_file(filename,file_type):
    '''
    Method for getting a dictionary from a file, of a given format. 
    
    Args:    
        filename: The filename for the file.
        file_type: The file_type for the file. Can be 'mat' for matlab, 'txt' for text, or 'pkl' for pickle.
    
    Returns:
        dict : Dictionary of values in file.
    '''
    if file_type=='mat':
        dictionary = si.loadmat(filename)
    elif file_type=='txt':
        dictionary = txt_file_to_dict(filename)
    elif file_type=='pkl':
        with open(filename,'rb') as in_file:
            dictionary = pickle.load(in_file) 
    else:
        raise ValueError
    return dictionary
    
def check_file_type_supported(file_type):
    '''
    Checks whether the file type is supported
    
    Returns: 
        bool : True if file_type is supported, False otherwise. 
    '''
    return file_type == 'mat' or 'txt' or 'pkl'

def safe_cast_to_array(in_array):
    '''
    Attempts to safely cast the input to an array. Takes care of border cases
    
    Args:
        in_array (array or equivalent): The array (or otherwise) to be converted to a list.
    
    Returns:
        array : array that has been squeezed and 0-D cases change to 1-D cases
    
    '''
    
    out_array = np.squeeze(np.array(in_array))
    
    if out_array.shape == ():
        out_array = np.array([out_array[()]]) 
    
    return out_array
    
def safe_cast_to_list(in_array):
    '''
    Attempts to safely cast a numpy array to a list, if not a numpy array just casts to list on the object.
    
    Args:
        in_array (array or equivalent): The array (or otherwise) to be converted to a list.
    
    Returns:
        list : List of elements from in_array
    
    '''
    
    if isinstance(in_array, np.ndarray):
        t_array = np.squeeze(in_array)
        if t_array.shape == ():
            out_list = [t_array[()]]
        else:
            out_list = list(t_array)
    else:
        out_list = list(in_array)
    
    return out_list
    
    
class NullQueueListener():
    '''
    Shell class with start and stop functions that do nothing. Queue listener is not implemented in python 2. Current fix is to simply use the multiprocessing class to pipe straight to the cmd line if running on python 2. This is class is just a placeholder.
    '''
    def start(self):
        '''
        Does nothing
        '''
        pass
    
    def stop(self):
        '''
        Does nothing
        '''
        pass



    
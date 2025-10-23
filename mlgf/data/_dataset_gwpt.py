from ._data_gwpt import Data_gwpt
from ._dataset import Dataset, sort_fnames_by_number, get_conf_nums
import os, warnings
    

class Dataset_gwpt(Dataset):    
    def __init__(self, iterable, loaded = {}, data_format = 'chk', preserve_order=True, load_data = True, purge_keys = [], core_projection_file_path = None, basis = 'saiao'):
        if preserve_order:
            self.fnames = list(iterable)
        else:
            self.fnames = sort_fnames_by_number(iterable)
        self.conf_nums = list(get_conf_nums(self.fnames))
        self.conf_nums_to_fnames = dict(zip(self.conf_nums, self.fnames))
        self.purge_keys = purge_keys
        self.basis = basis
       
        self.confs_are_unique = (len(self.conf_nums_to_fnames) == len(self.conf_nums))
        
        if not self.confs_are_unique:
            warnings.warn('Conf numbers are not unique.')
        
        self.loaded = loaded
        self.data_format = data_format
        self.load_data = load_data
        if self.data_format not in Dataset_gwpt._format_extension_tbl.keys():
            raise ValueError(f'Unknown data format {self.data_format}; acceptable values are {Dataset_gwpt._format_extension_tbl.keys()}')
        if not core_projection_file_path is None:
            minao_val = core_projection_file_path + '/minao_val.dat'
            minao_core = {'Si': core_projection_file_path + '/minao_core.dat', 'C' : core_projection_file_path + '/minao_core.dat', 'O' : core_projection_file_path + '/minao_core.dat'}
            self.val_core_dats = [minao_val, minao_core]
        else:
            self.val_core_dats = None

    def get_by_fname(self, fname):
        if fname in self.loaded:
            return self.loaded[fname]
        else:
            if self.data_format == 'joblib':
                raise NotImplementedError('joblib format not implemented for GWPT data')
            elif self.data_format == 'chk':
                dat = Data_gwpt.load_chk(fname, purge_keys = self.purge_keys, val_core_dats = self.val_core_dats, basis = self.basis)
            elif self.data_format == 'internal':
                raise NotImplementedError('internal format not implemented for GWPT data')
            else:
                raise ValueError(f'Unknown data format {self.data_format}; acceptable values are {Dataset._format_extension_tbl.keys()}')
            if self.load_data:
                self.loaded[fname] = dat
            return dat
        
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        return self.get_by_fname(fname)
    
    @staticmethod
    def from_files(file_list, dropout_files = [], data_format = 'chk', load_data = True, purge_keys = [], core_projection_file_path = None, basis = 'saiao'):
        return Dataset_gwpt([filename for filename in file_list if filename not in dropout_files], data_format=data_format, load_data = load_data, purge_keys = purge_keys, core_projection_file_path = core_projection_file_path, basis = basis)
            
    @staticmethod
    def from_srcdirectory(src, dropout_files = [], data_format = 'chk'):
        extension = Dataset_gwpt._format_extension_tbl[data_format]
        filenames = [f for f in os.listdir(src) if f.endswith(extension) and f not in dropout_files]
        return Dataset_gwpt([os.path.join(src, filename) for filename in filenames], data_format=data_format)
    
    def get_subset(self, indices, load_data = True):
        """Takes a subset of the dataset, specified by indices.

        Args:
            indices (list(int)): list of indices

        Returns:
            Dataset: subset specified by indices
        """
        indices = sorted(indices)
        fnames = [self.fnames[idx] for idx in indices]

        return Dataset_gwpt(fnames,
                loaded = {fname: self.loaded[fname] for fname in fnames if fname in self.loaded},
                data_format=self.data_format, load_data = load_data
        )

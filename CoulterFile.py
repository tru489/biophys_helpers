from pathlib import Path
import re
from dataclasses import dataclass, asdict
import numpy as np

class CoulterFile():
    def __init__(self, file_path) -> None:
        self.stat_tags = ['Mean', 'Mode', 'Median', 'SD', 'CV', 'MinSize', 
                          'MaxSize', 'SampleSize']
        self._populate_fields(file_path)

    def _populate_fields(self, file_path) -> None:
        """
        Populate class fields by reading coulter counter .#m4 file

        Args:
            file_path (string): path to coulter counter file
        """
        fp = Path(file_path)
        with fp.open('r') as file:
            lines = file.readlines()

        try:
            self.stats = self._get_selection_stats(lines)
        except ValueError:
            self.stats = None
        self.bin_edges_diameter, self.bin_edges_volume, self.bin_counts = \
            self._get_hist(lines)
        self.diameters, self.volumes = self._get_single_cell(lines)
    
    def _get_selection_stats(self, lines) -> dict:
        """
        Extracts pre-selected/gated statistic values from a single coulter 
        counter file

        Args:
            lines (list(str)): list of lines in coulter counter file 

        Raises:
            ValueError: Raises error if statistics are not in the coulter counter
                file

        Returns:
            dict: dictionary {stat name:value}
        """
        relevant_lines = self._get_file_section(lines, '[SizeStats]')
        
        # Extract numbers after the equals sign
        stat_dict = {}
        for line in relevant_lines:
            statname, val = re.match(r'([\w\(\)\,]+)=\s*([-\d\.]+)', line).groups()
            if statname in self.stat_tags:
                stat_dict[statname] = val
        return stat_dict

    def _get_hist(self, lines) -> tuple:
        """
        Get histogram bin edges and counts from coulter counter raw file

        Args:
            lines (list(str)): list of strings from coulter counter file
        """
        edges_diameter = self._get_file_section(lines, '[#Bindiam]')
        edges_diameter = [float(dm) for dm in edges_diameter]
        edges_volume = [4/3 * np.pi * (dm/2)**3 for dm in edges_diameter]

        bin_counts = self._get_file_section(lines, '[#Binheight]')
        bin_counts = [int(ct) for ct in bin_counts]

        return (np.array(edges_diameter), np.array(edges_volume), 
                np.array(bin_counts))

    def _get_single_cell(self, lines) -> np.array:
        counts_per_volt = 1 / (4 * 298.02e-9)
        kd_lst = self._get_file_section(lines, '[KDsave0]')
        kd_str = [str_ for str_ in kd_lst if 'Kd= ' in str_][0]
        get_param = lambda marker, search_str: \
            float(re.match(f'^{marker}' + r'([-\d\.]+)', search_str).groups()[0])
        kd = get_param('Kd= ', kd_str)

        param_lst = self._get_file_section(lines, '[instrument]')
        amp_str = [str_ for str_ in param_lst if 'Current= ' in str_][0]
        current = get_param('Current= ', amp_str) / 1000

        res_str = [str_ for str_ in param_lst if 'Gain= ' in str_][0]
        resistance = get_param('Gain= ', res_str) * 25

        mxht_str = [str_ for str_ in param_lst if 'MaxHtCorr= ' in str_][0]
        max_ht_corr = get_param('MaxHtCorr= ', mxht_str)

        pulse_strs = self._get_file_section(lines, '[#Pulses5hex]')
        get_first_hex = lambda str_: \
            re.match(r'^([A-Z\d]+),[A-Z\d,]+$', str_).groups()[0]
        hex_convert = [int(get_first_hex(str_), 16) for str_ in pulse_strs]
        hex_convert = np.array(hex_convert)
        
        height = hex_convert + max_ht_corr
        diameter = kd * ((height / (counts_per_volt * resistance * current))**(1/3))
        volume = 4/3 * np.pi * (diameter/2)**3

        return diameter, volume
        
    def _get_file_section(self, lines, start_marker) -> list:
        """
        Given a bracketed section marker, extract the lines of that section
        from the coulter counter file. The section ends at the next line that
        looks like a section header (starts with '[' and ends with ']').

        Args:
            lines (list(str)): lines from coulter counter raw file
            start_marker (str): bracketed start marker of lines of interest

        Raises:
            ValueError: start_marker was not found in the file

        Returns:
            list(str): lines between start_marker and the next section header
        """
        start_index = None
        for i, line in enumerate(lines):
            if start_marker in line:
                start_index = i
                break

        if start_index is None:
            raise ValueError(f"Section '{start_marker}' was not found in file.")

        result = []
        for line in lines[start_index + 1:]:
            stripped = line.strip()
            if stripped.startswith('[') and stripped.endswith(']'):
                break
            result.append(line)
        return result

    def get_stats(self) -> dict:
        """
        Getter for pre-selected coulter file stats

        Returns:
            dict: coulter file stats {name of stat: value}
        """
        return self.stats
    
    def get_diameters(self) -> np.array:
        return self.diameters
    
    def get_volumes_ungated(self) -> np.array:
        return self.volumes

    def get_volumes_gated(self) -> np.array:
        if self.stats is None:
            return None
        min_size = float(self.stats['MinSize'])
        max_size = float(self.stats['MaxSize'])
        mask = (self.volumes >= min_size) & (self.volumes <= max_size)
        return self.volumes[mask]


def pairwise_mean(lst) -> list:
    """
    Calculates the pairwise mean between adjacent elements in the list (mean 
    of elements 0 and 1, then 1 and 2, etc)

    Args:
        lst (float): original list
    Returns:
        list: pairwise mean list (length of len(lst)-1)
    """
    return np.array([(lst[i] + lst[i + 1]) / 2 for i in range(len(lst) - 1)])
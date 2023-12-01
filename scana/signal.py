import os
import json
import numpy
import pandas

__all__ = ['Signal']

MSEC_TO_SEC = 1.0e-3
SEC_TO_MSEC = 1.0e+3


class Signal:

    def __init__(self, freq, samples, *, label=None, unit=None, copy_data=False):
        self._freq = round(float(freq), 2)  # two decimals
        if not self._freq > 0.0:
            raise ValueError('Error, frequency must be greater than zero!')
        self._samples = numpy.array(samples, dtype=float, copy=copy_data)
        if not self._samples.ndim == 1:
            raise ValueError('Error, 1D sample set expected!')
        self._label = None if label is None else str(label)
        self._unit = None if unit is None else str(unit)

    def __len__(self):
        return len(self._samples)

    def __copy__(self):
        return Signal(self._freq, self._samples, label=self._label,
                      unit=self._unit, copy_data=True)

    def __str__(self):
        desc_str = 'freq={:.2f}_Hz,size={}'.format(self._freq, len(self._samples))
        if self._label is not None:
            desc_str += ',label="{}"'.format(self._label)
        if self._unit is not None:
            desc_str += ',unit={}'.format(self._unit)
        return 'Signal[{}]'.format(desc_str)

    def __repr__(self):
        desc_str = 'freq={:.2f}_Hz,size={}'.format(self._freq, len(self._samples))
        if self._label is not None:
            desc_str += ',label="{}"'.format(self._label)
        if self._unit is not None:
            desc_str += ',unit={}'.format(self._unit)
        return 'Signal[{},adr={}]'.format(desc_str, id(self))

    def delete_samples_idx(self, from_idx, to_idx):
        num_samples = self._samples.size
        if not num_samples > 0:
            raise RuntimeError('Error, can not delete samples from empty sample set!')
        from_idx = 0 if from_idx < 0 else from_idx
        to_idx = num_samples if to_idx > num_samples else to_idx
        if not to_idx > from_idx:
            from_idx, to_idx = to_idx, from_idx
        if from_idx == 0 and to_idx == num_samples:
            raise RuntimeError('Error, can not delete entire signal!')
        new_samples = numpy.delete(self._samples, slice(from_idx, to_idx))
        indices = numpy.arange(from_idx, to_idx, dtype=int)
        sub_samples = numpy.take(self._samples, indices)
        del self._samples
        self._samples = new_samples
        return sub_samples

    def delete_samples(self, time_start, time_end):
        num_samples = self._samples.size
        if not num_samples > 0:
            raise RuntimeError('Error, can not delete samples from empty sample set!')
        if time_start > time_end:
            time_start, time_end = time_end, time_start
        delta_t = (1.0 / self._freq) * SEC_TO_MSEC
        if time_start < 0.0 and time_end > (num_samples-1)*delta_t:
            raise RuntimeError('Error, can not delete entire signal!')
        from_idx = int(numpy.floor(time_start/delta_t))
        to_idx = int(numpy.ceil(time_end / delta_t))+1
        return self.delete_samples_idx(from_idx, to_idx)

    def insert_samples_idx(self, idx, samples):
        num_samples = self._samples.size
        idx = min(num_samples, max(idx, 0))
        samples = numpy.array(samples, dtype=float)
        if samples.ndim > 1 and not samples.size > 0:
            raise ValueError('Error, can not insert samples, wrong shape or empty!')
        new_samples = numpy.insert(self._samples, idx, samples)
        del self._samples
        self._samples = new_samples

    def insert_samples(self, time, samples):
        num_samples = self._samples.size
        if not num_samples > 0:
            raise RuntimeError('Error, can not insert samples to empty sample set!')
        delta_t = (1.0 / self._freq) * SEC_TO_MSEC
        time = min((num_samples-1)*delta_t, max(time, 0.0))
        idx = int(numpy.ceil(time/delta_t))+1
        idx = num_samples if idx > num_samples else idx
        self.insert_samples_idx(idx, samples)

    def sub_samples_idx(self, from_idx, to_idx):
        num_samples = self._samples.size
        if not num_samples > 0:
            raise RuntimeError('Error, can not extract sub-samples from empty sample set!')
        from_idx = 0 if from_idx < 0 else from_idx
        to_idx = num_samples if to_idx > num_samples else to_idx
        if not to_idx > from_idx:
            from_idx, to_idx = to_idx, from_idx
        if from_idx == 0 and to_idx == num_samples:
            return numpy.copy(self._samples)
        return numpy.copy(self._samples[from_idx:to_idx])

    def sub_samples(self, time_start, time_end):
        num_samples = self._samples.size
        if not num_samples > 0:
            raise RuntimeError('Error, can not extract sub-samples from empty sample set!')
        if time_start > time_end:
            time_start, time_end = time_end, time_start
        delta_t = (1.0 / self._freq) * SEC_TO_MSEC
        if time_start < 0.0 and time_end > (num_samples-1)*delta_t:
            return numpy.copy(self._samples)
        from_idx = int(numpy.floor(time_start/delta_t))
        to_idx = int(numpy.ceil(time_end / delta_t))+1
        return self.sub_samples_idx(from_idx, to_idx)

    def resample_signal(self, frequency):
        raise NotImplementedError('Error, method "Signal.resample_signal(...)" not implemented!')

    def sub_signal_idx(self, from_idx, to_idx):
        sub_samples = self.sub_samples_idx(from_idx, to_idx)
        return Signal(self._freq, sub_samples, label=self._label,
                      unit=self._unit, copy_data=True)

    def sub_signal(self, time_start, time_end):
        sub_samples = self.sub_samples(time_start, time_end)
        return Signal(self._freq, sub_samples, label=self._label,
                      unit=self._unit, copy_data=True)

    @property
    def frequency(self):
        return self._freq

    @property
    def label(self):
        return self._label

    @property
    def unit(self):
        return self._unit

    @property
    def samples(self):
        return self._samples

    def remove_transition_phases(self):
        num_samples = self._samples.size
        if not num_samples > 0:
            return None
        extrema, extrema_cnt, samples = list(), 0, list()
        extrema.append((0, self._samples[0]))
        for i in range(1, num_samples-1):
            val = self._samples[i]
            if self._samples[i-1] < val and self._samples[i+1] <= val:
                extrema.append((i, val))
                extrema_cnt += 1
            elif self._samples[i-1] <= val and self._samples[i+1] < val:
                extrema.append((i, val))
                extrema_cnt += 1
            elif self._samples[i-1] > val and self._samples[i+1] >= val:
                extrema.append((i, val))
                extrema_cnt += 1
            elif self._samples[i-1] >= val and self._samples[i+1] > val:
                extrema.append((i, val))
                extrema_cnt += 1
        extrema.append((num_samples, self._samples[-1]))
        if not extrema_cnt > 0:
            return None
        num_extrema = len(extrema)
        for i in range(1, num_extrema):
            from_ex, to_ex = extrema[i-1], extrema[i]
            for j in range(from_ex[0], to_ex[0]):
                val = to_ex[1] if 2*j >= from_ex[0]+to_ex[0] else from_ex[1]
                samples.append(val)
        return Signal(self._freq, samples, label=self._label, unit=self._unit,
                      copy_data=False)

    def time_range(self, *, start_time=0.0):
        num_samples = self._samples.size
        if not num_samples > 0:
            raise RuntimeError('Error, can not compute end time for empty sample set!')
        delta_t = (1.0 / self._freq) * SEC_TO_MSEC
        return start_time, (start_time+(num_samples-1) * delta_t)

    def sample_range(self):
        num_samples = self._samples.size
        if not num_samples > 0:
            raise RuntimeError('Error, can not compute end time for empty sample set!')
        return numpy.min(self._samples), numpy.max(self._samples)

    def time(self):
        num_samples = self._samples.size
        if not num_samples > 0:
            raise RuntimeError('Error, can not generate times for empty sample set!')
        delta_t = (1.0 / self._freq) * SEC_TO_MSEC
        return numpy.linspace(0.0, (num_samples-1) * delta_t, num=num_samples, endpoint=True)

    @staticmethod
    def _is_semicolon_separated_csv_file(path):
        with open(path, 'r') as fp:
            fp.readline()  # read (possible) header line
            line = fp.readline().strip()
        return ';' in line

    @staticmethod
    def _load_signal_dat(path, /, label=None, unit=None, time_scale=MSEC_TO_SEC):
        if not os.path.exists(path):
            raise IOError('Error, path "{}" does not exist!'.format(path))
        if not os.path.isfile(path):
            raise IOError('Error, path "{}" is not a file!'.format(path))
        time, samples = numpy.loadtxt(path, dtype=float, comments='#',
                                      usecols=(0, 1), unpack=True)
        delta_t = numpy.average(time[1:]-time[:-1])
        freq = 1.0 / (delta_t * time_scale)
        return Signal(freq, samples, label=label, unit=unit, copy_data=False)

    @staticmethod
    def _load_signal_csv(path, *, label=None, unit=None, time_scale=MSEC_TO_SEC):
        if not os.path.exists(path):
            raise IOError('Error, path "{}" does not exist!'.format(path))
        if not os.path.isfile(path):
            raise IOError('Error, path "{}" is not a file!'.format(path))
        time, samples = numpy.loadtxt(path, dtype=float, comments='#',  delimiter=',',
                                      usecols=(0, 1), unpack=True)
        delta_t = numpy.average(time[1:] - time[:-1])
        freq = 1.0 / (delta_t * time_scale)
        return Signal(freq, samples, label=label, unit=unit, copy_data=False)

    @staticmethod
    def _load_signal_scsv(path, *, label=None, unit=None, time_scale=MSEC_TO_SEC):
        if not os.path.exists(path):
            raise IOError('Error, path "{}" does not exist!'.format(path))
        if not os.path.isfile(path):
            raise IOError('Error, path "{}" is not a file!'.format(path))
        time, samples = numpy.loadtxt(path, dtype=float, comments='#',  delimiter=';',
                                      usecols=(0, 1), unpack=True)
        delta_t = numpy.average(time[1:] - time[:-1])
        freq = 1.0 / (delta_t * time_scale)
        return Signal(freq, samples, label=label, unit=unit, copy_data=False)

    @staticmethod
    def _load_signal_excel(path, *, label=None, unit=None, time_scale=MSEC_TO_SEC):
        if not os.path.exists(path):
            raise IOError('Error, path "{}" does not exist!'.format(path))
        if not os.path.isfile(path):
            raise IOError('Error, path "{}" is not a file!'.format(path))
        time, samples = pandas.read_excel(path).to_numpy(dtype=float).T
        delta_t = numpy.average(time[1:] - time[:-1])
        freq = 1.0 / (delta_t * time_scale)
        return Signal(freq, samples, label=label, unit=unit, copy_data=False)

    @staticmethod
    def load_signal(path, *, label=None, unit=None, time_scale=MSEC_TO_SEC):
        if os.path.isfile(path):
            if callable(label):
                label = label(path)
            if path.endswith('.dat'):
                return Signal._load_signal_dat(path, label=label, unit=unit, time_scale=time_scale)
            elif path.endswith('.csv'):
                if Signal._is_semicolon_separated_csv_file(path):
                    return Signal._load_signal_scsv(path, label=label, unit=unit, time_scale=time_scale)
                else:
                    return Signal._load_signal_csv(path, label=label, unit=unit, time_scale=time_scale)
            elif path.endswith('.xls'):
                return Signal._load_signal_excel(path, label=label, unit=unit, time_scale=time_scale)
            else:
                raise IOError('Error, unknown file format "{}"!'.format(path))
        else:
            raise IOError('Error, path "{}" is not a file!'.format(path))

    def _save_signal_dat(self, path):
        unit = self._unit if self._unit is not None else '-'
        header = 'Time(ms) Samples({})'.format(unit)
        data = numpy.column_stack((self.time(), self._samples))
        numpy.savetxt(path, data, delimiter=' ', header=header, fmt='%.16f')

    def _save_signal_csv(self, path):
        unit = self._unit if self._unit is not None else '-'
        header = 'Time(ms),Current({})'.format(unit)
        data = numpy.column_stack((self.time(), self._samples))
        numpy.savetxt(path, data, delimiter=',', header=header, fmt='%.16f')

    def _save_signal_scsv(self, path):
        unit = self._unit if self._unit is not None else '-'
        header = 'Time(ms);Current({})'.format(unit)
        data = numpy.column_stack((self.time(), self._samples))
        numpy.savetxt(path, data, delimiter=';', header=header, fmt='%.16f')

    def _save_signal_json(self, path):
        jdata = self.to_json()
        with open(path, 'w') as fp:
            json.dump(jdata, fp, indent=2)

    def _save_signal_excel(self, path):
        data = numpy.column_stack((self.time(), self._samples))
        header = ['Time (ms)', 'Current (pA)']
        data_frame = pandas.DataFrame(data)
        data_frame.to_excel(path, index=False, header=header)

    def save_signal(self, path, *, scsv=False):
        if path.endswith('.dat'):
            self._save_signal_dat(path)
        elif path.endswith('.csv'):
            if scsv:
                self._save_signal_scsv(path)
            else:
                self._save_signal_csv(path)
        elif path.endswith('.json'):
            self._save_signal_json(path)
        elif path.endswith('.xlsx'):
            self._save_signal_excel(path)
        else:
            raise IOError('Error, unknown file format "{}"!'.format(path))

    def _save_histogram_dat(self, path, bins, *, density=False, weights=None):
        hist, edges = numpy.histogram(self._samples, bins=bins, density=density, weights=weights)
        edge_from, edge_to = edges[:-1], edges[1:]
        header = 'EdgeFrom({}) EdgeTo({}) Count(1)'.format(self._unit, self._unit)
        data = numpy.column_stack((edge_from, edge_to, hist))
        numpy.savetxt(path, data, delimiter=' ', header=header, fmt=('%.8f', '%.8f', '%d'))

    def _save_histogram_csv(self, path, bins, *, density=False, weights=None):
        hist, edges = numpy.histogram(self._samples, bins=bins, density=density, weights=weights)
        edge_from, edge_to = edges[:-1], edges[1:]
        header = 'EdgeFrom({}),EdgeTo({}),Count(1)'.format(self._unit, self._unit)
        data = numpy.column_stack((edge_from, edge_to, hist))
        numpy.savetxt(path, data, delimiter=',', header=header, fmt=('%.8f', '%.8f', '%d'))

    def _save_histogram_scsv(self, path, bins, *, density=False, weights=None):
        hist, edges = numpy.histogram(self._samples, bins=bins, density=density, weights=weights)
        edge_from, edge_to = edges[:-1], edges[1:]
        header = 'EdgeFrom({});EdgeTo({});Count(1)'.format(self._unit, self._unit)
        data = numpy.column_stack((edge_from, edge_to, hist))
        numpy.savetxt(path, data, delimiter=';', header=header, fmt=('%.8f', '%.8f', '%d'))

    def _save_histogram_excel(self, path, bins, *, density=False, weights=None):
        hist, edges = numpy.histogram(self._samples, bins=bins, density=density, weights=weights)
        edge_from, edge_to = edges[:-1], edges[1:]
        header = ['EdgeFrom({})'.format(self._unit), 'EdgeTo({})'.format(self._unit), 'Count(1)']
        data = numpy.column_stack((edge_from, edge_to, hist))
        data_frame = pandas.DataFrame(data)
        data_frame.to_excel(path, index=False, header=header)

    def save_histogram(self, path, bins, *, density=False, weights=None, scsv=False):
        if path.endswith('.dat'):
            self._save_histogram_dat(path, bins, density=density, weights=weights)
        elif path.endswith('.csv'):
            if scsv:
                self._save_histogram_scsv(path, bins, density=density, weights=weights)
            else:
                self._save_histogram_csv(path, bins, density=density, weights=weights)
        elif path.endswith('.xls'):
            self._save_histogram_excel(path, bins, density=density, weights=weights)
        else:
            raise IOError('Error, unknown file format "{}"!'.format(path))

    def histogram(self, bins, *, density=False, weights=None, step_function=True):
        counts, edges = numpy.histogram(self._samples, bins=bins, density=density, weights=weights)
        if step_function:
            centers, values = list([edges[0]]), list([0.0])
            for i, value in enumerate(counts):
                centers += [edges[i], edges[i+1]]
                values += [value, value]
            centers.append(edges[-1])
            values.append(0.0)
            centers = numpy.array(centers, dtype=float)
            return centers, values
        else:
            values = numpy.array(counts, dtype=float)
            values = numpy.insert(values, 0, 0.0)
            values = numpy.append(values, 0.0)
            centers = numpy.array((edges[1:]+edges[:-1])*0.5, dtype=float)
            centers = numpy.insert(centers, 0, (3.0*edges[0]-edges[1])*0.5)
            centers = numpy.append(centers, (3.0*edges[-1]-edges[-2])*0.5)
            return centers, values

    def baseline_correction(self, bins):
        counts, edges = numpy.histogram(self._samples, bins=bins, density=False, weights=None)
        idx_max = numpy.argmax(counts)
        baseline_value = (edges[idx_max]+edges[idx_max+1])*0.5
        self._samples -= baseline_value

    def to_json(self):
        jdata = {'frequency': self._freq}
        if self._label is not None:
            jdata['label'] = self._label
        if self._unit is not None:
            jdata['unit'] = self._unit
        jdata['samples'] = list(self._samples)
        return jdata

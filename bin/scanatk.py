#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
# FILE: scanatk.py
# AUTHOR: Matthias A.F. Gsell
# EMAIL: matthias.gsell@medunigraz.at
# DATE: 02.2023
"""

import os
import json
import numpy
import pandas
import itertools

import tkinter
import tkinter.ttk
import tkinter.messagebox
import tkinter.simpledialog
import tkinter.filedialog
import tkinter.colorchooser

import matplotlib.pyplot as mplplot
import matplotlib.widgets as mplwidgets
import matplotlib.collections as mplcoll
import matplotlib.gridspec as mplgridspec
import matplotlib.backend_bases as mplbackendbases
import matplotlib.backends.backend_tkagg as mplbackend


import scana

"""
def numpy_error_callback(err, flag):
    print(err, flag)

np.seterr(all='call')
np.seterrcall(numpy_error_callback)
"""

FILE_PATH = os.path.abspath(os.path.dirname(__file__))


class Selection:

    def __init__(self, signal_artist, x_min, x_max, axes):
        self.signal_artist = signal_artist
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.artist = axes.axvspan(self.x_min, self.x_max, alpha=0.15, color='#AA00AA')

    def clear(self):
        self.signal_artist = None
        self.x_min = 0.0
        self.x_max = 0.0
        if self.artist is not None:
            self.artist.remove()
        self.artist = None


class FittingResultWindow(tkinter.Toplevel):

    OUT_PRECISION = 8

    def __init__(self, result, master=None, title='message', width=1024, height=768, pos_x=100, pos_y=100):
        self._result = result

        tkinter.Toplevel.__init__(self, master=master)
        width, height = int(width), int(height)
        pos_x, pos_y = int(pos_x), int(pos_y)
        self.title(str(title))
        self.geometry('{}x{}+{}+{}'.format(width, height, pos_x, pos_y))
        self._text = tkinter.Text(self)
        self._text.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True, padx=5, pady=5)
        self._text.config(state=tkinter.DISABLED)

        self._format_button = tkinter.Button(self, text='CSV format', command=self._format_button_click)
        self._format_button.pack(side=tkinter.LEFT, fill=tkinter.Y, expand=False, padx=5, pady=5)
        self._format = 0

        self._save_button = tkinter.Button(self, text='Save As', command=self._save_as_button_click)
        self._save_button.pack(side=tkinter.LEFT, fill=tkinter.Y, expand=False, padx=5, pady=5)

        self._print_result()

    def _format_button_click(self):
        if self._format == 0:
            self._format = 1
            self._format_button.config(text='SCSV format')
        elif self._format == 1:
            self._format = 2
            self._format_button.config(text='TXT format')
        else:
            self._format = 0
            self._format_button.config(text='CSV format')
        self._text.config(state=tkinter.NORMAL)
        self._text.delete("1.0", tkinter.END)
        self._text.config(state=tkinter.DISABLED)
        self._print_result()

    def _save_result_dat(self, path):
        header = 'level mu sigma rho area'
        datafmt = '%d %.16f %.16f %.16f %.16f'
        data = numpy.array(self._result[1], dtype=float)
        with open(path, 'w') as fp:
            numpy.savetxt(fp, data, delimiter=' ', header=header, fmt=datafmt)
            fp.write('\n{:.16f}\n'.format(self._result[2]))

    def _save_result_scsv(self, path):
        header = 'level;mu;sigma;rho;area'
        datafmt = ('%d', '%.16f', '%.16f', '%.16f', '%.16f')
        data = numpy.array(self._result[1], dtype=float)
        with open(path, 'w') as fp:
            numpy.savetxt(fp, data, delimiter=';', header=header, fmt=datafmt)
            fp.write(';;;;{:.16f}\n'.format(self._result[2]))

    def _save_result_csv(self, path):
        header = 'level,mu,sigma,rho,area'
        datafmt = ('%d', '%.16f', '%.16f', '%.16f', '%.16f')
        data = numpy.array(self._result[1], dtype=float)
        with open(path, 'w') as fp:
            numpy.savetxt(fp, data, delimiter=',', header=header, fmt=datafmt)
            fp.write(',,,,{:.16f}\n'.format(self._result[2]))

    def _save_result_excel(self, path):
        header = ['level', 'mu', 'sigma', 'rho', 'area']
        data_frame = pandas.DataFrame(self._result[1])
        data_frame.to_excel(path, index=False, header=header)

    def save_result(self, path, *, scsv=False):
        if path.endswith('.dat'):
            self._save_result_dat(path)
        elif path.endswith('.csv'):
            if scsv:
                self._save_result_scsv(path)
            else:
                self._save_result_csv(path)
        elif path.endswith('.xlsx'):
            self._save_result_excel(path)
        else:
            raise IOError('Error, unknown file format "{}"!'.format(path))

    def _save_as_button_click(self):
        file_types = (('All files', '*.*'),
                      ('Data files', '*.dat'),
                      ('CSV files', '*.csv'),
                      ('Excel files', '*.xls'))
        """
        kwargs = dict()
        if self._last_dir is not None and os.path.exists(self._last_dir):
            kwargs['initialdir'] = self._last_dir
        """
        file_name = tkinter.filedialog.asksaveasfilename(parent=self, filetypes=file_types)
        if file_name is not None and len(file_name) > 0:
            is_scsv = file_name.endswith('.csv') and \
                      tkinter.messagebox.askyesno('Delimiter', 'Use semicolon as delimiter?')
            try:
                self.save_result(file_name, scsv=is_scsv)
            except RuntimeError as ex:
                tkinter.messagebox.showerror(parent=self.master, title='Error', message=str(ex))

    def add_message(self, msg):
        self._text.config(state=tkinter.NORMAL)
        self._text.insert(tkinter.END, msg)
        self._text.see(tkinter.END)  # Scroll to the bottom
        self._text.config(state=tkinter.DISABLED)
        self._text.update()  # Refresh the widget

    def _print_result(self):
        if self._format == 0:
            self._print_result_txt()
        elif self._format == 1:
            self._print_result_csv()
        else:
            self._print_result_scsv()

    def _print_result_txt(self):
        label, data, tot_area = self._result
        self.add_message('Fitting "{}", Levels={}\n\n'.format(label, len(data)))
        flt_fmt = '{{:.{}f}}'.format(FittingResultWindow.OUT_PRECISION)
        msg_fmt = '  level {}: mu='+flt_fmt+', sigma='+flt_fmt+', rho='+flt_fmt+', area='+flt_fmt+'\n'
        for lvl_data in data:
            msg = msg_fmt.format(*lvl_data)
            self.add_message(msg)
        self.add_message(('total area='+flt_fmt+'\n').format(tot_area))

    def _print_result_csv(self):
        label, data, tot_area = self._result
        self.add_message('Fitting "{}", Levels={}\n\n'.format(label, len(data)))
        header = 'level,mu,sigma,rho,area\n'
        self.add_message(header)
        flt_fmt = '{{:.{}f}}'.format(FittingResultWindow.OUT_PRECISION)
        msg_fmt = '{},'+flt_fmt+','+flt_fmt+','+flt_fmt+','+flt_fmt+'\n'
        for lvl_data in data:
            msg = msg_fmt.format(*lvl_data)
            self.add_message(msg)
        self.add_message((',,,,'+flt_fmt+'\n').format(tot_area))

    def _print_result_scsv(self):
        label, data, tot_area = self._result
        self.add_message('Fitting "{}", Levels={}\n\n'.format(label, len(data)))
        header = 'level;mu;sigma;rho;area\n'
        self.add_message(header)
        flt_fmt = '{{:.{}f}}'.format(FittingResultWindow.OUT_PRECISION)
        msg_fmt = '{};'+flt_fmt+';'+flt_fmt+';'+flt_fmt+';'+flt_fmt+'\n'
        for lvl_data in data:
            msg = msg_fmt.format(*lvl_data)
            self.add_message(msg)
        self.add_message((';;;;'+flt_fmt+'\n').format(tot_area))


class Application:

    VERBOSE = 0
    COLOR_ITER = itertools.cycle(mplplot.rcParams['axes.prop_cycle'].by_key()['color'])
    STATE_FILE = os.path.join(FILE_PATH, 'scana.json')

    def __init__(self, title='scana', width=1024, height=768, pos_x=100, pos_y=100):
        width, height = int(width), int(height)
        pos_x, pos_y = int(pos_x), int(pos_y)
        self._axes_ratio = 0.9

        root = tkinter.Tk()
        root.title(str(title))
        root.geometry('{}x{}+{}+{}'.format(width, height, pos_x, pos_y))
        self._root = root
        # last directory in open-file dialog
        self._last_dir = FILE_PATH
        self._last_fit_level = 4
        # general widget variables
        self._hist_log_scale = tkinter.BooleanVar()
        self._plot_all_gaussians = tkinter.BooleanVar()
        self._signal_visible = tkinter.BooleanVar()
        self._density_hist = tkinter.BooleanVar()
        self._histogram_bins = tkinter.IntVar()
        self._filter_selected = tkinter.StringVar()
        self._filter_lpfreq = tkinter.DoubleVar()
        self._filter_hpfreq = tkinter.DoubleVar()
        self._filter_order = tkinter.IntVar()
        # general widgets
        self._file_menu = None
        self._edit_menu = None
        self._view_menu = None
        self._signals_listbox = None
        self._remove_button = None
        self._baseline_correction_button = None
        self._remove_transition_phases_button = None
        self._fit_histogram_button = None
        self._remove_histogram_fit_button = None
        self._visible_checkbutton = None
        self._density_checkbutton = None
        self._histogram_bins_scale = None
        self._histogram_update_button = None
        # matplotlib widgets
        self._figure = mplplot.figure()
        self._canvas = None
        self._ax_signal = None
        self._ax_signal_xpos = None
        self._ax_histogram = None
        self._span = None
        # open signal
        self._signals = list()
        self._signal_selected = None
        # selection and clipboard
        self._selection = None
        self._clipboard = None
        # load application state
        self._load_app_state()
        # initialize application
        self._init_widgets()
        self._init_figure(size_ratio=self._axes_ratio)
        self._update_gui()
        # update scaling
        self._hist_log_scale_button_click()

    def _save_app_state(self):
        json_data = dict(last_dir=self._last_dir, win_width=self._root.winfo_width(),
                         win_height=self._root.winfo_height(), axes_ratio=self._axes_ratio,
                         hist_log_scale=self._hist_log_scale.get(),
                         plot_all_gaussians=self._plot_all_gaussians.get(), list_fit_level=self._last_fit_level)
        with open(Application.STATE_FILE, 'w') as fp:
            json.dump(json_data, fp, indent=2)

    def _load_app_state(self):
        if not os.path.exists(Application.STATE_FILE):
            return
        with open(Application.STATE_FILE, 'r') as fp:
            json_data = json.load(fp)
        width = json_data.get('win_width', None)
        height = json_data.get('win_height', None)
        if width is not None and height is not None and width > 0 and height > 0:
            self._root.geometry('{}x{}'.format(width, height))
        last_dir = json_data.get('last_dir', None)
        if last_dir is not None and os.path.isdir(last_dir):
            self._last_dir = last_dir
        axes_ratio = json_data.get('axes_ratio', None)
        if axes_ratio is not None and 0.0 < axes_ratio < 1.0:
            self._axes_ratio = axes_ratio
        self._hist_log_scale.set(json_data.get('hist_log_scale', False))
        self._plot_all_gaussians.set(json_data.get('plot_all_gaussians', False))
        self._last_fit_level = json_data.get('list_fit_level', 4)

    def _save_state(self, file_name):
        jdata = list()
        for i, signal in enumerate(self._signals):
            signal_jdata = signal.to_json()
            signal_jdata['label'] = self._signals_listbox.get(i)
            jdata.append(signal_jdata)
        with open(file_name, 'w') as fp:
            json.dump(jdata, fp, indent=2)

    def _load_state(self, file_name):
        print('TODO: implement "Application._load_state(...)"!')

    def _open_signal(self, file_name):
        try:
            label_func = lambda s: os.path.splitext(os.path.basename(s))[0]

            signal = scana.Signal.load_signal(file_name, label=label_func)
            signal_artist = scana.SignalArtist(signal, self._ax_signal, self._ax_histogram)

            time_start, time_end = self._ax_signal.get_xlim()
            signal_artist.update_zoom(time_start, time_end)
            signal_artist.set_color(next(Application.COLOR_ITER))

            self._signals.append(signal_artist)
            self._signal_selected = signal_artist

            self._last_dir = os.path.dirname(file_name)
            basename = os.path.splitext(os.path.basename(file_name))[0]

            self._signals_listbox.insert(tkinter.END, '{} ({:.2f} Hz)'.format(basename, signal.frequency))
            self._signals_listbox.selection_clear(0, self._signals_listbox.size()-1)
            self._signals_listbox.selection_set(self._signals_listbox.size()-1)

            # clear selection
            if self._selection is not None:
                self._selection.clear()
                self._selection = None

        except RuntimeError as ex:
            tkinter.messagebox.showerror(parent=self._root, title='RuntimeError', message=str(ex))
        except OSError as ex:
            tkinter.messagebox.showerror(parent=self._root, title='OSError', message=str(ex))
        except ValueError as ex:
            tkinter.messagebox.showerror(parent=self._root, title='ValueError', message=str(ex))
        else:
            self._update_gui()
            self._update_plot(relim_signal=True, relim_histogram=True)

    def _dummy_callback(self, *args):
        pass

    def _refresh_view_button_click(self):
        self._update_plot(relim_signal=True, relim_histogram=True, scalex_signal=False,
                          scaley_signal=False, scalex_histogram=False, scaley_histogram=False)

    def _open_signal_button_click(self):
        file_types = (('All files', '*.*'),
                      ('Data files', '*.dat'),
                      ('CSV files', '*.csv'),
                      ('Excel files', '*.xls'))
        kwargs = dict()
        if self._last_dir is not None and os.path.exists(self._last_dir):
            kwargs['initialdir'] = self._last_dir

        file_name = tkinter.filedialog.askopenfilename(parent=self._root, filetypes=file_types, **kwargs)
        if file_name is not None and len(file_name) > 0 and os.path.exists(file_name):
            self._open_signal(file_name)

    def _save_signal_button_click(self):
        file_types = (('All files', '*.*'),
                      ('Data files', '*.dat'),
                      ('CSV files', '*.csv'),
                      ('JSON files', '*.json'))
        kwargs = dict()
        if self._last_dir is not None and os.path.exists(self._last_dir):
            kwargs['initialdir'] = self._last_dir
        file_name = tkinter.filedialog.asksaveasfilename(parent=self._root, filetypes=file_types)

        if file_name is not None and len(file_name) > 0:
            is_scsv = file_name.endswith('.csv') and \
                      tkinter.messagebox.askyesno('Delimiter', 'Use semicolon as delimiter?')
            try:
                self._signal_selected.signal.save_signal(file_name, scsv=is_scsv)
            except RuntimeError as ex:
                tkinter.messagebox.showerror(parent=self._root, title='Error', message=str(ex))

    def _save_histogram_button_click(self):
        file_types = (('All files', '*.*'),
                      ('Data files', '*.dat'),
                      ('CSV files', '*.csv'),
                      ('Excel files', '*.xls'))
        kwargs = dict()
        if self._last_dir is not None and os.path.exists(self._last_dir):
            kwargs['initialdir'] = self._last_dir

        file_name = tkinter.filedialog.asksaveasfilename(parent=self._root, filetypes=file_types)
        if file_name is not None and len(file_name) > 0:
            is_scsv = file_name.endswith('.csv') and \
                      tkinter.messagebox.askyesno('Delimiter', 'Use semicolon as delimiter?')
            try:
                self._signal_selected.signal.save_histogram(file_name, self._signal_selected.get_bins(), scsv=is_scsv)
            except RuntimeError as ex:
                tkinter.messagebox.showerror(parent=self._root, title='Error', message=str(ex))

    def _open_state_button_click(self):
        file_types = (('All files', '*.*'),
                      ('JSON files', '*.json'))
        kwargs = dict()
        if self._last_dir is not None and os.path.exists(self._last_dir):
            kwargs['initialdir'] = self._last_dir

        file_name = tkinter.filedialog.askopenfilename(parent=self._root, filetypes=file_types, **kwargs)
        if file_name is not None and len(file_name) > 0 and os.path.exists(file_name):
            self._load_state(file_name)

    def _save_state_button_click(self):
        file_types = (('All files', '*.*'),
                      ('JSON files', '*.json'))
        kwargs = dict()
        if self._last_dir is not None and os.path.exists(self._last_dir):
            kwargs['initialdir'] = self._last_dir
        file_name = tkinter.filedialog.asksaveasfilename(parent=self._root, filetypes=file_types)
        if file_name is not None and len(file_name) > 0:
            try:
                self._save_state(file_name)
            except RuntimeError as ex:
                tkinter.messagebox.showerror(parent=self._root, title='Error', message=str(ex))

    def _clear_state_button_click(self):
        self._signals_listbox.delete(0, tkinter.END)
        self._signal_selected = None
        self._signals = list()
        if self._selection is not None:
            self._selection.clear()
            self._selection = None
        self._clipboard = None
        self._update_gui()
        self._update_plot(relim_signal=True, relim_histogram=True)

    def _exit_button_click(self):
        self._save_app_state()
        self._root.quit()
        self._root.destroy()

    def _delete_selection_button_click(self):
        if self._selection is None:
            return
        signal_artist = self._selection.signal_artist
        try:
            time_start, time_end = float(self._selection.x_min), float(self._selection.x_max)
            signal_artist.push_signal()
            signal_artist.signal.delete_samples(time_start, time_end)
        except RuntimeError as ex:
            tkinter.messagebox.showerror(parent=self._root, title='Error', message=str(ex))
        else:
            # update artists
            signal_artist.update_artists()
        finally:
            # reset selection
            self._selection.clear()
            self._selection = None
            # update GUI and plots
            self._update_gui()
            self._update_plot(relim_signal=True, relim_histogram=True)

    def _clear_clipboard_button_click(self):
        self._clipboard = None
        if self._selection is not None:
            self._selection.clear()
            self._selection = None
        # update GUI and plots
        self._update_gui()
        self._update_plot(relim_signal=False, relim_histogram=False)

    def _undo_button_click(self):
        if self._signal_selected is None:
            return
        if self._signal_selected.pop_signal():
            # update artists
            self._signal_selected.update_artists()
            # update GUI and plots
            self._update_gui()
            self._update_plot(relim_signal=True, relim_histogram=True)

    def _change_color_button_click(self):
        if self._signal_selected is None:
            return
        color = self._signal_selected.get_color()
        hex_color = tkinter.colorchooser.askcolor(parent=self._root, initialcolor=color)[1]
        if hex_color is not None:
            self._signal_selected.set_color(hex_color)
            self._update_plot()

    def _change_alpha_value_button_click(self):
        if self._signal_selected is None:
            return
        initial_value = self._signal_selected.get_alpha()
        title = 'Select Alpha Value'
        prompt = 'Min: 0.0, Max: 1.0'
        value = tkinter.simpledialog.askfloat(parent=self._root, title=title, prompt=prompt,
                                              minvalue=0.0, maxvalue=1.0,
                                              initialvalue=initial_value)
        if value is not None:
            self._signal_selected.set_alpha(value)
            self._update_plot()

    def _change_line_width_button_click(self):
        if self._signal_selected is None:
            return
        initial_value = self._signal_selected.get_linewidth()
        title = 'Select Line Width'
        prompt = 'Min: 0.1, Max: 10.0'
        value = tkinter.simpledialog.askfloat(parent=self._root, title=title, prompt=prompt,
                                              minvalue=0.1, maxvalue=10.0,
                                              initialvalue=initial_value)
        if value is not None:
            self._signal_selected.set_linewidth(value)
            self._update_plot()

    def _show_only_selected_button_click(self):
        for signal in self._signals:
            signal.set_visible(False)
        if self._signal_selected is not None:
            self._signal_selected.set_visible(True)
        # update GUI and plots
        self._update_gui()
        self._update_plot(relim_histogram=True)

    def _show_all_button_click(self):
        for signal in self._signals:
            signal.set_visible(True)
        # update GUI and plots
        self._update_gui()
        self._update_plot(relim_histogram=True)

    def _hist_log_scale_button_click(self):
        if self._hist_log_scale.get():
            self._ax_histogram.set_xscale('log')
        else:
            self._ax_histogram.set_xscale('linear')
        self._update_plot(relim_histogram=True)

    """
    def _hist_prop_density_button_click(self):
        for signal in self._signals:
            signal.set_density(self._hist_prop_density.get())
        self._update_plot(relim_histogram=True)
    """

    def _signals_listbox_select(self, event):
        selection = self._signals_listbox.curselection()
        if len(selection) == 0:
            return
        self._signal_selected = self._signals[selection[0]]
        if self._signal_selected.get_visible():
            self._signal_selected.to_front()
        # clear selection
        if self._selection is not None:
            self._selection.clear()
            self._selection = None
        # update GUI and plots
        self._update_gui()
        self._update_plot(relim_histogram=True)

    def _visible_checkbutton_click(self):
        if self._signal_selected is None:
            return
        visible = self._signal_visible.get()
        self._signal_selected.set_visible(visible)
        if visible:
            self._signal_selected.to_front()
        self._update_plot(relim_histogram=True)

    def _density_histogram_checkbutton_click(self):
        if self._signal_selected is None:
            return
        density = self._density_hist.get()
        self._signal_selected.set_density(density)
        self._update_plot(relim_histogram=True)

    def _remove_signal_button_click(self):
        if self._signal_selected is None:
            return
        index = self._signals.index(self._signal_selected)
        self._signals_listbox.delete(index)
        self._signals.pop(index)
        # del self._signal_selected
        self._signal_selected = None
        # clear selection
        if self._selection is not None:
            self._selection.clear()
            self._selection = None
        # update GUI and plots
        self._update_gui()
        self._update_plot(relim_histogram=True)

    def _baseline_correction_button_click(self):
        if self._signal_selected is None:
            return
        self._signal_selected.do_baseline_correction()
        # update GUI and plots
        self._update_gui()
        self._update_plot(relim_signal=True, relim_histogram=True)

    def _remove_transition_phases_button_click(self):
        if self._signal_selected is None:
            return
        signal = self._signal_selected.signal.remove_transition_phases()
        if signal is None:
            return
        signal_artist = scana.SignalArtist(signal, self._ax_signal, self._ax_histogram)

        time_start, time_end = self._ax_signal.get_xlim()
        signal_artist.update_zoom(time_start, time_end)
        signal_artist.set_color(next(Application.COLOR_ITER))

        self._signals.append(signal_artist)
        self._signal_selected = signal_artist

        self._signals_listbox.insert(tkinter.END, '{} ({:.2f} Hz)'.format(signal.label, signal.frequency))
        self._signals_listbox.selection_clear(0, self._signals_listbox.size() - 1)
        self._signals_listbox.selection_set(self._signals_listbox.size() - 1)

        # clear selection
        if self._selection is not None:
            self._selection.clear()
            self._selection = None

        self._update_gui()
        self._update_plot(relim_signal=True, relim_histogram=True)

    def _fit_histogram_button_click(self):
        if self._signal_selected is None:
            return

        title = 'Levels'
        prompt = 'Min: 1, Max: 10'
        value = tkinter.simpledialog.askinteger(parent=self._root, title=title, prompt=prompt,
                                                minvalue=1, maxvalue=10,
                                                initialvalue=self._last_fit_level)
        if value is not None:
            self._last_fit_level = value
            plotall = self._plot_all_gaussians.get()
            rval = self._signal_selected.update_histogram_fit(order=value, plot_all=plotall)
            if rval is not None:
                FittingResultWindow(rval, master=self._root, title='fitting result',
                                    width=640, height=480)

            self._update_gui()
            self._update_plot(relim_signal=False, relim_histogram=False)

    def _remove_histogram_fit_button_click(self):
        if self._signal_selected is None:
            return
        self._signal_selected.remove_histogram_fit()

        self._update_gui()
        self._update_plot(relim_signal=False, relim_histogram=True)

    def _histogram_update_button_click(self):
        if self._signal_selected is None:
            return
        bins = int(self._histogram_bins.get())
        self._signal_selected.set_bins(bins)
        # update plots
        self._update_plot(relim_histogram=True)

    def _about_button_click(self):
        msg = 'scana V1.0\n' \
              'Author: Matthias A.F. Gsell\n' \
              'E-Mail: matthias.gsell@medunigraz.at\n' \
              'Date: 02.2023'
        tkinter.simpledialog.messagebox.showinfo(parent=self._root,
                                                 title='About', message=msg)

    def _on_mouse_button_release(self, event):
        if self._selection is not None:
            # reset selection
            self._selection.clear()
            self._selection = None

        if self._signal_selected is None or self._clipboard is None:
            return

        if event.inaxes is not None and event.inaxes is self._ax_signal and \
                event.button == mplbackendbases.MouseButton.MIDDLE:
            signal_artist = self._signal_selected
            try:
                time_insert = event.xdata
                signal_artist.push_signal()
                signal_artist.signal.insert_samples(time_insert, self._clipboard[0])
                if self._clipboard[1] != signal_artist.signal.frequency:
                    msg = 'Frequency mismatch!'
                    tkinter.messagebox.showwarning(parent=self._root, title='Warning',
                                                   message=msg)
                # self._clipboard = None
            except RuntimeError as ex:
                signal_artist.pop_signal(apply=False)
                tkinter.messagebox.showerror(parent=self._root, title='Error', message=str(ex))
            else:
                # update artists
                signal_artist.update_artists()
            finally:
                # update GUI and plots
                self._update_gui()
                self._update_plot(relim_signal=True, relim_histogram=True)

    def _on_mouse_move(self, event):
        self._ax_signal_xpos = None
        if event.inaxes is not None and event.inaxes is self._ax_signal:
            self._ax_signal_xpos = event.xdata

    def _on_key_release(self, event):
        if event.inaxes is not None and event.inaxes is self._ax_signal:
            if event.key in ('d', 'delete'):
                self._delete_selection_button_click()
            elif event.key in ('i', 'insert'):
                if self._selection is not None:
                    # reset selection
                    self._selection.clear()
                    self._selection = None

                if self._signal_selected is None or self._clipboard is None or self._ax_signal_xpos is None:
                    return

                signal_artist = self._signal_selected
                try:
                    time_insert = self._ax_signal_xpos
                    signal_artist.push_signal()
                    signal_artist.signal.insert_samples(time_insert, self._clipboard[0])
                    if self._clipboard[1] != signal_artist.signal.frequency:
                        msg = 'Frequency mismatch!'
                        tkinter.messagebox.showwarning(parent=self._root, title='Warning',
                                                       message=msg)
                    # self._clipboard = None
                except RuntimeError as ex:
                    signal_artist.pop_signal(apply=False)
                    tkinter.messagebox.showerror(parent=self._root, title='Error', message=str(ex))
                else:
                    # update artists
                    signal_artist.update_artists()
                finally:
                    # update GUI and plots
                    self._update_gui()
                    self._update_plot(relim_signal=True, relim_histogram=True)

    def _on_x_lims_change(self, event_axes):
        if not len(self._signals) > 0:
            return
        if event_axes is not None and event_axes is self._ax_signal:
            time_start, time_end = event_axes.get_xlim()
            for signal in self._signals:
                signal.update_zoom(time_start, time_end)
            self._update_plot(relim_histogram=True)

    def _init_menu_bar(self):
        menu_bar = tkinter.Menu(self._root, tearoff=0)

        file_menu = tkinter.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label='Open Signal', command=self._open_signal_button_click)
        file_menu.add_command(label='Save Signal', command=self._save_signal_button_click,
                              state=tkinter.NORMAL)
        file_menu.add_command(label='Save Histogram', command=self._save_histogram_button_click)
        file_menu.add_separator()
        file_menu.add_command(label='Open State', command=self._open_state_button_click)
        file_menu.add_command(label='Save State', command=self._save_state_button_click)
        file_menu.add_command(label='Clear State', command=self._clear_state_button_click)
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self._exit_button_click)
        menu_bar.add_cascade(label='File', menu=file_menu)

        edit_menu = tkinter.Menu(menu_bar, tearoff=0)
        edit_menu.add_command(label='Delete Selection', command=self._delete_selection_button_click)
        edit_menu.add_command(label='Clear Clipboard', command=self._clear_clipboard_button_click)
        edit_menu.add_separator()
        edit_menu.add_command(label='Undo', command=self._undo_button_click)
        edit_menu.add_separator()
        edit_menu.add_command(label='Change Color', command=self._change_color_button_click)
        edit_menu.add_command(label='Change Alpha Value', command=self._change_alpha_value_button_click)
        edit_menu.add_command(label='Change Line Width', command=self._change_line_width_button_click)
        menu_bar.add_cascade(label='Edit', menu=edit_menu)

        view_menu = tkinter.Menu(menu_bar, tearoff=0)
        view_menu.add_command(label='Show selected only', command=self._show_only_selected_button_click)
        view_menu.add_command(label='Show all', command=self._show_all_button_click)
        view_menu.add_separator()
        view_menu.add_checkbutton(label='Histogram Log-Scale', onvalue=1, offvalue=0,
                                  variable=self._hist_log_scale, command=self._hist_log_scale_button_click)
        view_menu.add_separator()
        view_menu.add_checkbutton(label='Plot all Gaussians', onvalue=1, offvalue=0,
                                  variable=self._plot_all_gaussians)
        menu_bar.add_cascade(label='View', menu=view_menu)

        menu_bar.add_command(label='About', command=self._about_button_click)

        self._root.config(menu=menu_bar)

        self._file_menu = file_menu
        self._edit_menu = edit_menu
        self._view_menu = view_menu

    def _init_widgets(self):
        self._init_menu_bar()

        main_frame = tkinter.Frame(self._root, width=300, borderwidth=5)
        main_frame.pack(side=tkinter.LEFT, fill=tkinter.Y)
        main_frame.pack_propagate(False)

        # ===========================================================
        # SIGNALS FRAME
        # ===========================================================

        signal_frame = tkinter.LabelFrame(main_frame, text='Signals')
        signal_frame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        signals_listbox = tkinter.Listbox(signal_frame, selectmode=tkinter.SINGLE)
        signals_listbox.bind('<<ListboxSelect>>', self._signals_listbox_select)
        signals_listbox.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        # Remove Signal
        remove_button = tkinter.Button(signal_frame, text='Remove', command=self._remove_signal_button_click)
        remove_button.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        # Baseline Correction
        baseline_correction_button = tkinter.Button(signal_frame, text='Baseline Correction',
                                                    command=self._baseline_correction_button_click)
        baseline_correction_button.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        # Remove Transition Phases
        remove_transition_phases_button = tkinter.Button(signal_frame, text='Remove Transition Phases',
                                                         command=self._remove_transition_phases_button_click)
        remove_transition_phases_button.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        # Fit histogram
        fit_histogram_button = tkinter.Button(signal_frame, text='Fit Histogram',
                                              command=self._fit_histogram_button_click)
        fit_histogram_button.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        # Remove histogram fit
        remove_histogram_fit_button = tkinter.Button(signal_frame, text='Remove Histogram Fit',
                                                     command=self._remove_histogram_fit_button_click)
        remove_histogram_fit_button.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        # Visibility
        visible_checkbox = tkinter.Checkbutton(signal_frame, text='Visible', variable=self._signal_visible,
                                               anchor=tkinter.W, command=self._visible_checkbutton_click)
        visible_checkbox.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        # Density-Histogram
        density_checkbox = tkinter.Checkbutton(signal_frame, text='Density-Histogram', variable=self._density_hist,
                                               anchor=tkinter.W, command=self._density_histogram_checkbutton_click)
        density_checkbox.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        # ===========================================================
        # HISTOGRAM FRAME
        # ===========================================================

        histogram_frame = tkinter.LabelFrame(main_frame, text='Histogram')
        histogram_frame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        self._histogram_bins.set(scana.DEFAULT_NUM_BINS)
        histogram_bins_label = tkinter.Label(histogram_frame, text='Number of Bins', anchor=tkinter.W)
        histogram_bins_label.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)
        histogram_bins_scale = tkinter.Scale(histogram_frame, variable=self._histogram_bins, orient=tkinter.HORIZONTAL,
                                             from_=10, to=500, resolution=1, state=tkinter.NORMAL)
        histogram_bins_scale.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        histogram_update_button = tkinter.Button(histogram_frame, text='Update',
                                                 command=self._histogram_update_button_click)
        histogram_update_button.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        # ===========================================================
        # FILTER FRAME
        # ===========================================================

        filter_frame = tkinter.LabelFrame(main_frame, text='Filters')
        filter_frame.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        filter_combobox = tkinter.ttk.Combobox(filter_frame, textvariable=self._filter_selected,
                                               values=('Butterworth', 'Bessel', 'Gaussian'))
        filter_combobox.bind('<<ComboboxSelected>>', self._dummy_callback)
        filter_combobox.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        # Filter Order
        self._filter_order.set(1)
        filter_order_label = tkinter.Label(filter_frame, text='Filter Order', anchor=tkinter.W)
        filter_order_label.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)
        filter_order_scale = tkinter.Scale(filter_frame, variable=self._filter_order, orient=tkinter.HORIZONTAL,
                                           from_=1, to=10, resolution=1, state=tkinter.NORMAL)
        filter_order_scale.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        # Low-Pass Filter
        self._filter_lpfreq.set(1.0)
        filter_lpfreq_label = tkinter.Label(filter_frame, text='Low-Pass Frequency [Hz]', anchor=tkinter.W)
        filter_lpfreq_label.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)
        filter_lpfreq_scale = tkinter.Scale(filter_frame, variable=self._filter_lpfreq, orient=tkinter.HORIZONTAL,
                                            from_=1.0, to=200.0, resolution=1.0, state=tkinter.NORMAL)
        filter_lpfreq_scale.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        filter_apply_button = tkinter.Button(filter_frame, text='Apply Low-Pass Filter', command=self._dummy_callback)
        filter_apply_button.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        # High-Pass Filter
        self._filter_hpfreq.set(100.0)
        filter_hpfreq_label = tkinter.Label(filter_frame, text='High-Pass Frequency [Hz]', anchor=tkinter.W)
        filter_hpfreq_label.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)
        filter_hpfreq_scale = tkinter.Scale(filter_frame, variable=self._filter_hpfreq, orient=tkinter.HORIZONTAL,
                                            from_=1.0, to=200.0, resolution=1.0, state=tkinter.NORMAL)
        filter_hpfreq_scale.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        filter_apply_button2 = tkinter.Button(filter_frame, text='Apply High-Pass Filter', command=self._dummy_callback)
        filter_apply_button2.pack(side=tkinter.TOP, fill=tkinter.X, padx=5, pady=5)

        # hide filter-frame (finish implementation)
        filter_frame.pack_forget()

        # ===========================================================
        # CANVAS
        # ===========================================================

        # add canvas and toolbar
        canvas = mplbackend.FigureCanvasTkAgg(self._figure, master=self._root)
        canvas.get_tk_widget().pack(side=tkinter.RIGHT, fill=tkinter.BOTH, expand=True)
        toolbar = mplbackend.NavigationToolbar2Tk(canvas, canvas.get_tk_widget())

        """
        sep = tkinter.Frame(toolbar, height='18p', relief=tkinter.RIDGE, bg='DarkGray')
        sep.pack(side=tkinter.LEFT, padx='3p')
        refresh_view_button = tkinter.Button(toolbar, text='Refresh', command=self._refresh_view_button_click)
        refresh_view_button.pack(side=tkinter.LEFT)
        """

        toolbar.pack(side=tkinter.BOTTOM)
        toolbar.update()
        canvas.mpl_connect('button_release_event', self._on_mouse_button_release)
        canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        canvas.mpl_connect('key_release_event', self._on_key_release)

        self._signals_listbox = signals_listbox
        self._remove_button = remove_button
        self._baseline_correction_button = baseline_correction_button
        self._remove_transition_phases_button = remove_transition_phases_button
        self._fit_histogram_button = fit_histogram_button
        self._remove_histogram_fit_button = remove_histogram_fit_button
        self._visible_checkbutton = visible_checkbox
        self._density_checkbutton = density_checkbox
        self._histogram_bins_scale = histogram_bins_scale
        self._histogram_update_button = histogram_update_button
        self._canvas = canvas

    def _span_selection_event(self, x_min, x_max):
        if self._signal_selected is None or not self._signal_selected.get_visible():
            return
        if self._selection is not None:
            self._selection.clear()
            self._selection = None
        time_start, time_end = self._ax_signal.get_xlim()
        time_range = self._signal_selected.signal.time_range()
        x_beg, x_end = max(time_range[0], x_min), min(time_range[1], x_max)
        if not (x_end-x_beg) > 0.01*(time_end-time_start):
            return
        # update selection
        signal_artist = self._signal_selected
        self._selection = Selection(signal_artist, x_beg, x_end, self._ax_signal)
        # update clipboard
        sub_samples = signal_artist.signal.sub_samples(x_beg, x_end)
        self._clipboard = (sub_samples, signal_artist.signal.frequency)
        # update GUI and plots
        self._update_gui()
        self._update_plot()

    def _init_figure(self, size_ratio=0.9):
        if not 0.0 < size_ratio < 1.0:
            raise RuntimeError('Error, axes size ratio not between 0.0 and 1.0!')
        width_ratios = (size_ratio, 1.0-size_ratio)
        grid_spec = mplgridspec.GridSpec(1, 2, wspace=0.01, hspace=0.1,
                                         top=0.975, bottom=0.1, left=0.05, right=0.975,
                                         width_ratios=width_ratios)

        signal_axis = self._figure.add_subplot(grid_spec[0, 0])
        signal_axis.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.5)
        signal_axis.minorticks_on()
        signal_axis.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
        signal_axis.set_xlabel('Time [ms]')
        signal_axis.set_ylabel('Current [pA]')
        signal_axis.callbacks.connect('xlim_changed', self._on_x_lims_change)

        histogram_axis = self._figure.add_subplot(grid_spec[0, 1], sharey=signal_axis)
        histogram_axis.tick_params(top=False, bottom=False, left=False, right=False,
                                   labelleft=False, labelbottom=False)

        self._span = mplwidgets.SpanSelector(signal_axis, self._span_selection_event, 'horizontal',
                                             useblit=True, button=1, props=dict(alpha=0.25, facecolor='#00AA00'))
        self._ax_signal = signal_axis
        self._ax_histogram = histogram_axis

    def _update_gui(self):
        # disable, not functional
        self._file_menu.entryconfigure('Open State', state=tkinter.DISABLED)
        if not len(self._signals) > 0:
            self._file_menu.entryconfigure('Save State', state=tkinter.DISABLED)
            self._file_menu.entryconfigure('Clear State', state=tkinter.DISABLED)
            self._view_menu.entryconfigure('Show all', state=tkinter.DISABLED)
            self._view_menu.entryconfigure('Histogram Log-Scale', state=tkinter.DISABLED)
            self._view_menu.entryconfigure('Plot all Gaussians', state=tkinter.DISABLED)
        else:
            self._file_menu.entryconfigure('Save State', state=tkinter.NORMAL)
            self._file_menu.entryconfigure('Clear State', state=tkinter.NORMAL)
            self._view_menu.entryconfigure('Show all', state=tkinter.NORMAL)
            self._view_menu.entryconfigure('Histogram Log-Scale', state=tkinter.NORMAL)
            self._view_menu.entryconfigure('Plot all Gaussians', state=tkinter.NORMAL)

        if self._selection is None:
            self._edit_menu.entryconfigure('Delete Selection', state=tkinter.DISABLED)
        else:
            self._edit_menu.entryconfigure('Delete Selection', state=tkinter.NORMAL)

        if self._clipboard is None:
            self._edit_menu.entryconfigure('Clear Clipboard', state=tkinter.DISABLED)
        else:
            self._edit_menu.entryconfigure('Clear Clipboard', state=tkinter.NORMAL)

        if self._signal_selected is None:
            self._file_menu.entryconfigure('Save Signal', state=tkinter.DISABLED)
            self._file_menu.entryconfigure('Save Histogram', state=tkinter.DISABLED)
            self._edit_menu.entryconfigure('Undo', state=tkinter.DISABLED)
            self._edit_menu.entryconfigure('Change Color', state=tkinter.DISABLED)
            self._edit_menu.entryconfigure('Change Alpha Value', state=tkinter.DISABLED)
            self._edit_menu.entryconfigure('Change Line Width', state=tkinter.DISABLED)
            self._view_menu.entryconfigure('Show selected only', state=tkinter.DISABLED)
            self._remove_button.config(state=tkinter.DISABLED)
            self._baseline_correction_button.config(state=tkinter.DISABLED)
            self._remove_transition_phases_button.config(state=tkinter.DISABLED)
            self._fit_histogram_button.config(state=tkinter.DISABLED)
            self._remove_histogram_fit_button.config(state=tkinter.DISABLED)
            self._visible_checkbutton.config(state=tkinter.DISABLED)
            self._density_checkbutton.config(state=tkinter.DISABLED)
            self._histogram_bins_scale.config(state=tkinter.DISABLED)
            self._histogram_update_button.config(state=tkinter.DISABLED)
        else:
            self._file_menu.entryconfigure('Save Signal', state=tkinter.NORMAL)
            self._file_menu.entryconfigure('Save Histogram', state=tkinter.NORMAL)
            if self._signal_selected.undo_stack_size() > 0:
                self._edit_menu.entryconfigure('Undo', state=tkinter.NORMAL)
            else:
                self._edit_menu.entryconfigure('Undo', state=tkinter.DISABLED)
            self._edit_menu.entryconfigure('Change Color', state=tkinter.NORMAL)
            self._edit_menu.entryconfigure('Change Alpha Value', state=tkinter.NORMAL)
            self._edit_menu.entryconfigure('Change Line Width', state=tkinter.NORMAL)
            self._view_menu.entryconfigure('Show selected only', state=tkinter.NORMAL)
            self._signal_visible.set(self._signal_selected.get_visible())
            self._density_hist.set(self._signal_selected.get_density())
            self._remove_button.config(state=tkinter.NORMAL)
            self._baseline_correction_button.config(state=tkinter.NORMAL)
            self._remove_transition_phases_button.config(state=tkinter.NORMAL)
            self._fit_histogram_button.config(state=tkinter.NORMAL)
            if self._signal_selected.has_histogram_fit:
                self._remove_histogram_fit_button.config(state=tkinter.NORMAL)
            else:
                self._remove_histogram_fit_button.config(state=tkinter.DISABLED)
            self._visible_checkbutton.config(state=tkinter.NORMAL)
            self._density_checkbutton.config(state=tkinter.NORMAL)
            self._histogram_bins.set(self._signal_selected.get_bins())
            self._histogram_bins_scale.config(state=tkinter.NORMAL)
            self._histogram_update_button.config(state=tkinter.NORMAL)

    def _clear_axes(self):
        self._ax_signal.clear()
        self._ax_histogram.clear()

    def _update_plot(self, *, relim_signal=False, relim_histogram=False, scalex_signal=True,
                     scaley_signal=True, scalex_histogram=True, scaley_histogram=True):
        # re-limit axis
        if relim_signal:
            self._ax_signal.relim(visible_only=True)
            self._ax_signal.autoscale_view(scalex=scalex_signal, scaley=scaley_signal)
        if relim_histogram:
            self._ax_histogram.relim(visible_only=True)
            self._ax_histogram.autoscale_view(scalex=scalex_histogram, scaley=scaley_histogram)
        # re-draw canvases
        self._canvas.draw()

    def main_loop(self):
        self._root.protocol('WM_DELETE_WINDOW', self._exit_button_click)
        self._root.mainloop()


if __name__ == '__main__':
    app = Application()
    app.main_loop()


import copy
import numpy
import matplotlib.axes
import matplotlib.lines
import matplotlib.patches
import matplotlib.collections

import scana

__all__ = ['SignalArtist', 'DEFAULT_NUM_BINS']

DEFAULT_NUM_BINS = 100


class SignalArtist:

    DEFAULT_COLOR = '#00AABB'
    DEFAULT_ALPHA = 1.0
    DEFAULT_ALPHA_SCALE_FILL = 0.5
    DEFAULT_LINE_WIDTH = 1.5

    def __init__(self, signal, ax_signal, ax_histogram, *, bins=DEFAULT_NUM_BINS, density=True):

        if not isinstance(signal, scana.Signal):
            raise TypeError('Error, wrong type, {} expected!'.format(scana.Signal))
        if not isinstance(ax_signal, matplotlib.axes.Axes):
            raise TypeError('Error, wrong type, {} expected!'.format(matplotlib.axes.Axes))
        if not isinstance(ax_histogram, matplotlib.axes.Axes):
            raise TypeError('Error, wrong type, {} expected!'.format(matplotlib.axes.Axes))

        # artist properties
        self._visible = True
        self._color = SignalArtist.DEFAULT_COLOR
        self._alpha = SignalArtist.DEFAULT_ALPHA
        self._linewidth = SignalArtist.DEFAULT_LINE_WIDTH

        self._signal = signal
        self._ax_signal = ax_signal
        self._ax_histogram = ax_histogram
        self._bins = int(bins)
        self._density = bool(density)

        # undo stack and zoom-signal
        self._undo_stack = list()
        self._zoom_signal = None
        # all the artists
        self._signal_artist = None
        self._histogram_artist = None
        self._histogram_fill_artist = None
        self._histogram_zoom_artist = None
        self._histogram_fit_artist = None

        self.update_artists()

    def __del__(self):
        if self._signal_artist is not None:
            self._signal_artist.remove()
            self._signal_artist = None
        if self._histogram_artist is not None:
            self._histogram_artist.remove()
            self._histogram_artist = None
        if self._histogram_fill_artist is not None:
            self._histogram_fill_artist.remove()
            self._histogram_fill_artist = None
        if self._histogram_zoom_artist is not None:
            self._histogram_zoom_artist.remove()
            self._histogram_zoom_artist = None
        if self._histogram_fit_artist is not None:
            self._histogram_fit_artist.remove()
            self._histogram_fit_artist = None
        del self._signal
        if self._zoom_signal is not None:
            del self._zoom_signal
        del self._undo_stack

    @property
    def has_histogram_fit(self):
        return self._histogram_fit_artist is not None

    def _create_signal_artist(self, signal, axes):
        # create artist
        time = signal.time()
        artist = matplotlib.lines.Line2D(time, signal.samples)

        # set artist properties
        artist.set_visible(self._visible)
        artist.set_color(self._color)
        artist.set_alpha(self._alpha)
        artist.set_linewidth(self._linewidth)
        artist.set_linestyle('solid')

        # add artist to axes
        axes.add_artist(artist)

        return artist

    def _create_histogram_artist(self, signal, bins, axes):
        # create artist
        edges, values = signal.histogram(bins, density=self._density)
        hist_data = numpy.column_stack((values, edges))
        fill_artist = matplotlib.patches.Polygon(hist_data, closed=True, edgecolor=None)
        line_artist = matplotlib.lines.Line2D(values, edges)

        # set artist properties
        fill_artist.set_visible(self._visible)
        fill_artist.set_color(self._color)
        fill_artist.set_alpha(self._alpha * SignalArtist.DEFAULT_ALPHA_SCALE_FILL)
        line_artist.set_visible(self._visible)
        line_artist.set_color(self._color)
        line_artist.set_alpha(self._alpha)
        line_artist.set_linewidth(self._linewidth)
        line_artist.set_linestyle('solid')

        # add artist to axes
        axes.add_artist(fill_artist)
        axes.add_artist(line_artist)

        return fill_artist, line_artist

    def _create_histogram_zoom_artist(self, signal, bins, axes):
        # create artist
        edges, values = signal.histogram(bins, density=self._density)
        line_artist = matplotlib.lines.Line2D(values, edges)

        # set artist properties
        line_artist.set_visible(self._visible)
        line_artist.set_color(self._color)
        line_artist.set_alpha(self._alpha)
        line_artist.set_linewidth(self._linewidth)
        line_artist.set_linestyle('solid')

        # add artist to axes
        axes.add_artist(line_artist)

        return line_artist

    def _update_histogram_style(self):
        if self._histogram_zoom_artist is None:
            self._histogram_artist.set_alpha(self._alpha)
            self._histogram_fill_artist.set_alpha(self._alpha * SignalArtist.DEFAULT_ALPHA_SCALE_FILL)
            self._histogram_artist.set_linestyle('solid')
        else:
            self._histogram_fill_artist.set_alpha(self._alpha * SignalArtist.DEFAULT_ALPHA_SCALE_FILL**3)
            self._histogram_artist.set_alpha(self._alpha * SignalArtist.DEFAULT_ALPHA_SCALE_FILL**2)
            self._histogram_artist.set_linestyle('dashed')

    def _update_histogram_zoom_artists(self):
        # remove artist from figure and call destructor
        if self._histogram_zoom_artist is not None:
            self._histogram_zoom_artist.remove()
            self._histogram_zoom_artist = None

        if self._zoom_signal is not None:
            self._histogram_zoom_artist = \
                self._create_histogram_zoom_artist(self._zoom_signal, self._bins, self._ax_histogram)

    def to_front(self):
        # signal artist
        if self._signal_artist is not None:
            self._signal_artist.remove()
            self._ax_signal.add_artist(self._signal_artist)
        # histogram artist
        if self._histogram_artist is not None:
            self._histogram_artist.remove()
            self._ax_histogram.add_artist(self._histogram_artist)
        # histogram zoom artist
        if self._histogram_zoom_artist is not None:
            self._histogram_zoom_artist.remove()
            self._ax_histogram.add_artist(self._histogram_zoom_artist)
        # histogram fit artist
        if self._histogram_fit_artist is not None:
            self._histogram_fit_artist.remove()
            self._ax_histogram.add_artist(self._histogram_fit_artist)

    def update_zoom(self, time_start, time_end):
        time_range = self._signal.time_range()
        if time_start > time_range[0] and time_end < time_range[1]:
            self._zoom_signal = self._signal.sub_signal(time_start, time_end)
        else:
            self._zoom_signal = None
        self._update_histogram_zoom_artists()
        self._update_histogram_style()

    def update_artists(self):
        # erase signal artist if it is not None
        if self._signal_artist is not None:
            self._signal_artist.remove()
            self._signal_artist = None
        # create signal artist
        self._signal_artist = self._create_signal_artist(self._signal, self._ax_signal)

        # erase histogram artists if they are not None
        if self._histogram_fill_artist is not None:
            self._histogram_fill_artist.remove()
            self._histogram_fill_artist = None
        if self._histogram_artist is not None:
            self._histogram_artist.remove()
            self._histogram_artist = None
        self.remove_histogram_fit()
        # create artists
        self._histogram_fill_artist, self._histogram_artist = \
            self._create_histogram_artist(self._signal, self._bins, self._ax_histogram)
        self._update_histogram_zoom_artists()
        self._update_histogram_style()

    def update_histogram_fit(self, *, order=4, plot_all=True):
        self.remove_histogram_fit()

        hist_fit = scana.HistogramFit(order=order)
        centers, values = self._signal.histogram(self._bins, density=self._density, step_function=False)
        params = hist_fit.fit(centers, values)

        if params is None:
            return None

        max_value = numpy.max(values)
        fit_values = hist_fit.exp_func_sum(centers, *params)

        tot_area = numpy.trapz(fit_values, centers)
        line_segments = [numpy.column_stack((fit_values, centers))]
        data_frame = list()

        for i in range(order):
            mu, sigma, rho = params[3*i:3*(i+1)]
            gauss_mu = mu
            gauss_sigma = 1.0 / numpy.sqrt(2.0 * sigma * sigma)
            gauss_rho = (numpy.sqrt(numpy.pi) / sigma) * rho
            line_segments.append(((-max_value * 0.1, mu), (max_value * 1.1, mu)))
            fit_values = hist_fit.exp_func(centers, mu, sigma, rho)
            area = numpy.trapz(fit_values, centers)
            data_frame.append([i, gauss_mu, gauss_sigma, gauss_rho, area])
            if plot_all:
                line_segments.append(numpy.column_stack((fit_values, centers)))

        self._histogram_fit_artist = matplotlib.collections.LineCollection(line_segments)

        # set artist properties
        self._histogram_fit_artist.set_visible(self._visible)
        self._histogram_fit_artist.set_color(self._color)
        self._histogram_fit_artist.set_alpha(self._alpha)
        self._histogram_fit_artist.set_linewidth(self._linewidth)
        self._histogram_fit_artist.set_linestyle('solid')

        self._ax_histogram.add_artist(self._histogram_fit_artist)

        return self._signal.label, data_frame, tot_area

    def remove_histogram_fit(self):
        if self._histogram_fit_artist is not None:
            self._histogram_fit_artist.remove()
            self._histogram_fit_artist = None

    def undo_stack_size(self):
        return len(self._undo_stack)

    def push_signal(self):
        self._undo_stack.append(copy.copy(self._signal))

    def pop_signal(self, *, apply=True):
        if not len(self._undo_stack) > 0:
            return False
        if apply:
            self._signal = self._undo_stack.pop()
        return True

    @property
    def signal(self):
        return self._signal

    @property
    def hist_bins(self):
        return self._bins

    @property
    def density_hist(self):
        return self._density

    def do_baseline_correction(self):
        self.push_signal()
        self._signal.baseline_correction(self._bins)
        self.update_artists()

    def get_bins(self):
        return self._bins

    def set_bins(self, bins):
        bins = int(bins)
        if bins == self._bins:
            return
        self._bins = bins

        # remove artists from figure and call destructor
        if self._histogram_artist is not None:
            self._histogram_artist.remove()
            self._histogram_artist = None
        if self._histogram_fill_artist is not None:
            self._histogram_fill_artist.remove()
            self._histogram_fill_artist = None
        self._histogram_fill_artist, self._histogram_artist = \
            self._create_histogram_artist(self._signal, self._bins, self._ax_histogram)
        self._update_histogram_zoom_artists()
        self._update_histogram_style()

    def get_density(self):
        return self._density

    def set_density(self, density):
        density = bool(density)
        if density == self._density:
            return
        self._density = density

        # remove artists from figure and call destructor
        if self._histogram_artist is not None:
            self._histogram_artist.remove()
            self._histogram_artist = None
        if self._histogram_fill_artist is not None:
            self._histogram_fill_artist.remove()
            self._histogram_fill_artist = None
        self._histogram_fill_artist, self._histogram_artist = \
            self._create_histogram_artist(self._signal, self._bins, self._ax_histogram)
        self._update_histogram_zoom_artists()
        self._update_histogram_style()

    def get_visible(self):
        return self._visible

    def set_visible(self, visible):
        self._visible = bool(visible)
        self._signal_artist.set_visible(self._visible)
        self._histogram_artist.set_visible(self._visible)
        self._histogram_fill_artist.set_visible(self._visible)
        if self._histogram_zoom_artist is not None:
            self._histogram_zoom_artist.set_visible(self._visible)
        if self._histogram_fit_artist is not None:
            self._histogram_fit_artist.set_visible(self._visible)

    def get_color(self):
        return self._color

    def set_color(self, color):
        self._color = color
        self._signal_artist.set_color(self._color)
        self._histogram_artist.set_color(self._color)
        self._histogram_fill_artist.set_color(self._color)
        if self._histogram_zoom_artist is not None:
            self._histogram_zoom_artist.set_color(self._color)
        if self._histogram_fit_artist is not None:
            self._histogram_fit_artist.set_color(self._color)

    def get_alpha(self):
        return self._alpha

    def set_alpha(self, alpha):
        self._alpha = min(1.0, max(float(alpha), 0.0))
        self._signal_artist.set_alpha(self._alpha)
        self._histogram_artist.set_alpha(self._alpha)
        self._histogram_fill_artist.set_alpha(self._alpha * SignalArtist.DEFAULT_ALPHA_SCALE_FILL)
        if self._histogram_zoom_artist is not None:
            self._histogram_zoom_artist.set_alpha(self._alpha * SignalArtist.DEFAULT_ALPHA_SCALE_FILL)
        if self._histogram_fit_artist is not None:
            self._histogram_fit_artist.set_alpha(self._alpha * SignalArtist.DEFAULT_ALPHA_SCALE_FILL)
        self._update_histogram_style()

    def get_linewidth(self):
        return self._linewidth

    def set_linewidth(self, linewidth):
        self._linewidth = max(float(linewidth), 0.1)
        self._signal_artist.set_linewidth(self._linewidth)
        self._histogram_artist.set_linewidth(self._linewidth)
        if self._histogram_zoom_artist is not None:
            self._histogram_zoom_artist.set_linewidth(self._linewidth)
        if self._histogram_fit_artist is not None:
            self._histogram_fit_artist.set_linewidth(self._linewidth)

    def to_json(self):
        jdata = {'visible': self._visible,
                 'color': self._color,
                 'alpha': self._alpha,
                 'linewidth': self._linewidth,
                 'signal': self._signal.to_json()}
        if len(self._undo_stack) > 0:
            undo_jdata = list()
            for signal in self._undo_stack:
                undo_jdata.append(signal.to_json())
            jdata['undo'] = undo_jdata
        return jdata





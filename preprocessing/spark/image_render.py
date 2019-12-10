import numpy as np
import matplotlib
matplotlib.use('Agg')
from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO

# Converting an image with the Keras API
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


class ImageRender():
    format_date = '%Y%m%d'
    volume_height = 0.2
    bar_width = 1
    rsi_height = 0.2

    '''
    plot_volume: Indicates if the volume bars should be displayed
    plot_sma: True if the SMA lines should be displayed
    plt_rsi: True if the RSI chart should be displayed
    figsize: Number of inches of a squared image
    use_adjusted: Use ADJ_CLOSE as reference column
    dpi: point per inch
    '''
    def __init__(self, plot_volume=True, plot_sma=True, plot_rsi=False, figsize=5, use_adjusted=True, dpi=72):
        self.plot_volume = plot_volume
        self.figsize = (figsize, figsize)
        self.adjusted = use_adjusted
        self.dpi = dpi
        self.plot_rsi = plot_rsi
        self.plot_sma = plot_sma
        # self.client = storage.Client()

    def _render_price(self, axis, candlesticks, df):
        axis.clear()
        candlestick_ohlc(axis, candlesticks, width=ImageRender.bar_width, colorup='g', colordown='r');
        if self.plot_sma:
            df['SMA50'].plot(ax=axis, color='cyan')
            df['SMA200'].plot(ax=axis, color='orange')
        # Shift price axis up to give volume chart space
        if self.plot_volume:
            ylim = axis.get_ylim()
            axis.set_ylim(ylim[0] - (ylim[1] - ylim[0]) * ImageRender.volume_height,
                          ylim[1] + (ylim[1] - ylim[0]) * ImageRender.rsi_height)
            axis.get_yaxis().set_visible(False)
        axis.tick_params(
            labelcolor='r',
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=False,
            labelbottom=False,
            labeltop=False,
            labelright=False,
            labelleft=False)  # labels along the bottom edge are off

    # Render daily volume part
    def _render_volume(self, axis, candlesticks):
        axis.clear()
        # get data from candlesticks for a bar plot
        dates = [x[0] for x in candlesticks]
        dates = np.asarray(dates)
        volume = [x[5] for x in candlesticks]
        volume = np.asarray(volume)
        # make bar plots and color differently depending on up/down for the day
        pos = [x[1] - x[4] <= 0 for x in candlesticks]
        neg = [x[1] - x[4] > 0 for x in candlesticks]
        axis.bar(dates[pos], volume[pos], color='blue', width=ImageRender.bar_width, align='center')
        axis.bar(dates[neg], volume[neg], color='yellow', width=ImageRender.bar_width, align='center')
        axis.set_ylim(0, max(volume) / ImageRender.volume_height)
        axis.get_yaxis().set_visible(False)
        axis.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=False,
            labelbottom=False,
            labeltop=False,
            labelright=False,
            labelleft=False)  # labels along the bottom edge are off

    def _render_rsi(self, axis, df):
        axis.clear()
        df['RSI14'].plot(ax=axis, color='magenta')
        ylim = axis.get_ylim()
        axis.set_ylim(ylim[1] - (ylim[1] - ylim[0]) / ImageRender.rsi_height, ylim[1])
        axis.axhline(y=min(df['RSI14']), color='black', linewidth=1)
        axis.get_yaxis().set_visible(False)
        axis.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            right=False,
            left=False,
            labelbottom=False,
            labeltop=False,
            labelright=False,
            labelleft=False)  # labels along the bottom edge are off

    def render_df(self, symbol, df):
        date1 = df.iloc[0]['DATE']
        date2 = df.iloc[-1]['DATE']
        cols = ['INDEX', 'ADJ_OPEN', 'ADJ_HIGH', 'ADJ_LOW', 'ADJ_CLOSE', 'ADJ_VOLUME'] if self.adjusted else [
            'INDEX', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        candlesticks = [tuple(x) for x in df[cols].values]
        fig, ax = plt.subplots(figsize=self.figsize)
        self._render_price(ax, candlesticks, df)
        if self.plot_volume:
            ax2 = ax.twinx()
            self._render_volume(ax2, candlesticks)
            self._render_rsi(ax.twinx(), df)
        path = symbol + '_' + date1.strftime(ImageRender.format_date) + '_' + date2.strftime(
            ImageRender.format_date) + '_' + str(len(candlesticks)) + '.png'
        # Salvar imagen
        plt.savefig(path, dpi=self.dpi)

    def render(self, df):
        # df = df.reset_index()
        # Extract date of first and last day
        cols = ['INDEX', 'ADJ_OPEN', 'ADJ_HIGH', 'ADJ_LOW', 'ADJ_CLOSE', 'ADJ_VOLUME'] if self.adjusted else [
            'INDEX', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']
        # Transform data into list of tuples
        candlesticks = [tuple(x) for x in df[cols].values]

        # Create figure using matplotlib OO
        fig = Figure(figsize=self.figsize)
        canvas = FigureCanvas(fig)
        ax = fig.subplots()
        self._render_price(ax, candlesticks, df)
        if self.plot_volume:
            ax2 = ax.twinx()
            self._render_volume(ax2, candlesticks)
        if self.plot_rsi:
            self._render_rsi(ax.twinx(), df)
        # Save it to a temporary buffer instead to a file on HDFS
        buf = BytesIO()

        # Save image into the buffer
        fig.savefig(buf, format="png", dpi=self.dpi)

        return self._convert_to_array(buf)

    # Converts an image into a list of numbers
    def _convert_to_array(self, buf):
        img = load_img(buf)
        img_array = img_to_array(img)
        img_array_flat = img_array.ravel().tolist()
        img_array_flat = [int(x) for x in img_array_flat]
        return img_array_flat
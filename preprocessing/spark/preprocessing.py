import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
import argparse
from pyspark.sql.functions import col, lag, lead, stddev, when, udf
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import pandas_udf, PandasUDFType
from datetime import datetime
import logging

import matplotlib
matplotlib.use('Agg')
import copy
import ta
from image_render import ImageRender


class Log4j(object):
    """Wrapper class for Log4j JVM object.
    :param spark: SparkSession object.
    """
    def __init__(self, spark):
        # get spark app details with which to prefix all messages
        conf = spark.sparkContext.getConf()
        app_id = conf.get('spark.app.id')
        app_name = conf.get('spark.app.name')

        log4j = spark._jvm.org.apache.log4j
        message_prefix = '<' + app_name + ' ' + app_id + '>'
        self.logger = log4j.LogManager.getLogger(message_prefix)

    def log(self, text):
        self.logger.info(text)
        logger = logging.getLogger("log")
        logger.info(text)


def images(spark, log, inputpath, outputpath, df_in, figsize, window_size, volume, window_offset, dpi, target, sma, rsi):
    path = outputpath + "/images"
    log.log("Generating Images and saving output to: " + path)

    # Apply filter to remove rows we don't want - for now
    sp500 = spark.read.csv(inputpath, header=True, inferSchema=True, sep=';')
    stocks_list = sp500.select(col('SYMBOL')).distinct().collect()
    stocks_list = [ x[0] for x in stocks_list ]
    df = df_in.where(col('SYMBOL').isin(stocks_list))
    df = df.where(col('DATE') >= '2000-01-01')
    df_grouped = df.groupby('SYMBOL').count()
    df_grouped = df_grouped[df_grouped['count'] > window_size.value].collect()
    stocks_list = [x[0] for x in df_grouped]
    df = df.where(col('SYMBOL').isin(stocks_list))
    log.log("Num of Rows: " + str(df.count()))

    # Get column names and broadcast, and get current schema
    dpi_value =int(dpi.value)
    size = int(figsize.value)
    schema = copy.deepcopy(df.schema)
    for i in range(0, 3 * ( size * dpi_value ) * ( size * dpi_value ) ):
        schema = schema.add(('C'+str(i)), IntegerType())
    column_names = spark.sparkContext.broadcast(df.columns)
    log.log("Column names value: " + str(column_names.value))

    # Get number of different symbols
    distinct_symbols = df.select(col('SYMBOL')).distinct().collect()
    distinct_symbols = {x[0]:i for i,x in enumerate(distinct_symbols)}
    num_symbols = len(distinct_symbols)
    log.log("Num Distinct Symbols: " + str(num_symbols))

    # Repartition to the number of symbols
    def symbol_partitioning(k):
        return distinct_symbols[k]

    udf_symbol_hash = udf(lambda str: symbol_partitioning(str))
    df = df.rdd \
        .map(lambda x: (x['SYMBOL'], x)) \
        .partitionBy(num_symbols, symbol_partitioning) \
        .toDF()

    #df = df.repartition(num_symbols, 'SYMBOL')
    log.log("Num Partitions after repartitioning: " + str(df.rdd.getNumPartitions()))
    log.log("Output Schema Length: " + str(len(list(df.schema))))


    log.log("Input Schema Length: " + str(len(list(schema))))

    @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
    def generate_images(data):

        # Reshape Pandas Dataframe
        #df = data['_2'].apply(pd.Series)
        #df.columns = column_names.value

        df = data.reset_index().rename(columns={'index': 'INDEX'})
        df['SMA50'] = df[target.value].rolling(50).mean()
        df['SMA200'] = df[target.value].rolling(200).mean()
        df['RSI14'] = ta.rsi(df[target.value], n=14)
        renderer = ImageRender(figsize=int(figsize.value), plot_volume=volume.value, dpi=int(dpi.value), plot_sma=sma.value, plot_rsi=rsi.value)

        window = window_size.value
        offset = window_offset.value
        l = [np.nan for i in range(0, window - 1)]
        for i in range(0, len(df) + 1 - window, offset):
            sample = df[i:i + window]
            l.append(renderer.render(sample))
            l = l + [ np.nan for j in range(0,offset-1)]
        l = l[:len(df)]

        df['IMAGE'] = l
        df = df.dropna()
        l = df['IMAGE'].values.tolist()
        #df_aux = pd.DataFrame(OrderedDict([('C' + str(i), [z[i] for z in l]) for i in range(0, len(l[0]))]))
        df_aux = pd.DataFrame( {'C' + str(i): [z[i] for z in l] for i in range(0, len(l[0]))} )
        df_aux = df_aux.reset_index(drop=True)
        df = df.drop(columns=['INDEX', 'IMAGE', 'SMA50', 'SMA200', 'RSI14'])
        df = df.reset_index(drop=True)
        #for c in df_aux.columns:
            #df[c] = df_aux[c]

        df = pd.concat([df.reset_index(drop=True), df_aux.reset_index(drop=True)], axis=1)
        return df

    log.log("Num Partitions: " + str(df.rdd.getNumPartitions()))

    # Rearrange columns
    df_schema = copy.deepcopy(df.schema)
    for x in df_schema['_2'].dataType:
        df = df.withColumn(x.name, df['_2'][x.name])
    df = df.drop('_2')
    df = df.drop('_1')

    log.log("Num Partitions: " + str(df.rdd.getNumPartitions()))

    # Dataframe
    df = df.groupBy('SYMBOL').apply(generate_images)
    log.log("Output Schema Length: " + str(len(list(df.schema))))

    # Save files in folder
    df.write.csv(path, header=True, mode='overwrite')

def main():

    parser = argparse.ArgumentParser()
    # Read Arguments
    parser.add_argument("--std", "-s", help="Number of STDEVs for labelling")
    parser.add_argument("--window", "-w", help="Window size")
    parser.add_argument("--ahead", "-a", help="Days ahead to check")
    parser.add_argument("--target", "-t", help="Target column to be used")
    parser.add_argument("--bucket", "-b", help="GS bucket name")
    parser.add_argument("--input", "-i", help="Path to input data within bucket")
    parser.add_argument("--images", "-k", help="Flag to indicate to generate images")

    parser.add_argument("--volume", "-v", help="Plot Volume?")
    parser.add_argument("--sma", "-m", help="Plot SMA?")
    parser.add_argument("--rsi", "-r", help="Plot RSI?")
    parser.add_argument("--figsize", "-fig", help="Figure Size")
    parser.add_argument("--offset", "-f", help="Offset")
    parser.add_argument("--dpi", "-d", help="DPI")
    parser.add_argument("--stocks", "-c", help="Stocks")

    # Init Context and Session
    spark = SparkSession.builder.appName('Preprocessing').getOrCreate()

    # read arguments from the command line
    args = parser.parse_args()

    # Set parameters for execution
    stdevs = spark.sparkContext.broadcast(float(args.std) if args.std else 1.0)
    days_ahead = spark.sparkContext.broadcast(int(args.ahead) if args.ahead else 10)
    target = spark.sparkContext.broadcast(args.target if args.target else 'ADJ_CLOSE')
    figsize = spark.sparkContext.broadcast(int(args.figsize) if args.figsize else 1)
    volume = spark.sparkContext.broadcast(args.volume if args.volume else True)
    sma = spark.sparkContext.broadcast(args.sma if args.sma else True)
    rsi = spark.sparkContext.broadcast(args.rsi if args.rsi else False)
    window_size = spark.sparkContext.broadcast(int(args.window) if args.window else 21)
    window_offset = spark.sparkContext.broadcast(int(args.offset) if args.offset else 5)
    dpi = spark.sparkContext.broadcast(int(args.dpi) if args.dpi else 72)
    stocks = args.stocks if args.stocks else 'sp500.csv'
    create_images = args.images if args.images else False

    # Initialize logger
    log = Log4j(spark)

    # Log used parameters
    log.log("Std. Devs: " + str(stdevs.value))
    log.log("Window size: " + str(window_size.value))
    log.log("Window offset: " + str(window_offset.value))
    log.log("Days ahead: " + str(days_ahead.value))
    log.log("Column to be used: " + str(target.value))
    log.log("Plot volume: " + str(volume.value))
    log.log("Figure Size: " + str(figsize.value))
    log.log("DPI: " + str(dpi.value))
    log.log("Create Images: " + str(create_images))


    # Read source dataframe from GS Bucket
    # This file contains all the prices we are going to use
    inputpath = 'gs://' + args.bucket + '/' + args.input
    df = spark.read.csv(inputpath, header=True, inferSchema=True)

    log.log("Num Partitions: " + str(df.rdd.getNumPartitions()))

    ## LABEL CALCULATION

    # Calculate pct daily change
    df = df.withColumn('PREV_DAY', lag(df[target.value]).over(Window.partitionBy("SYMBOL").orderBy("DATE")))
    df = df.withColumn('CHANGE', 100 * (df[target.value] - df['PREV_DAY']) / df[target.value])

    # Get N number of days ahead
    df = df.withColumn('AHEAD', lead(df[target.value], count=days_ahead.value).over(Window.partitionBy("SYMBOL").orderBy("DATE")))

    # Calculate Annual HVOl
    windowSpec = Window.partitionBy("SYMBOL").orderBy(col('DATE').cast('long')).rowsBetween(-window_size.value, 0)
    df = df.withColumn('HVOL', stddev(col(target.value) * np.sqrt(252)).over(windowSpec))

    # Calculate Upper and Lower limits
    df = df.withColumn('UPPER_LIMIT',
            col(target.value) + col(target.value) * ((col('HVOL') * np.sqrt(days_ahead.value) / (np.sqrt(252))) * stdevs.value / 100))
    df = df.withColumn('LOWER_LIMIT',
            col(target.value) - col(target.value) * ((col('HVOL') * np.sqrt(days_ahead.value) / (np.sqrt(252))) * stdevs.value / 100))

    # Finally, calculate the label
    df = df.withColumn('LABEL',
            (when(col('AHEAD') >  col('UPPER_LIMIT'), 1)
            .when(col('AHEAD') >= col('LOWER_LIMIT'), 0)
            .otherwise(-1)))

    # Build path to save file
    data_label = datetime.now().strftime("%Y%m%d_%H%M%S") + '_W' + str(window_size.value) + '_A' + str(days_ahead.value) + '_O' + str(window_offset.value) + '_S' + str(stdevs.value)
    outputpath = 'gs://' + args.bucket + '/data/' + data_label
    log.log("Writting file to GCS: " + outputpath)

    # Check if create images flag is active
    if create_images == True:
        # Generate image data for CNN
        log.log("Generating Images")
        images(spark, log, 'gs://' + args.bucket + '/data/' + stocks, outputpath, df, figsize, window_size, volume, window_offset, dpi, target, sma, rsi)
    else:
        # Save prices with label for LSTM
        log.log("Save prices with label")
        df.repartition(1).write.csv((outputpath + "/labels"), header=True, mode='overwrite')


if __name__ == '__main__':
    main()
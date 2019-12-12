import quandl
import pandas as pd
from tqdm import tqdm
import time
import argparse
from google.cloud import storage

# API Key a utilizar
# Limites: Authenticated users have a limit of 300 calls per 10 seconds,
# 2,000 calls per 10 minutes and a limit of 50,000 calls per day.
# Authenticated users of free data feeds have a concurrency limit of one;
# that is, they can make one call at a time and have an additional call in the queue.
quandl.ApiConfig.api_key = ""

# URL con la lista de stocks
url_stocks = 'https://old.nasdaq.com/screening/companies-by-name.aspx?letter=0&exchange=XXXXX&render=download'


# Clase para controlar el numero de peticiones a la API sin pasarnos
class RequestCounter():
    def __init__(self, max_calls, seconds):
        self.max = max_calls
        self.seconds = seconds
        self.counter = []

    def new_request(self):
        while (len(self.counter) >= self.max):
            time.sleep(1)
            self.refresh()
        self.counter.append(time.time())

    def refresh(self):
        self.counter = filter(lambda x: x + self.seconds > time.time(), self.counter)

def main():
    parser = argparse.ArgumentParser()
    # add long and short argument
    parser.add_argument("--max", "-m", help="Max calls to Quandl")
    parser.add_argument("--bucket", "-b", help="GS bucket name")
    # read arguments from the command line
    args = parser.parse_args()

    # Create bucket
    bucket = create_bucket(args.bucket)


    # Leer lista de stock y quedarme con las columnas relevantes
    nasdaq = pd.read_csv(url_stocks.replace('XXXXX', 'nasdaq'))
    nyse = pd.read_csv(url_stocks.replace('XXXXX', 'nyse'))
    stocks = pd.concat([nasdaq, nyse])
    stocks = stocks[['Symbol', 'Name', 'Sector', 'industry']]
    stocks = stocks.drop_duplicates(subset='Symbol')

    counter = RequestCounter(2000, 599)  # 2000 llamadas en 600 segundos

    calls = int(args.max) if args.max else len(stocks)
    #dataframes = []
    valid = 0
    # for i in tqdm(range(0,len(stocks))):
    for i in tqdm(range(0, calls)):
        counter.new_request()
        s = read_api(stocks.iloc[i]['Symbol'])
        if not s is None:
            #dataframes.append(s)
            s = s.reset_index()
            s.columns = [x.upper().replace(' ','').replace('-','').replace('.','_') for x in s.columns]
            if valid == 0:
                s.to_csv('file.csv', index=False)
            else:
                s.to_csv('file.csv', mode='a', header=False, index=False)
            valid = valid + 1

    #exportar = pd.concat(dataframes, ignore_index=True)
    print(" %i stocks read from API" % valid)

    #exportar = exportar.reset_index()
    exportar = pd.read_csv('file.csv')

    #upload to bucket
    bucket.blob('data/precios.csv').upload_from_string(exportar.to_csv(index=False, index_label=False), 'text/csv')


# Proceso de cada valor leido
def read_api(symbol):
    try:
        df = quandl.get("WIKI/"+symbol)
        df['Symbol'] = symbol
        return df
    except:
        pass

def create_bucket(bucket_name):
    """Creates a new bucket."""
    storage_client = storage.Client()
    bucket = storage_client.create_bucket(bucket_name)
    print('Bucket {} created'.format(bucket.name))
    return bucket


if __name__ == '__main__':
    main()
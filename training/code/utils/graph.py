import os,sys
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt

def save_time_series_plt(dates,y, name="timeseries.jpg", output_folder="training_results"):
    # start = datetime(2019, 8, 1)
    # dates = [ start + timedelta(0,i) for i in range(0,3000,15) ]
    # y = [np.random.randint(0,100) for i in range(0,3000,15) ]
    # plt.gcf().autofmt_xdate
    if not y:
        return
    date_form = DateFormatter("%Y-%m-%d-%H:%M:%S")
    plt.gca().xaxis.set_major_formatter(date_form)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    if type(y[0]) == int or type(y[0])== float :
        plt.plot(dates, y)
    else:
        for i,ydash in enumerate(y):
            plt.plot(dates, ydash, label=f'gpu{i}')
        plt.legend()
    plt.savefig(os.path.curdir + f"/{output_folder}/{name}")
    plt.close()

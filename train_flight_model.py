import sys
import pandas

if __name__ == '__main__':
    navdata = pandas.read_csv(sys.argv[1], sep='\t', header=None).rename(columns={0: "timestamp"})
    commands = pandas.read_csv(sys.argv[2], sep='\t', header=None).rename(columns={0: "timestamp"})


    begining_timestamp = commands.iloc[0, 0] - (commands.iloc[0, 0] % 20) + 20
    nav_len = navdata.shape[0]-1
    end_timestamp = navdata.iloc[nav_len, 0] - (navdata.iloc[nav_len, 0] % 20)

    new_navdata = pandas.DataFrame(columns=navdata.columns)
    counter=0
    for cur_timestamp in range(int(begining_timestamp), int(end_timestamp), 20):
        cur_datawindow = navdata.loc[(cur_timestamp < navdata['timestamp']) & (navdata['timestamp']< cur_timestamp + 20), :]
        if cur_datawindow.shape[0] < 1: continue
        # print(navdata.loc[(cur_timestamp < navdata['timestamp']) & (navdata['timestamp']< cur_timestamp + 20), :].shape)
        counter += 1
        new_row = {}
        for column in navdata:
            new_row[column] = cur_datawindow[column].mean(axis=0)
        new_navdata.loc[cur_timestamp] = pandas.Series(new_row)
    print(counter)
    print(new_navdata.shape)

import sys
import pandas

WINDOW_LENGTH = 20

if __name__ == '__main__':
    navdata = pandas.read_csv(sys.argv[1], sep='\t', header=None).rename(columns={0: "timestamp"})
    commands = pandas.read_csv(sys.argv[2], sep='\t', header=None).rename(columns={0: "timestamp"})

    begining_timestamp = commands.iloc[0, 0] - (commands.iloc[0, 0] % WINDOW_LENGTH) + WINDOW_LENGTH
    nav_len = navdata.shape[0]-1
    end_timestamp = navdata.iloc[nav_len, 0] - (navdata.iloc[nav_len, 0] % WINDOW_LENGTH)

    new_navdata = pandas.DataFrame(columns=navdata.columns)
    new_commands = pandas.DataFrame(columns=commands.columns)
    counter = 0
    prev_command = [0.0, 0.0, 0.0, 0.0]
    for cur_timestamp in range(int(begining_timestamp), int(end_timestamp), WINDOW_LENGTH):
        cur_datawindow = navdata.loc[(cur_timestamp < navdata['timestamp']) & (navdata['timestamp'] < cur_timestamp + WINDOW_LENGTH), :]
        if cur_datawindow.shape[0] < 1: continue
        counter += 1
        new_row = {}
        for column in navdata:
            new_row[column] = cur_datawindow[column].mean(axis=0)
        new_navdata.loc[cur_timestamp] = pandas.Series(new_row)

        cur_command = commands.loc[(cur_timestamp < commands['timestamp']) & (commands['timestamp'] < cur_timestamp + WINDOW_LENGTH), :]
        if cur_command.shape[0] < 1:
            cur_command = prev_command
        else:
            cur_command = cur_command.values.tolist()
            prev_command = cur_command
        new_commands.loc[cur_timestamp] = pandas.Series(cur_command)
    print(counter)
    print(new_commands.shape)

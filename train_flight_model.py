import sys
import pandas

WINDOW_LENGTH = 20
NAVDATA_OFFSET_MILIS = 20

if __name__ == '__main__':
    navdata = pandas.read_csv(sys.argv[1], sep='\t', header=None).rename(columns={0: "timestamp"})
    commands = pandas.read_csv(sys.argv[2], sep='\t', header=None).rename(columns={0: "timestamp"})

    # the data of our interest start with the beginning of the command data and end with the last sensor reading
    begining_timestamp = commands.iloc[0, 0] - (commands.iloc[0, 0] % WINDOW_LENGTH) + WINDOW_LENGTH
    nav_len = navdata.shape[0]-1
    end_timestamp = navdata.iloc[nav_len, 0] - (navdata.iloc[nav_len, 0] % WINDOW_LENGTH)

    # we'll now create a average over a WINDOW_LENGTH ms time window, and we'll create a table of commands that covers
    # all the time. We assume that the commands have a sticky behavior (the command changes only when another arrives)
    new_navdata = pandas.DataFrame(columns=navdata.columns)
    new_commands = pandas.DataFrame(columns=commands.columns)
    # just a debug counter
    counter = 0
    prev_command = pandas.DataFrame({'timestamp': begining_timestamp, 1: .0, 2: .0, 3: .0, 4: .0, }, index=[begining_timestamp])
    for cur_timestamp in range(int(begining_timestamp), int(end_timestamp), WINDOW_LENGTH):
        data_from_current_window = navdata.loc[(cur_timestamp < navdata['timestamp']) & (navdata['timestamp'] < cur_timestamp + WINDOW_LENGTH), :]
        if data_from_current_window.shape[0] < 1: continue
        counter += 1
        # count the average over the timewindow for each column and add it to the new data
        new_row = {}
        for column in navdata:
            new_row[column] = data_from_current_window[column].mean(axis=0)
        new_row_timestamp = cur_timestamp + NAVDATA_OFFSET_MILIS
        new_row['timestamp'] = new_row_timestamp
        new_navdata.loc[new_row_timestamp] = pandas.Series(new_row)

        # get the command fiting to that timewindow
        cur_commands = commands.loc[(cur_timestamp < commands['timestamp']) & (commands['timestamp'] < cur_timestamp + WINDOW_LENGTH), :]
        # if there's no command, use the previous one
        if cur_commands.shape[0] < 1:
            cur_command = prev_command
        # if there's more than 1 command in the current window, use the first one for the current window and the last
        # one for the next
        else:
            cur_command = cur_commands.head(1)
            prev_command = cur_commands.tail(1)

        cur_command['timestamp'] = cur_timestamp
        new_commands.loc[cur_timestamp] = pandas.Series(cur_command.iloc[0, :])

    # do a database join to the navdata and commands on the timestamp column
    joined_data = pandas.merge(new_navdata, new_commands, on='timestamp', how='inner')

    # debug prints
    print(counter)
    print(new_commands.tail(1))
    print(new_navdata.tail(1))
    print(joined_data.tail(1))
    print(joined_data.shape)

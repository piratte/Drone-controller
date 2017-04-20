import sys
from pprint import pprint

import pandas
import numpy as np

WINDOW_LENGTH = 20
NAVDATA_OFFSET_MILIS = 20

if __name__ == '__main__':
    if len(sys.argv) != 3 or '-h' in sys.argv or '--help' in sys.argv:
        print("Usage: %s NAVDATA_PATH COMMANDS_PATH" % sys.argv[0])
        sys.exit(1)
    navdata = pandas.read_csv(sys.argv[1], sep='\t', header=None).rename(columns={0: "timestamp"})
    commands = pandas.read_csv(sys.argv[2], sep='\t', header=None).rename(columns={0: "timestamp"})

    # the data of our interest start with the beginning of the command data and end with the last sensor reading
    begining_timestamp = commands.iloc[0, 0] - (commands.iloc[0, 0] % WINDOW_LENGTH) + WINDOW_LENGTH
    nav_len = navdata.shape[0]-1
    end_timestamp = navdata.iloc[nav_len, 0] - (navdata.iloc[nav_len, 0] % WINDOW_LENGTH)

    # we'll now create a average over a WINDOW_LENGTH ms time window, and we'll create a table of commands that covers
    # all the time. We assume that the commands have a sticky behavior (the command changes only when another arrives)
    joined_data = []
    new_navdata = []
    # just a debug counter
    counter = 0
    prev_command = pandas.DataFrame({'timestamp': begining_timestamp, 1: .0, 2: .0, 3: .0, 4: .0, }, index=[begining_timestamp])
    for cur_timestamp in range(int(begining_timestamp), int(end_timestamp), WINDOW_LENGTH):
        data_from_current_window = navdata.loc[(cur_timestamp < navdata['timestamp']) & (navdata['timestamp'] < cur_timestamp + WINDOW_LENGTH), :]
        if data_from_current_window.shape[0] < 1: continue
        counter += 1
        # count the average over the timewindow for each column and add it to the new data
        new_row_timestamp = cur_timestamp + NAVDATA_OFFSET_MILIS
        new_row = [cur_timestamp]
        for column in navdata:
            if column == 'timestamp': continue
            new_row.append(data_from_current_window[column].mean(axis=0))
        new_navdata.append(tuple(new_row))
        # if counter > 10: sys.exit()

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
        # get the only row and remove timestamp
        new_row.extend(cur_command.values.tolist()[0][1:])
        joined_data.append(tuple(new_row))

    new_navdata = np.array(new_navdata)
    joined_data = np.array(joined_data)

    # debug prints
    print(counter, new_navdata.shape, joined_data.shape)
    np.save("joined_data", joined_data)
    np.save("new_navdata", new_navdata)


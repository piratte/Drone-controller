import sys
from pprint import pprint

import pandas
import numpy as np

WINDOW_LENGTH = 10
NAVDATA_OFFSET_MILIS = 10

NAVDATA_COLUMNS = ['timestamp', 'state', 'battery level', 'magnetometer x', 'magnetometer y', 'magnetometer z',
                   'pressure', 'temp', 'wind speed', 'wind angle', 'wind compensation: pitch',
                   'wind compensation: roll', 'rotation around y', 'rotation around x', 'rotation around z',
                   'altitude', 'velocity x', 'velocity y', 'velocity z', 'acceleration x', 'acceleration y',
                   'acceleration z', 'motor 1 power', 'motor 2 power', 'motor 3 power', 'motor 4 power',
                   'board timestamp']

WANTED_NAV_COLUMNS = ['timestamp', 'state', 'magnetometer x', 'magnetometer y', 'magnetometer z',
                      'rotation around y', 'rotation around x', 'rotation around z',
                      'altitude', 'velocity x', 'velocity y', 'velocity z',
                      'acceleration x', 'acceleration y', 'acceleration z',
                      'motor 1 power', 'motor 2 power', 'motor 3 power', 'motor 4 power']

COMMAND_COLUMNS = ['timestamp', 'left-right tilt', 'front-back tilt', 'vertical speed', 'angular speed']

if __name__ == '__main__':
    if len(sys.argv) != 3 or '-h' in sys.argv or '--help' in sys.argv:
        print("Usage: %s NAVDATA_PATH COMMANDS_PATH" % sys.argv[0])
        sys.exit(1)
    navdata = pandas.read_csv(sys.argv[1], sep='\t', names=NAVDATA_COLUMNS, header=None, usecols=WANTED_NAV_COLUMNS)
    commands = pandas.read_csv(sys.argv[2], sep='\t', names=COMMAND_COLUMNS, header=None)

    print("CSV loaded")
    # for WINDOW_LENGTH, NAVDATA_OFFSET_MILIS in [(10,10), (15,10), (15,15), (20,10), (20,15)]:
    # the data of our interest start with the beginning of the command data and end with the last sensor reading
    min_com_timestamp = commands['timestamp'].min()
    begining_timestamp = min_com_timestamp - (min_com_timestamp % WINDOW_LENGTH) + WINDOW_LENGTH
    nav_len = navdata.shape[0]-1
    max_nav_timestamp = navdata['timestamp'].max()
    end_timestamp = max_nav_timestamp - (max_nav_timestamp % WINDOW_LENGTH)

    print(begining_timestamp, end_timestamp)
    # we'll now create a average over a WINDOW_LENGTH ms time window, and we'll create a table of commands that covers
    # all the time. We assume that the commands have a sticky behavior (the command changes only when another arrives)
    joined_data = []
    new_navdata = []
    # just a debug counter
    counter = 0
    prev_command = pandas.DataFrame({'timestamp': begining_timestamp,
                                     'left-right tilt': .0,
                                     'front-back tilt': .0,
                                     'vertical speed': .0,
                                     'angular speed': .0, }, index=[begining_timestamp])

    for cur_timestamp in range(int(begining_timestamp), int(end_timestamp), WINDOW_LENGTH):
        data_from_current_window = navdata.loc[(cur_timestamp + NAVDATA_OFFSET_MILIS < navdata['timestamp']) &
                                               (navdata['timestamp'] < cur_timestamp + NAVDATA_OFFSET_MILIS + WINDOW_LENGTH), :]
        if data_from_current_window.shape[0] < 1: continue
        counter += 1
        # count the average over the timewindow for each column and add it to the new data
        new_row_timestamp = cur_timestamp + NAVDATA_OFFSET_MILIS
        new_row = [cur_timestamp]
        for column in navdata:
            if column == 'timestamp': continue
            new_row.append(data_from_current_window[column].mean(axis=0))
        new_navdata.append(tuple(new_row))

        # get the command fiting to that timewindow
        cur_commands = commands.loc[(cur_timestamp < commands['timestamp']) &
                                    (commands['timestamp'] < cur_timestamp + WINDOW_LENGTH), :]
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
    np.save("joined_data_long_reduced_%d_%d" % (WINDOW_LENGTH, NAVDATA_OFFSET_MILIS), joined_data)
    np.save("new_navdata_long_reduced_%d_%d" % (WINDOW_LENGTH, NAVDATA_OFFSET_MILIS), new_navdata)


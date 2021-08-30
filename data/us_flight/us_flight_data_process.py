import bz2
import csv
from functools import partial

import numpy as np


class BZ2_CSV_LineReader(object):
    def __init__(self, filename, buffer_size=4*1024):
        self.filename = filename
        self.buffer_size = buffer_size

    def readlines(self):
        with open(self.filename, 'rb') as file:
            for row in csv.reader(self._line_reader(file)):
                yield row

    def _line_reader(self, file):
        buffer = ''
        decompressor = bz2.BZ2Decompressor()
        reader = partial(file.read, self.buffer_size)

        for bindata in iter(reader, b''):
            block = decompressor.decompress(bindata).decode('utf-8')
            buffer += block
            if '\n' in buffer:
                lines = buffer.splitlines(True)
                if lines:
                    buffer = '' if lines[-1].endswith('\n') else lines.pop()
                    for line in lines:
                        yield line

# 1	Year	1987-2008
# 2	Month	1-12
# 3	DayofMonth	1-31
# 4	DayOfWeek	1 (Monday) - 7 (Sunday)
# 5	DepTime	actual departure time (local, hhmm)
# 6	CRSDepTime	scheduled departure time (local, hhmm)
# 7	ArrTime	actual arrival time (local, hhmm)
# 8	CRSArrTime	scheduled arrival time (local, hhmm)
# 9	UniqueCarrier	unique carrier code
# 10	FlightNum	flight number
# 11	TailNum	plane tail number
# 12	ActualElapsedTime	in minutes
# 13	CRSElapsedTime	in minutes
# 14	AirTime	in minutes
# 15	ArrDelay	arrival delay, in minutes
# 16	DepDelay	departure delay, in minutes
# 17	Origin	origin IATA airport code
# 18	Dest	destination IATA airport code
# 19	Distance	in miles
# 20	TaxiIn	taxi in time, in minutes
# 21	TaxiOut	taxi out time in minutes
# 22	Cancelled	was the flight cancelled?
# 23	CancellationCode	reason for cancellation (A = carrier, B = weather, C = NAS, D = security)
# 24	Diverted	1 = yes, 0 = no
# 25	CarrierDelay	in minutes
# 26	WeatherDelay	in minutes
# 27	NASDelay	in minutes
# 28	SecurityDelay	in minutes
# 29	LateAircraftDelay	in minutes

x_data = np.empty((1,6))
y_data = np.empty((1,1))

bz2_csv_filename = '2008.csv.bz2'
for row in BZ2_CSV_LineReader(bz2_csv_filename).readlines():
    if row[0] == '2008':
        # if int(row[1]) < 4: until April..
        if int(row[1]) <= 12:
            if row[-1] != 'NA':
                x_array = np.array([float(row[1]),
                                    float(row[2]),
                                    float(row[3]),
                                    float(row[4]),
                                    float(row[6]),
                                    float(row[18])])

                y_array = np.array([float(row[-1])])

                x_data = np.vstack((x_data, x_array))
                y_data = np.vstack((y_data, y_array))

                print(x_array, y_array)


x_data = x_data[1:,:]
y_data = y_data[1:,:]

np.savez('./us_flight_data_year08.npz',
         x_data=x_data,
         y_data=y_data)

# np.savez('./us_flight_data.npz',
#          x_data=x_data,
#          y_data=y_data)
    # print(row[-1]=='NA')
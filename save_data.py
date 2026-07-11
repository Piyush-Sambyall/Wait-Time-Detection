import serial
import csv

ser = serial.Serial('COM3',115200)   # change COM port

with open('sensor_data.csv','w',newline='') as f:
    writer = csv.writer(f)

    while True:
        line = ser.readline().decode().strip()
        print(line)

        data = line.split(',')

        if len(data)==7:
            writer.writerow(data)
            f.flush()
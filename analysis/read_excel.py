from xlrd import open_workbook
# import datetime
# import delorean
#
# dt = datetime.datetime.utcnow()
# delorean.Delorean(dt, timezone="UTC").epoch

# timestampFilename = '/home/dinesh/PycharmProjects/MasterThesis/signals/Suhas/shoulder/fast/UltrasenseII.xls'
# filename = '/home/dinesh/PycharmProjects/MasterThesis/signals/Suhas/shoulder/fast/1_38_50_31_rec.wav'
#
# startTime = filename.split('/').__getitem__(filename.split('/').__len__()-1).split('.').__getitem__(0).split('_')
# startHour = startTime.__getitem__(0)
# startMin = startTime.__getitem__(1)
# startSec = startTime.__getitem__(2)
# startMilliSec = startTime.__getitem__(3)
# wb = open_workbook(timestampFilename)
# sheet = wb.sheet_by_name("timestampsheet")
#
# c = sheet.cell(1,3).value
# xf = str(c)
# print  xf
# print sheet.nrows
# print sheet.ncols
# for sheet in wb.sheets():
#     number_of_rows = sheet.nrows
#     number_of_columns = sheet.ncols
#     print number_of_rows
#     print number_of_columns



## Read from file :

timestampsPath = '/home/dinesh/PycharmProjects/MasterThesis/signals/Suhas/shoulder/fast/timestamps.txt'

activity_start = []
activity_end = []
data = open(timestampsPath).read()
rows =  data.split('\n')
for r in rows:
    val = r.split('\t')
    if len(val[0])!=0:
        activity_start.append(val[0])
        activity_end.append(val[1])
print activity_start
print activity_end

timestamp = 13.19
xlocs = 283
sample_len = 581632
timebins = 283
binsize = 1024
samplerate = 44100

n1 = (timebins * binsize)/2
n2 = samplerate * timestamp * timebins
xlocation = (((timebins * binsize)/2) + (samplerate * timestamp * timebins))/sample_len
print xlocation

# print ((xlocs * sample_len / timebins) + (0.5 * binsize)) / samplerate
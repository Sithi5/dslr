import math
import numpy as np


def count(data):
    count = 0
    for i in data:
        if not np.isnan(i):
            count += 1
    return float(count)

def mean(data):
    sum = 0
    c = count(data)
    for i in data:
        if not np.isnan(i):
            sum += i
    mean = sum/c
    return float(mean)

def std(data):
    m = mean(data)
    c = count(data)
    sum_std = 0
    for i in data:
        if not np.isnan(i):
            sum_std += (i - m)**2
    var = sum_std / c
    std = var**(0.5)
    return float(std)

def min(data):
    min = data[0]
    for i in data:
        if not np.isnan(i) and i < min:
            min = i
    return min

def percentile(data, percentile):
    data = np.sort(data)
    k = (len(data)-1) * percentile
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    d0 = data[int(f)] * (c-k)
    d1 = data[int(c)] * (k-f)
    return d0+d1


def quarter(data):
    tmp_data = []
    for i in data:
        if not np.isnan(i):
            tmp_data.append(i)
    res = percentile(tmp_data, 0.25)
    return res

def median(data):
    tmp_data = []
    for i in data:
        if not np.isnan(i):
            tmp_data.append(i)
    res = percentile(tmp_data, 0.50)
    return res

def three_quarters(data):
    tmp_data = []
    for i in data:
        if not np.isnan(i):
            tmp_data.append(i)
    res = percentile(tmp_data, 0.75)
    return res

def max(data):
    max = 0
    for i in data:
        if not np.isnan(i) and i > max:
            max = i
    return max
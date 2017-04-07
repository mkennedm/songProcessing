'''=============================================================================
FILE: song_processing.py
AUTH: Matthew Kennedy, Kyle Liberti
MAIL: mkennedm@bu.edu, kliberti@bu.edu
DESC: Classifies zebra finch songs by bird
VERS: Python3
REQ : Following packages must be installed on your system
      - scipy
      - scikit

IMPORTANT
---------
The following data files must be present in the same directory as the python
file in order for it to run correctly:
      - Song_Aggregate_Data.txt
      - Song_Aggregate_Data2.txt

WARN: The function 'play' will only work properly on linux machines
============================================================================='''
from scipy.io.wavfile import write
from scipy import fft, arange
import numpy as np
import os
import csv
import math
import cmath
import random as r
import matplotlib.pyplot as plt
import time
import random
'''==============================================
             FIELDS
=============================================='''
CSV1 = "./Song_Aggregate_Data.txt"
CSV2 = "./Song_Aggregate_Data2.txt"
SAMPLE_FREQUENCY = 24400
TIME = 41869
TRAINING_SIZE = 120
PERCENT_TRAINING = 0.8
PERCENT_TESTING = 1 - PERCENT_TRAINING
OUT = "song.wav"
COLOR = "b"
'''==============================================
             FORMATING FUNCTIONS
=============================================='''
def import_csv(f):
    ''' Converts CSV file, f, into a 2D numpy array '''
    reader = csv.reader(open(f, "r"), delimiter=',')
    x = list(reader)
    return np.array(x).astype('float')

def play(stream):
    '''
    Plays audio created from the raw data inputted, stream
    NOTE: Thie will only work on linux machines
    '''
    scaled = np.float32(stream/np.max(np.abs(stream)))
    write(OUT, SAMPLE_FREQUENCY, scaled)
    os.system("play " + OUT)

def pad_with_zeros(d1, d2):
    '''
    Pads every row of 2D list, d2, with zeros to match
    dimensions of d1
    '''
    result = np.zeros(d1.shape)
    for r in range(len(d2)):
        for c in range(len(d2[0])):
            result[r][c] = d2[r][c]
    return result
'''==============================================
             COMPUTAIONAL FUNCTIONS
=============================================='''
def cos_sim(s1, s2):
    ''' Computes the cosine similarity btwn two numpy arrays '''
    p = sum([s1[i]*s2[i] for i in range(len(s1))])
    A = cmath.sqrt(sum([a**2 for a in s1]))
    B = cmath.sqrt(sum([b**2 for b in s2]))
    sim = p/(A*B)
    return sim

def pearson(s1, s2):
    ''' Computes the Pearson correlation btwn two numpy arrays '''
    return np.cov(s1, s2)[0][1]/(np.std(s1)*np.std(s2))

def fourier(s):
    ''' Preforms Fast Fourier transform on inputted audio stream, s '''
    l = len(s)
    y = 2*fft(s)/l
    y = y[:(l//2)]
    return abs(y)

def avg(s):
    ''' Returns a list of the averages of the columns of 2D list, s '''
    columns = np.sum(s, axis=0)
    avg = [c/len(s) for c in columns]
    return avg

def running_mean(x, N):
    '''
    Smooths the inputted list, x, using the running average of N numbers.
    Credit: http://stackoverflow.com/questions/13728392/moving-average-or-running-mean/27681394#27681394
    '''
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

def training(d1):
    ''' Returns a randomized list of songs and generated noise '''
    songs1 = r.sample(d1, int(PERCENT_TRAINING*len(d1)))
    rands = [r.sample(list(s), len(s)) for s in songs1]
    training_set = songs1 + rands
    r.shuffle(training_set)
    return training_set

def calc_bins(x):
    '''
    Determines bin number for list using Scott's rule
    Credit: http://stackoverflow.com/questions/6876358/how-to-keep-a-dynamical-histogram
    '''
    bin_width = 3.5*np.std(x)*len(x)**(-1/3.0)
    bin_size = math.ceil((max(x) - min(x))/bin_width)
    return bin_size

'''==============================================
             PLOTTING FUNCTIONS
=============================================='''
def plot_line(y):
    x = [(x/24400.0)*1000.0 for x in range(len(y))]
    plt.xlabel('Time(ms)')
    plt.ylabel('Voltage(mV)')
    plt.plot(x, y, color=COLOR)
    plt.show()

def plot_hist(x):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hist(x, bins = calc_bins(x), color = COLOR)
    ax.set_xlim(-1, 1)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.show()

def plot_fft(y):
    '''
    Plots frequency spectrum
    Credit: http://glowingpython.blogspot.com/2011/08/how-to-plot-frequency-spectrum-with.html
    '''
    n = len(y)
    frq = frq[0:n//2] # one side frequency range
    Y = 2*fft(y)/n # fft computing and normalization
    Y = Y[0:n//2]
    frq = arange(len(Y))

    plt.plot(frq, abs(Y), COLOR) # plotting the spectrum
    plt.xlabel('Freq (Hz)')
    plt.ylabel('|Y(freq)|')
    plt.show()

def plot_clusters(avg, train):
    x = [float(cos_sim(avg, i)) for i in train]
    y = [float(pearson(avg, i)) for i in train]
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Pearson Correlation')
    plt.scatter(x, y, color = COLOR)
    plt.show()

'''==============================================
             RUN
=============================================='''
def plot_raw_line():
    DATA = import_csv(CSV1)
    plot_line(DATA[0])

def plot_raw_rand_line():
    DATA = import_csv(CSV1)
    s = DATA[0]
    s = r.sample(s, len(s))
    plot_line(s)

def plot_raw_mult_line():
    DATA = import_csv(CSV1)

    plt.figure(1)
    y1 = DATA[0]
    y2 = DATA[5]
    y3 = DATA[75]

    x = [(x/24400.0)*1000.0 for x in range(len(y1))]
    plt.xlabel('Time(ms)')
    plt.ylabel('Voltage(mV)')

    plt.subplot(311)
    plt.plot(x, y1, color=COLOR)
    plt.subplot(312)
    plt.plot(x, y2, color=COLOR)
    plt.subplot(313)
    plt.plot(x, y3, color=COLOR)
    plt.show()

def plot_raw_hist():
    DATA = import_csv(CSV1)
    x = training(DATA)
    a = avg(DATA)
    x = [float(cos_sim(a, d)) for d in x]
    plot_hist(x)

def plot_transformation():
    DATA = import_csv(CSV1)
    fig = plt.figure()

    y = DATA[0]
    x = [(x/24400.0)*1000.0 for x in range(len(y))]

    ax1 = fig.add_subplot(311)
    ax1.set_xlabel('Time(ms)')
    ax1.set_ylabel('Voltage(mV)')
    ax1.plot(x, y, color='r')

    n = len(y)
    Y = 2*fft(y)/n # fft computing and normalization
    Y = Y[0:n//2]
    frq = arange(len(Y))

    ax2 = fig.add_subplot(312)
    ax2.set_xlabel('Freq(Hz)')
    ax2.set_ylabel('Power(dB)')
    ax2.plot(frq, abs(Y), color = "y") # plotting the spectrum

    y = abs(Y)
    x = frq

    N = calc_bins(y)
    y = running_mean(y, N)
    ax3 = fig.add_subplot(313)
    ax3.set_xlabel('Freq(Hz)')
    ax3.set_ylabel('Power(dB)')
    ax3.plot(x, y, color = "g")
    plt.show()

def plot_proc_hist():
    DATA = import_csv(CSV1)
    N = calc_bins(DATA[0]) #100
    DATA = [running_mean(fourier(d), N) for d in DATA]
    x = training(DATA)
    a = avg(DATA)
    x = [float(cos_sim(a, d)) for d in x]
    plot_hist(x)

def plot_proc_cluster():

    DATA1 = import_csv(CSV1)
    N = calc_bins(DATA1[0])
    DATA1 = [running_mean(fourier(d), N) for d in DATA1]
    a = avg(DATA1)

    x = [float(cos_sim(a, d1)) for d1 in DATA1]
    y = [float(pearson(a, d1)) for d1 in DATA1]
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Pearson Correlation')
    plt.scatter(x, y, color = 'b')

    rands = [r.sample(list(s), len(s)) for s in DATA1]
    x = [float(cos_sim(a, q)) for q in rands]
    y = [float(pearson(a, q)) for q in rands]
    plt.scatter(x, y, color = 'r')

    plt.show()

def plot_proc_color_cluster():
    DATA1 = import_csv(CSV1)
    DATA2 = import_csv(CSV2)
    DATA2 = pad_with_zeros(DATA1, DATA2)
    N = calc_bins(DATA1[0])

    DATA1 = [running_mean(fourier(d), N) for d in DATA1]
    DATA2 = [running_mean(fourier(d), N) for d in DATA2]

    a = avg(DATA1)
    x = [float(cos_sim(a, d1)) for d1 in DATA1]
    y = [float(pearson(a, d1)) for d1 in DATA1]
    cluster1 = [(x[i], y[i]) for i in range(len(x))]
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Pearson Correlation')
    plt.scatter(x, y, color = 'b')

    x = [float(cos_sim(a, d2)) for d2 in DATA2]
    y = [float(pearson(a, d2)) for d2 in DATA2]
    cluster2 = [(x[i], y[i]) for i in range(len(x))]
    plt.scatter(x, y, color = 'm')

    rands = [r.sample(list(s), len(s)) for s in DATA1]
    x = [float(cos_sim(a, q)) for q in rands]
    y = [float(pearson(a, q)) for q in rands]
    cluster3 = [(x[i], y[i]) for i in range(len(x))]
    plt.scatter(x, y, color = 'r')

    plt.show()

def play_song():
    DATA = import_csv(CSV2)
    s = DATA[7]
    play(s)

def format_data_for_kmeans():
    '''
    Formats data for K-Means algorithm in the form [length, [time], [frequency], [power]]
    Returns a list of 146 lists in the specified format
    '''
    DATA1 = import_csv(CSV1)
    DATA2 = import_csv(CSV2)

    # Converted to milliseconds for convenience
    time1 =  np.array([(i/SAMPLE_FREQUENCY)*1000.0 for i in range(len(DATA1[0]))])
    time2 =  np.array([(i/SAMPLE_FREQUENCY)*1000.0 for i in range(len(DATA2[0]))])

    # milliseconds
    raw_len1 = (len(time1)/SAMPLE_FREQUENCY)*1000
    raw_len2 = (len(time2)/SAMPLE_FREQUENCY)*1000

    N1 = calc_bins(DATA1[0])
    N2 = calc_bins(DATA2[0])

    # decibels
    power1 = [running_mean(fourier(x), N1) for x in DATA1]
    power2 = [running_mean(fourier(x), N2) for x in DATA2]

    # hertz
    frq1 = arange(len(power1[0]))
    frq2 = arange(len(power2[0]))

    b1 = [[raw_len1, time1, frq1, power1[i]] for i in range(len(DATA1))]
    b2 = [[raw_len1, time2, frq2, power2[i]] for i in range(len(DATA2))]
    return b1+b2

def cleanSplit(firstList):
    bigSet = firstList[:120]
    smallSet = firstList[120:]
    return [bigSet, smallSet]

def morph(soundData):
    sound = soundData.copy()
    sound.pop(1)
    sound[1] = average(sound[1])
    sound[2] = average(sound[2])
    return sound

def morphAll(data):
    d = []
    for datum in data:
        d.append(morph(datum))
    return d

def randomSort(sets):
    clusters = [[],[]]
    length = len(sets)
    for s in range(length):
        sound = sets[s]
        clusters[random.randint(0,1)].append(sound)
    return clusters

def sort(sets):
    clusters = [[],[]]
    length = len(sets)
    for s in range(length):
        sound = sets[s]
        if s < 120:
            clusters[0].append(sound)
        else:
            clusters[1].append(sound)
    return clusters

def centroid(cluster):
    length = len(cluster)
    cent = [[],[],[]]
    for i in range(length):
        sound = cluster[i]
        cent[0].append(sound[0])
        cent[1].append(sound[1])
        cent[2].append(sound[2])
    cent = [average(elem) for elem in cent]
    return cent

def average(nums):
    if len(nums) == 0:
        return 0
    return sum(nums)/len(nums)

def cosSim(a, b):
    '''
    Computes cosine similarty between two lists and returns a list of type
    numpy.float64. NOTE: cos_sim returns a list of type numpy.complex128
    '''
    prod = dotProduct(a, b)
    return prod / (magnitude(a) * magnitude(b))

def dotProduct(a, b):
    prods = [a[i] * b[i] for i in range(len(a))]
    return sum(prods)

def magnitude(nums):
    nums = [num**2 for num in nums]
    return math.sqrt(sum(nums))

def reorder(clusters, cents):
     # is moving every sound into a new list going to be extremely slow?
    newClusts = [[] for c in clusters]
    allSounds = mergeLists(clusters)
    for sound in allSounds:
        if cosSim(sound, cents[0]) > cosSim(sound, cents[1]):
            newClusts[0].append(sound)
        else:
            newClusts[1].append(sound)
    return newClusts

def mergeLists(members):
    l = []
    for member in members:
        for m in member:
            l.append(m)
    return l

def needReordering(clusters, cents):
    for i in range(len(clusters)):
        cluster = clusters[i]
        for sound in cluster:
            sims = [cosSim(sound, cent) for cent in cents]
            # what if two indices have the same value?
            if sims.index(max(sims)) != i:
                return True
    return False

def clusterize(data):
    clusters = randomSort(data)
    cents = [centroid(c) for c in clusters]
    #print("clusters[0] size = " + str(len(clusters[0])))
    #print("clusters[1] size = " + str(len(clusters[1])))
    loops = 0
    while needReordering(clusters, cents):
        loops += 1
        clusters = reorder(clusters, cents)
        cents = [centroid(c) for c in clusters]
    #print("finished in " + str(loops) + " loops")
    return clusters

"""def perfect(randStart, easyStart):
    randBig = biggest(randStart)
    easyBig = biggest(easyStart)
    for elem in easyBig:
        if elem not in randBig:
            print("failed at " + str(elem))
            return False
    return True"""

def accuracy(randStart, easyStart):
    randBig = biggest(randStart)
    easyBig = biggest(easyStart)
    correct = 0
    for elem in easyBig:
        if elem in randBig:
            correct += 1
    return correct / len(easyBig)

def perfect(rand, start):
    return accuracy(rand, start) == 1

def biggest(x):
    sizes = [len(elem) for elem in x]
    index = sizes.index(max(sizes))
    return x[index]

def randSublist(original, desiredSize):
    sublist = []
    for i in range(desiredSize):
        index = random.randint(0, (len(original) - 1))
        elem = original.pop(index) # shrinks original
        sublist.append(elem)
    return sublist

def percToSize(perc, length):
    return int(perc/100 * length)

def train(data, percentage, hard):
    clusters = cleanSplit(data)
    sizes = [percToSize(percentage, len(cluster)) for cluster in clusters]
    # training starts off with correct clusters
    training = [randSublist(clusters[i], sizes[i]) for i in range(len(clusters))]
    if hard:
        training = clusterize(mergeLists(training))
    #print("cluster lengths: " + str([len(c) for c in training]))
    return [training, clusters]

def fillClusters(clusters, freeBirds):
    freeBirds = mergeLists(freeBirds)

    for bird in freeBirds:
        cent0 = centroid(clusters[0])
        cent1 = centroid(clusters[1])
        if cosSim(bird, cent0) > cosSim(bird, cent1):
            clusters[0].append(bird)
        else:
            clusters[1].append(bird)

    return clusters

def kMeansSuccess(data, percent):
    '''
    Classifies data using k-means algorithm
    Takes two arguments, data which is generated by the function:
    format_data_for_kmeans() and percentage, which should be given
    as a whole number. If you train on 90% of the data and test on 10%,
    call kMeansSuccess(data, 90). If the algorithm worked perfectly,
    it'll return True.  If there were any mistakes, it'll return the
    percentage that were grouped correctly
    '''
    easy = cleanSplit(data)
    training, testing = train(data, percent, True)
    hardClusters = fillClusters(training, testing)
    if perfect(hardClusters, easy):
        return True
    return accuracy(hardClusters, easy)

start = time.clock()
numbers = morphAll(format_data_for_kmeans())
end = time.clock()
elapsed = end - start
success = kMeansSuccess(numbers, 80)
if(success == True):
    print("All data points were correctly classified")
else:
    print("Percentage of data points correctly classified: " + success)
print("time elapsed: " + str(elapsed) + " seconds")

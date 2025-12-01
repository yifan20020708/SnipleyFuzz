import getopt
import os
import sys
import time
import random
import time 
import threading
import pandas as pd
from scipy.cluster import hierarchy
sys.path.append(r'..')
from snr import Messenger
from seed import Message, Seed

#  Golbal var
queue = []
restoreSeed = ''
outputfold = ''
device_type = ''
device_name = ''
number_array = []
path_score = []
crash_number = 0
round = 0


def info():
    global queue, outputfold, device_name
    path = os.path.join(outputfold, f"{device_name}_statistics.txt")
    while(1):
        seed_number = len(queue)
        number_sum = 0
        for number in number_array:
            number_sum += number
        if len(number_array) != 0:
            average = number_sum / len(number_array)
        else:
            average = 0 
        with open(path, 'w') as f:
            f.writelines("seed-number: " + str(seed_number) + "\n")
            f.writelines("path-number: " + str(len(path_score)) + "\n")
            f.writelines("crash-number: " + str(crash_number) + "\n")
            f.writelines("mutation-number: " + str(average) + "\n")
        time.sleep(5)


# read the input file and store it as seed
def readInputFile(file):
    s = Seed()
    lines = []
    with open(file, 'r') as f:
        lines = f.read().split("\n")
    for i in range(0, len(lines)):
        if "========" in lines[i]:
            mes = Message()
            for j in range(i + 1, len(lines)):
                if "========" in lines[j]:
                    i = j
                    break
                if ":" in lines[j]:
                    mes.append(lines[j])
            s.append(mes)
    return s


# read the input fold and store them as seeds
def readInputFold(fold):
    seeds = []
    files = os.listdir(fold)
    for file in files:
        print("Loading file: ", os.path.join(fold, file))
        seeds.append(readInputFile(os.path.join(fold, file)))
    return seeds


# Write the probe result that has been run into the output
def writeRecord(queue, fold):
    with open(os.path.join(fold, 'ProbeRecord.txt'), 'w') as f:
        for i in range(len(queue)):
            f.writelines("========Seed " + str(i) + "========\n")
            for j in range(len(queue[i].M)):

                f.writelines("Message Index-" + str(j) + "\n")  # write the message information
                for header in queue[i].M[j].headers:
                    f.writelines(header + ":" + queue[i].M[j].raw[header] + '\n')
                f.writelines("\n")

                f.writelines('Original Response' + "\n")  # write the original response
                f.writelines(queue[i].R[j])

                f.writelines('Probe Result:' + "\n")  # write the results of probe
                f.writelines('PI' + "\n")  # PI
                for n in queue[i].PI[j]:
                    f.write(str(n) + " ")
                f.writelines("\n")
                f.writelines('PR and PS' + "\n")
                for n in range(len(queue[i].PR[j])):
                    f.writelines("(" + str(n) + ") " + queue[i].PR[j][n])
                    f.writelines(str(queue[i].PS[j][n]) + "\n")
            f.writelines("\n")
            f.writelines("\n")
    return 0


# Read the probe results from the record, thus skip the probe process and directly start the mutation test.
def readRecordFile(file):
    queue = []
    with open(os.path.join(file), 'r') as f:
        lines = f.readlines()
        i = 0
        while i <= len(lines):
            if lines[i].startswith("========Seed"):
                seedStart = i + 1
                seedEnd = len(lines)
                for j in range(i + 1, len(lines)):
                    if lines[i].startswith("========Seed"):
                        seedEnd = j
                seed = Seed()
                index = seedStart

                while index <= seedEnd:

                    if lines[index].startswith('Message Index'):
                        message = Message()
                        responseStart = seedEnd
                        for j in range(index, seedEnd):
                            if lines[j].startswith('Original Response'):
                                responseStart = j
                                break
                        for line in lines[index + 1:responseStart - 1]:
                            message.append(line)
                        seed.M.append(message)
                        index = responseStart

                    if lines[index].startswith('Original Response'):
                        index = index + 1
                        seed.R.append(lines[index])

                    if lines[index].startswith('PI'):
                        index = index + 1
                        PIstr = lines[index]
                        PI = []
                        for n in PIstr.strip().split(' '):
                            PI.append(int(n))
                        seed.PI.append(PI)

                    if lines[index].startswith('PR and PS'):
                        index = index + 1
                        ends = seedEnd
                        PR = []
                        PS = []
                        for j in range(index, seedEnd):
                            if lines[j].startswith('Message Index'):
                                ends = j
                                break
                        for j in range(index, ends):
                            if lines[j].startswith("("):
                                PR.append(lines[j][3:])
                            elif lines[j][0].isdigit():
                                PS.append(float(lines[j].strip()))
                        seed.PR.append(PR)
                        seed.PS.append(PS)

                    index = index + 1

                i = index
                queue.append(seed)

            i = i + 1
    return queue


def dryRun(queue):
    global restoreSeed, device_type
    m = Messenger(restoreSeed, device_type)
    for i in range(0, len(queue)):
        seed = m.DryRunSend(queue[i])
        queue[i] = seed
    return False


def update_path_score(seed):
    global path_score
    for i in range(len(seed.M)):
        responsePool = seed.PR[i]
        scorePool = seed.PS[i]
        for k in range(len(responsePool)):
            path_and_score = {}
            response = responsePool[k]
            if path_score:
                flag = True
                for j in range(len(path_score)):
                    target = path_score[j]["response"]
                    target_score = path_score[j]["score"]
                    c = SimilarityScore(target.strip(), response.strip())
                    if c >= target_score:
                        flag = False
                        break
                if flag:
                    path_and_score["response"] = response
                    path_and_score["score"] = scorePool[k]
                    path_score.append(path_and_score)
            else:
                path_and_score["response"] = response
                path_and_score["score"] = scorePool[k]
                path_score.append(path_and_score)


# Calculate the edit distance of two string   
def EditDistanceRecursive(str1, str2):
    edit = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            edit[i][j] = min(edit[i - 1][j] + 1, edit[i][j - 1] + 1, edit[i - 1][j - 1] + d)
    return edit[len(str1)][len(str2)]


# Calculate the similarity score of two string
def SimilarityScore(str1, str2):
    ED = EditDistanceRecursive(str1, str2)
    return round((1 - (ED / max(len(str1), len(str2)))) * 100, 2)


# Use heuristics to detect the meaning of each byte in the message
def Probe(Seed):
    global restoreSeed, device_type, path_score

    print("*** Probe ")
    m = Messenger(restoreSeed, device_type)
    for index in range(len(Seed.M)):

        responsePool = []
        similarityScore = []
        probeResponseIndex = []

        print(Seed.M[index].raw["Content"].strip())  # test only
        # original message
        response1 = m.ProbeSend(Seed, index)  # send the probe message   ####### 
        time.sleep(1)
        response2 = m.ProbeSend(Seed, index)  # send the probe message twice

        responsePool.append(response1)
        Res_score = SimilarityScore(response1.strip(), response2.strip())
        similarityScore.append(SimilarityScore(response1.strip(), response2.strip()))
        
        if path_score:
            global_flag = True
            for i in range(0, len(path_score)):
                global_target = path_score[i]["response"]
                global_score = path_score[i]["score"]
                global_c = SimilarityScore(global_target.strip(), response1.strip())
                if global_c >= global_score:
                    global_flag = False
                    break
            if global_flag:
                path_and_score = {}
                path_and_score["response"] = response1
                path_and_score["score"] = Res_score
                path_score.append(path_and_score)
        else:
            path_and_score = {}
            path_and_score["response"] = response1
            path_and_score["score"] = Res_score
            path_score.append(path_and_score)

        # probe process
        for i in range(0, len(Seed.M[index].raw["Content"])):
            temp = Seed.M[index].raw["Content"]
            Seed.M[index].raw["Content"] = Seed.M[index].raw["Content"].strip()[:i] + Seed.M[index].raw["Content"].strip()[i + 1:]  # delete ith byte

            response1 = m.ProbeSend(Seed, index)  # send the probe message   ####### 
            time.sleep(1)
            response2 = m.ProbeSend(Seed, index)  # send the probe message twice
            print(response1,end='')
            
            if path_score:
               global_flag = True
               for k in range(0, len(path_score)):
                   global_target = path_score[k]["response"]
                   global_score = path_score[k]["score"]
                   global_c = SimilarityScore(global_target.strip(), response1.strip())
                   if global_c >= global_score:
                       global_flag = False
                       break
               if global_flag:
                   path_and_score = {}
                   path_and_score["response"] = response1
                   path_and_score["score"] = SimilarityScore(response1.strip(), response2.strip())
                   path_score.append(path_and_score)

            if responsePool:
                flag = True
                for j in range(0, len(responsePool)):
                    target = responsePool[j]
                    score = similarityScore[j]
                    c = SimilarityScore(target.strip(), response1.strip())
                    if c >= score:
                        flag = False
                        probeResponseIndex.append(j)
                        print(str(j)+" ", end='') 
                        sys.stdout.flush()
                        break
                if flag:
                    responsePool.append(response1)
                    similarityScore.append(SimilarityScore(response1.strip(), response2.strip()))
                    probeResponseIndex.append(j + 1)
                    #print(j + 1)  # test only

            Seed.M[index].raw["Content"] = temp  # restore the message

        Seed.PR.append(responsePool)
        Seed.PS.append(similarityScore)
        Seed.PI.append(probeResponseIndex)

    return Seed


def getFeature(response, score):
    feature = {}
    feature['a'] = 0
    feature['n'] = 0
    feature['s'] = 0
    length = len(response)
    score = score
    cur = ''
    pre = ''
    for i in range(len(response)):
        if response[i].isdigit():
            cur = 'n'
        elif response[i].isalpha():
            cur = 'a'
        else:
            cur = 's'

        if pre == '':
            pre = cur
        elif pre != cur:
            feature[pre] = feature[pre] + 1
        pre = cur
    feature[cur] = feature[cur] + 1
    return [feature['a'], feature['n'], feature['s'], length, score]


def formSnippets(pi, cluster, index):
    snippet = []
    for i in range(index):
        c1 = int(cluster[i][0])
        c2 = int(cluster[i][1])
        p = int(cluster[i][3])
        for j in range(len(pi)):
            if pi[j] == c1 or pi[j] == c2:
                pi[j] = p
    i = 0
    while i < len(pi)-1:
        j = i
        skip = True
        while j <= len(pi) and skip:
            j = j + 1
            if pi[j] != pi[i]:
                snippet.append([i, j - 1])
                skip = False
            if j == len(pi)-1:
                snippet.append([i, j])
                skip = False
        i = j
    return snippet


def interesting(oldSeed,index):
    global queue
    global restoreSeed, device_type
    m = Messenger(restoreSeed, device_type)
    print(oldSeed.M[index].raw["Content"])
    seed = Seed()
    seed.M = oldSeed.M
    seed = m.DryRunSend(seed)
    seed = Probe(seed)
    queue.append(seed)


def writeOutput(seed):
    global outputfold
    localtime = time.localtime(time.time())
    file = 'Crash-'+localtime+'.txt'
    with open(os.path.join(outputfold, file), 'w') as f:
        for i in range(len(seed)):
            f.writelines("Message Index-" + str(i) + "\n")  # write the message information
            for header in seed.M[i].headers:
                f.writelines(header + ":" + seed.M[i].raw[header] + '\n')
            f.writelines("\n")
    print("Found a crash @ "+localtime)
    sys.exit()


def responseHandle(seed, info):
    global crash_number
    if info.startswith("#interesting"):
        print("~~Get Interesting in :")
        interesting(seed, int(info.split('-')[1]))
        return False
    if info.startswith("#error"):
        print("~~Something wrong with the target infomation (e.g. IP addresss or port)")
    if info.startswith("#crash"):
        crash_number += 1
        print(f"Crash!!!!  number({str(crash_number)})")
        writeOutput(seed)
    return True


def SnippetMutate(seed, restoreSeed):
    global device_type
    m = Messenger(restoreSeed, device_type)
    for i in range(len(seed.M)):
        pool = seed.PR[i]
        poolIndex = seed.PI[i]
        similarityScores = seed.PS[i]

        featureList = []
        for j in range(len(pool)):
            featureList.append(getFeature(pool[j].strip(), similarityScores[j]))
        df = pd.DataFrame(featureList)
        cluster = hierarchy.linkage(df, method='average', metric='euclidean')
        seed.ClusterList.append(cluster)
        mutatedSnippet = []
        for index in range(len(cluster)):
            snippetsList = formSnippets(poolIndex, cluster, index)
            for snippet in snippetsList:
                if snippet not in mutatedSnippet:
                    mutatedSnippet.append(snippet)
                    tempMessage = seed.M[i].raw["Content"]

                    # ========  BitFlip ========
                    print("--BitFlip")
                    message = seed.M[i].raw["Content"]
                    asc = ""
                    for o in range(snippet[0], snippet[1]):
                        asc=asc+(chr(255-ord(message[o])))
                    message = message[:snippet[0]] + asc + message[snippet[1] + 1:]
                    seed.M[i].raw["Content"] = message
                    responseHandle(seed, m.SnippetMutationSend(seed, i, path_score))
                    seed.M[i].raw["Content"] = tempMessage

                    # ========  Empty ========
                    print("--Empty")
                    message = seed.M[i].raw["Content"]
                    message = message[:snippet[0]] + message[snippet[1]+1:]
                    seed.M[i].raw["Content"] = message
                    responseHandle(seed, m.SnippetMutationSend(seed, i, path_score))
                    seed.M[i].raw["Content"] = tempMessage

                    # ========  Repeat ========
                    print("--Repeat")
                    message = seed.M[i].raw["Content"]
                    t = random.randint(2, 5)
                    message = message[:snippet[0]] + message[snippet[0]:snippet[1]] * t + message[snippet[1] + 1:]
                    seed.M[i].raw["Content"] = message
                    responseHandle(seed, m.SnippetMutationSend(seed, i, path_score))
                    seed.M[i].raw["Content"] = tempMessage

                    # ========  Interesting ========
                    print("--Interesting")
                    interestingString = ['on','off','True','False','0','1']
                    for t in interestingString:
                        message = seed.M[i].raw["Content"]
                        message = message[:snippet[0]] + t + message[snippet[1] + 1:]
                        seed.M[i].raw["Content"] = message
                        responseHandle(seed, m.SnippetMutationSend(seed, i, path_score))
                        seed.M[i].raw["Content"] = tempMessage
        seed.Snippet.append(mutatedSnippet)
    return 0


def Havoc(queue, restoreSeed):
    print("*Havoc")
    global device_type
    m = Messenger(restoreSeed, device_type)

    t = random.randint(0,len(queue)-1)
    seed = queue[t]

    i = random.randint(0,len(seed.M)-1)
    snippets = seed.Snippet[i]
    message = seed.M[i].raw["Content"]
    tempMessage = seed.M[i].raw["Content"]

    n = random.randint(0,len(snippets)-1)
    snippet = snippets[n]

    pick = random.randint(0,5)
    
    if pick == 0: 
        asc = ""
        for o in range(snippet[0], snippet[1]):
            asc=asc+(chr(255-ord(message[o])))
        message = message[:snippet[0]] + asc + message[snippet[1] + 1:]
        seed.M[i].raw["Content"] = message
        temp = responseHandle(seed, m.SnippetMutationSend(seed,i))
        seed.M[i].raw["Content"] = tempMessage
        return temp

    elif pick == 1: 
        message = seed.M[i].raw["Content"]
        message = message[:snippet[0]] + message[snippet[1]+1:]
        seed.M[i].raw["Content"] = message
        temp = responseHandle(seed, m.SnippetMutationSend(seed,i))
        seed.M[i].raw["Content"] = tempMessage
        return temp
    
    elif pick == 2: 
        message = seed.M[i].raw["Content"]
        t = random.randint(2, 5)
        message = message[:snippet[0]] + message[snippet[0]:snippet[1]] * t + message[snippet[1] + 1:]
        seed.M[i].raw["Content"] = message
        temp = responseHandle(seed, m.SnippetMutationSend(seed,i))
        seed.M[i].raw["Content"] = tempMessage
        return temp

    elif pick == 3: 
        interestingString = ['on','off','True','False','0','1']
        interesting = random.randint(0,5)
        t = interestingString[interesting]
        message = seed.M[i].raw["Content"]
        message = message[:snippet[0]] + t + message[snippet[1] + 1:]
        seed.M[i].raw["Content"] = message
        temp = responseHandle(seed, m.SnippetMutationSend(seed,i))
        seed.M[i].raw["Content"] = tempMessage
        return temp
    
    elif pick == 4:
        start = random.randint(0,len(message)-1)
        end = random.randint(start,len(message))
        asc = ""
        for o in range(start, end):
            asc=asc+(chr(255-ord(message[o])))
        message = message[:start] + asc + message[end + 1:]
        seed.M[i].raw["Content"] = message
        temp = responseHandle(seed, m.SnippetMutationSend(seed,i))
        seed.M[i].raw["Content"] = tempMessage
        return temp

    return True


def getArgs(argv):
    inputfold = ''
    outputfold = ''
    restorefile = ''
    recordfile = ''
    devicetype = ''
    devicename = ''
    try:
        opts, args = getopt.getopt(argv, "hi:r:o:c:d:t:", ["ifold=", "rfile=", "ofold=", "cfile=", "devicename=", "devicetype="])
    except getopt.GetoptError:
        print('Snipuzz.py -i <inputfold> -r <restrefile> -o <outputfold> -d <devicename> -t <devicetype> (-c <recordfile>)')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfold> -r <restrefile> -o <outputfold> (-c <recordfile>)')
            sys.exit()
        elif opt in ("-i", "--ifold"):
            inputfold = arg
        elif opt in ("-r", "--rfile"):
            restorefile = arg
        elif opt in ("-o", "--ofold"):
            outputfold = arg
        elif opt in ("-c", "--cfile"):
            recordfile = arg
        elif opt in ("-d", "--devicename"):
            devicename = arg
        elif opt in ("-t", "--devicetype"):
            devicetype = arg
        if not recordfile:
            recordfile = 'unavailable'
    print('Input fold: ', inputfold)
    print('Restore file: ', restorefile)
    print('Output fold: ', outputfold)
    print('Record file: ', recordfile)
    print('Device name: ', devicename)
    print('Device type: ', devicetype)
    return inputfold, restorefile, outputfold, recordfile, devicename, devicetype


def main(argv):
    global queue, restoreSeed, outputfold, device_type, device_name, round, number_array
    inputfold, restorefile, outputfold, recordfile, device_type, device_name = getArgs(argv)
    restoreSeed = readInputFile(restorefile)
    queue =  readInputFold(inputfold)
    if recordfile and os.path.exists(recordfile):
        queue = readRecordFile(recordfile)
        for seed in queue:
            update_path_score(seed)
        thread = threading.Thread(target = info)
        thread.start() 
        for seed in queue:
            seed.display()
        if (dryRun(queue)):  # Dry Run
            print('#### Dry run failed, check the inputs or connection.')
            sys.exit()
    else:
        queue = readInputFold(inputfold)
        if (dryRun(queue)):  # Dry Run
            print('#### Dry run failed, check the inputs or connection.')
            sys.exit()
        for i in range(len(queue)):
            queue[i] = Probe(queue[i])
        thread = threading.Thread(target = info)
        thread.start()
        writeRecord(queue, outputfold)
    skip = False
    number = 0
    while (1):
        if not skip:
            i=0
            while i < len(queue):
                if not queue[i].isMutated:
                    SnippetMutate(queue[i], restoreSeed)
                i = i + 1
        skip = True
        number += 1
        skip = Havoc(queue, restoreSeed)
        if skip == False:
            round += 1
            number_array.append(number)
            number = 0


if __name__ == "__main__":
    main(sys.argv[1:])
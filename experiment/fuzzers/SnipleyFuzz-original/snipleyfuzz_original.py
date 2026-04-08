# -*- coding: utf-8 -*-
import os
import sys
import time
import random
import time
import string
import argparse
import threading
import logging
import numpy as np
from pathlib import Path
from scipy.stats import binom
import pandas as pd
from scipy.cluster import hierarchy
from colorama import init, Fore
sys.path.append(r'..')
import cfg
from base import Message, Seed
from interact import Messenger, SimilarityScore

# 进行日志初始化
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 全局变量
queue = []
restoreSeed = '' 
outputfold = ''
history_combination = []
crash_number = 0
number_array = []
round = 0
path_score = []
device_name = ''
device_type = ''

# 统计运行信息
def info():
    global queue, crash_number, number_array, path_score, outputfold, device_name
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


def readInputFold(fold):
    seeds = []
    files = os.listdir(fold)
    for file in files:
        logger.info(f"Loading file: {os.path.join(fold, file)}")
        seeds.append(readInputFile(os.path.join(fold, file)))
    return seeds


def writeRecord(queue, fold):
    global device_name
    with open(os.path.join(fold, f'{device_name}_ProbeRecord.txt'), 'w') as f:
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


def readRecordFile(file):
    queue = []
    with open(os.path.join(file), 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            if lines[i].startswith("========Seed"):
                seedStart = i + 1
                seedEnd = len(lines)
                for j in range(i + 1, len(lines)):
                    if lines[j].startswith("========Seed"):
                        seedEnd = j
                        break
                seed = Seed()
                index = seedStart
                while index < seedEnd:
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
    return queue


def dryRun(queue):
    logger.info(f"{Fore.BLUE}Start to exec dryRun Process!{Fore.RESET}")
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
                
                
def Probe(Seed):
    global restoreSeed, path_score, device_type
    m = Messenger(restoreSeed, device_type)
    for index in range(len(Seed.M)):
        
        responsePool = []
        similarityScore = []
        probeResponseIndex = []
        
        # Calculation of self-similarity scores
        response1 = m.ProbeSend(Seed, index)  # send the probe message   
        time.sleep(1)
        response2 = m.ProbeSend(Seed, index)  # send the probe message twice
        
        logger.info("========" + "Message" + str(index) + "========")
        logger.info("Message" + str(index) + ":(first)" + response1)
        logger.info("Message" + str(index) + ":(second)" + response2)
        logger.info("========" + "Message" + str(index) + "========")
        
        responsePool.append(response1)
        Res_score = SimilarityScore(response1.strip(), response2.strip())
        similarityScore.append(Res_score)

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
           
           # Calculation of self-similarity scores
           response1 = m.ProbeSend(Seed, index)  # send the probe message
           if response1 == '#crash':
               writeOutput(Seed)
           time.sleep(1)
           response2 = m.ProbeSend(Seed, index)  # send the probe message twice
           if response2 == '#crash':
               writeOutput(Seed)
               
           logger.info(Seed.M[index].raw["Content"])
           logger.info("Mutation" + str(i) + ":(first)" + response1)
           logger.info("Mutation" + str(i) + ":(second)" + response2)
           
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
                       logger.info("Mutation" + str(i) + " is similar")
                       sys.stdout.flush()
                       break
               if flag:
                    responsePool.append(response1)
                    similarityScore.append(SimilarityScore(response1.strip(), response2.strip()))
                    logger.info("Mutation" + str(i) + " is unique" + "\n")
                    probeResponseIndex.append(j + 1)
            
           Seed.M[index].raw["Content"] = temp  # restore the message

        Seed.PR.append(responsePool)
        Seed.PS.append(similarityScore)
        Seed.PI.append(probeResponseIndex)

    return Seed


def writeOutput(seed):
    global outputfold, crash_number, device_name
    localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
    file = f'{device_name}-Crash-' + str(crash_number) + ":" + localtime + '.txt'

    with open(os.path.join(outputfold, file), 'w') as f:
        for i in range(len(seed.M)):
            f.writelines("Message Index-" + str(i) + "\n")  # write the message information
            for header in seed.M[i].headers:
                f.writelines(header + ":" + seed.M[i].raw[header] + '\n')
            f.writelines("\n")
    logger.info("Found a crash @ " + localtime)
    sys.exit()
    

def getFeature(response, score):
    feature = {}
    feature['a'] = 0  # Letter count in response
    feature['n'] = 0  # Digit count in response
    feature['s'] = 0  # Special character count in response
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
                pi[j] = len(cluster) + 1 + i
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


def interesting(oldSeed, index):
    global queue
    global restoreSeed, device_type
    m = Messenger(restoreSeed, device_type)
    logger.info(oldSeed.M[index].raw["Content"])
    seed = Seed()
    for i in range(len(oldSeed.M)):
        message = Message()
        seed.M.append(message)
    seed.M[index].headers = oldSeed.M[index].headers
    for i in seed.M[index].headers:
        seed.M[index].raw[i] = oldSeed.M[index].raw[i]
    seed = m.DryRunSend(seed)
    seed = Probe(seed)
    queue.append(seed)
    
    
def responseHandle(seed, info):
    global crash_number
    if info.startswith("#interesting"):
        logger.info("~~Get Interesting in :")
        interesting(seed, int(info.split('-')[1]))
        return False
    if info.startswith("#error"):
        logger.error("~~Something wrong with the target infomation (e.g. IP addresss or port)")
    if info.startswith("#crash"):
        crash_number += 1
        logger.error(f"Crash!!!!  number({str(crash_number)})")
        writeOutput(seed)
    return True


def SnippetMutate(seed, restoreSeed):
    global path_score, device_type
    m = Messenger(restoreSeed, device_type)
    
    for i in range(len(seed.M)):
        pool = seed.PR[i]
        poolIndex = seed.PI[i]
        similarityScores = seed.PS[i]
        poolIndex_tmp = []
        for ii in poolIndex:
            poolIndex_tmp.append(ii)
    
        # update the number of times the message was used and the interval
        seed.M[i].number_used_message += 1
        seed.M[i].interval_message = 0
        for j in range(len(seed.M)):
            if j != i:
                seed.M[j].interval_message += 1
                
        logger.info(f"{Fore.BLUE}Start to exec SnippetMutate process for Message{i}! {i+1}/{len(seed.M)}{Fore.RESET}")
        
        featureList = []
        for j in range(len(pool)):
            featureList.append(getFeature(pool[j].strip(), similarityScores[j]))
            
        df = pd.DataFrame(featureList)
        cluster = hierarchy.linkage(df, method='average', metric='euclidean')
        
        seed.ClusterList.append(cluster)
        
        mutatedSnippet = []
        for index in range(len(cluster)):
            snippetsList = formSnippets(poolIndex, cluster, index)
            snippet_all_info_array = {}
            snippet_info_array = []
            logger.info(f"{Fore.BLUE}Start to exec SnippetMutate process for snippet in cluster round{index}! {index + 1}/{len(cluster)}{Fore.RESET}")
            
            # Initialize the snippet property for message
            for snippet in snippetsList:
                if "Fail to bind device" in pool[poolIndex_tmp[snippet[0]]]:
                    continue
                snippet_info = {}
                snippet_info["fragment"] = snippet
                snippet_info["number"] = 0
                snippet_info["shapley"] = 0
                snippet_info_array.append(snippet_info)
            snippet_all_info_array["snippets"] = snippet_info_array
            snippet_all_info_array["number"] = 0
            snippet_all_info_array["interested"] = 0
            seed.M[i].snippet.append(snippet_all_info_array)
            
            for snippet_index in range(len(seed.M[i].snippet[index]["snippets"])):
                snippet = seed.M[i].snippet[index]["snippets"][snippet_index]
                fragment = snippet["fragment"]
                seed.M[i].snippet[index]["snippets"][snippet_index]["number"] += 1
                
                logger.info(f"{Fore.BLUE}Start to exec SnippetMutate process for snippet{snippet_index}! {snippet_index+1}/{len(seed.M[i].snippet[index])} (round{index}/{len(cluster)-1}:message{i}){Fore.RESET}")
        
                if fragment not in mutatedSnippet:
                    tempString = '"id":1,'
                    mutatedSnippet.append(fragment)
                    tempMessage = seed.M[i].raw["Content"]
                    
                    # ========  BitFlip ========
                    logger.info("--BitFlip")
                    message = seed.M[i].raw["Content"]
                    asc = ""
                    for o in range(fragment[0], fragment[1] + 1):
                        asc = asc + (chr(255 - ord(message[o])))
                    message = message[:fragment[0]] + asc + message[fragment[1] + 1:]
                    if tempString in message:
                        seed.M[i].raw["Content"] = message
                        if responseHandle(seed, m.SnippetMutationSend(seed,i,path_score)) == False:
                            seed.M[i].snippet[index]["snippets"][snippet_index]["shapley"] += 1000
                            seed.number_interested += 1000
                            seed.M[i].number_interested_message += 1000
                            seed.M[i].snippet[index]["interested"] += 1000
                        seed.M[i].raw["Content"] = tempMessage
                    
                    # ========  Empty ========
                    logger.info("--Empty")
                    message = seed.M[i].raw["Content"]
                    message = message[:fragment[0]] + message[fragment[1]+1:]
                    if tempString in message:
                        seed.M[i].raw["Content"] = message
                        if responseHandle(seed, m.SnippetMutationSend(seed,i,path_score)) == False:
                            seed.M[i].snippet[index]["snippets"][snippet_index]["shapley"] += 1000
                            seed.number_interested += 1000
                            seed.M[i].number_interested_message += 1000
                            seed.M[i].snippet[index]["interested"] += 1000
                        seed.M[i].raw["Content"] = tempMessage
                        
                    # ========  Repeat ========
                    logger.info("--Repeat")
                    message = seed.M[i].raw["Content"]
                    t = random.randint(2, 5)
                    if message[fragment[0]:fragment[1]+1].isdigit() == False:
                        message = message[:fragment[0]] + message[fragment[0]:fragment[1]+1] * t + message[fragment[1] + 1:]
                        if tempString in message and message.count('"id"') == 1:
                            seed.M[i].raw["Content"] = message
                            if responseHandle(seed, m.SnippetMutationSend(seed,i,path_score)) == False:
                                seed.M[i].snippet[index]["snippets"][snippet_index]["shapley"] += 1000
                                seed.number_interested += 1000
                                seed.M[i].number_interested_message += 1000
                                seed.M[i].snippet[index]["interested"] += 1000
                            seed.M[i].raw["Content"] = tempMessage
                    
                    # ========  Random Bytes Flip =========
                    logger.info("--Random Bytes Flip")
                    message = seed.M[i].raw["Content"]
                    index_array = []
                    for index_number in range(fragment[0], fragment[1] + 1):
                        index_array.append(index_number)
                    mutation_number = random.randint(1, fragment[1] - fragment[0] + 1)
                    mutation_array = random.sample(index_array, mutation_number)
                    asc = ""
                    for o in range(fragment[0], fragment[1]+1):
                        if o in mutation_array:
                            asc = asc + (chr(255 - ord(message[o])))
                        else:
                            asc = asc + message[o]
                    message = message[:fragment[0]] + asc + message[fragment[1] + 1:]
                    if tempString in message:
                        seed.M[i].raw["Content"] = message
                        if responseHandle(seed, m.SnippetMutationSend(seed,i,path_score)) == False:
                            seed.M[i].snippet[index]["snippets"][snippet_index]["shapley"] += 1000
                            seed.number_interested += 1000
                            seed.M[i].number_interested_message += 1000
                            seed.M[i].snippet[index]["interested"] += 1000
                        seed.M[i].raw["Content"] = tempMessage
                    
                    # ========  Random Bytes increase(Type one) ========
                    logger.info("--Random Bytes increase(Type one)")
                    message = seed.M[i].raw["Content"]
                    index_array = []
                    message_front = message[:fragment[0]]
                    message_behind = message[fragment[1] + 1:]
                    for index_number in range(fragment[0], fragment[1] + 1):
                        index_array.append(index_number)
                    mutation_number = random.randint(1, fragment[1] - fragment[0] + 1)
                    mutation_array = random.sample(index_array, mutation_number)
                    # random increase a letter
                    message = seed.M[i].raw["Content"]
                    asc = ""
                    for o in range(fragment[0], fragment[1]+1):
                        if o in mutation_array:
                            asc = asc + message[o] + random.choice(string.ascii_letters)
                        else:
                            asc = asc + message[o]
                    message = message_front + asc + message_behind
                    if tempString in message:
                        seed.M[i].raw["Content"] = message
                        if responseHandle(seed, m.SnippetMutationSend(seed,i,path_score)) == False:
                            seed.M[i].snippet[index]["snippets"][snippet_index]["shapley"] += 1000
                            seed.number_interested += 1000
                            seed.M[i].number_interested_message += 1000
                            seed.M[i].snippet[index]["interested"] += 1000
                        seed.M[i].raw["Content"] = tempMessage
                    # random increase a special characters
                    message = seed.M[i].raw["Content"]
                    asc = ""
                    for o in range(fragment[0], fragment[1]+1):
                        if o in mutation_array:
                            asc = asc + message[o] + random.choice(string.punctuation)
                        else:
                            asc = asc + message[o]
                    message = message_front + asc + message_behind
                    if tempString in message:
                        seed.M[i].raw["Content"] = message
                        if responseHandle(seed, m.SnippetMutationSend(seed,i,path_score)) == False:
                            seed.M[i].snippet[index]["snippets"][snippet_index]["shapley"] += 1000
                            seed.number_interested += 1000
                            seed.M[i].number_interested_message += 1000
                            seed.M[i].snippet[index]["interested"] += 1000
                        seed.M[i].raw["Content"] = tempMessage
        seed.Snippet.append(mutatedSnippet)
    return 0


def sim_score_for_seed(seed1, seed2):
    message_list = []
    n = 2
    for message2 in seed2.M:
        message_list.append(message2.raw["Content"].strip())

    similarity_score = 0
    for message1 in seed1.M:
        score_list = []
        for Content in message_list:
            score_list.append(calculate_ngram_similarity_message(message1.raw["Content"].strip(), Content, n))
        score_for_message = max(score_list)
        similarity_score += score_for_message
        index = score_list.index(score_for_message)
        message_list.pop(index)
    
    similarity_score /= len(seed1.M)
    return similarity_score


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


def generate_ngrams(text, n):
    # Generates an N-gram of the specified length
    ngrams = []
    text_length = len(text)
    for i in range(text_length - n + 1):
        ngrams.append(text[i:i+n])
    return ngrams


def calculate_ngram_similarity_message(text1, text2, n):
    # Calculate the N-gram similarity between two texts
    ngrams1 = generate_ngrams(text1, n)
    ngrams2 = generate_ngrams(text2, n)
    # Computes intersection and union
    intersection = len(set(ngrams1).intersection(ngrams2))
    union = len(set(ngrams1).union(ngrams2))
    # Associative editing distance
    edit_distance = EditDistanceRecursive(text1, text2)
    similarity = (intersection / union) - (edit_distance / max(len(text1), len(text2)))
    return similarity


def seed_potential(queue):
    ratio_PR = 0.1
    ratio_interested = 1 - ratio_PR
    least_probability = 0.2
    score_array = []
    potential_array = []
    # calculate seed score
    for seed in queue:
        average_PR = 0
        for i in range(len(seed.M)):
            average_PR += len(seed.PR[i])
        average_PR = average_PR / len(seed.M)
        average_interested = seed.number_interested / len(seed.M)
        score = ratio_PR * average_PR + ratio_interested * average_interested
        score_array.append(score)
    # calculate seed potential
    for i in range(len(queue)):
        similarity = 0
        for j in range(len(queue)):
            similarity += score_array[j] * sim_score_for_seed(queue[i], queue[j])
        potential_array.append(similarity)
    # process the potential
    max_potential = max(potential_array)
    min_potential = min(potential_array)
    if max_potential == min_potential:
        for i in range(len(potential_array)):
            potential_array[i] = 1
    else:
        increase = (1 - least_probability) / (max_potential - min_potential)
        for i in range(len(potential_array)):
            potential_array[i] = least_probability + increase * (potential_array[i] - min_potential)
    return potential_array


def message_potential(seed):
    ratio_PR_message = 0.1
    ratio_interested_message = 1 - ratio_PR_message
    least_probability = 0.05
    n = 2
    score_array_message = []
    potential_array_message = []
    # calculate message score
    for i in range(len(seed.M)):
        score_message = ratio_PR_message * len(seed.PR[i]) + ratio_interested_message * seed.M[i].number_interested_message
        score_array_message.append(score_message)
    # calculate message potential
    for i in range(len(seed.M)):
        similarity = 0
        for j in range(len(seed.M)):
            similarity += score_array_message[j] * calculate_ngram_similarity_message(seed.M[i].raw["Content"].strip(), seed.M[j].raw["Content"].strip(), n)
        potential_array_message.append(similarity)
    # process message potential
    max_potential = max(potential_array_message)
    min_potential = min(potential_array_message)
    if max_potential == min_potential:
        for i in range(len(potential_array_message)):
            potential_array_message[i] = 1
    else:
        increase = (1 - least_probability) / (max_potential - min_potential)
        for i in range(len(potential_array_message)):
            potential_array_message[i] = least_probability + increase * (potential_array_message[i] - min_potential)
    return potential_array_message


def discrete_exponential(p, n_samples):
    samples = np.arange(1, n_samples + 1)
    probabilities = (1 - p)**samples * p
    probabilities /= np.sum(probabilities)
    return np.random.choice(samples, size=1, p=probabilities)


def update_shapley_snippet(seed, message_index, snippets, mutation_index_array, mutation_types, response, cluster_round, restoreSeed):
    global history_combination, number_array, path_score, device_type
    m = Messenger(restoreSeed, device_type)
    logger.info(f"{Fore.BLUE}UPDATE! Length:{len(mutation_index_array)}{Fore.RESET}")
    if len(mutation_index_array) == 1:
        seed.M[message_index].snippet[cluster_round]["snippets"][mutation_index_array[0]]["shapley"] += 1000
        return ""
    # process a snippet at a time
    for i in range(len(mutation_index_array)):
        # generate a list except the target snippet
        update_mutation_array_index = []
        update_mutation_types = []
        for j in range(len(mutation_index_array)):
            if j != i:
                update_mutation_array_index.append(mutation_index_array[j])
                update_mutation_types.append(mutation_types[j])
        # generate a mutated message according to update_mutation_array_index
        tempMessage = seed.M[message_index].raw["Content"]
        length_message = len(tempMessage)
        for k in range(len(update_mutation_array_index)):
            seed = mutation_generation(seed, message_index, snippets[update_mutation_array_index[k]]["fragment"], update_mutation_types[k], length_message)
        # recovery
        response_update = m.sendMessage(seed.M[message_index])
        # Using function recursion to update shapley values
        if responseHandle(seed, m.SnippetMutationSend(seed, message_index, path_score)) == True:
            seed.M[message_index].raw["Content"] = tempMessage
            seed.M[message_index].snippet[cluster_round]["snippets"][mutation_index_array[i]]["shapley"] += 1000
        elif response_update == response:
            number_array.append(0)
            seed.M[message_index].raw["Content"] = tempMessage
            history_combination_information = {}
            history_combination_information["mutation_index"] = update_mutation_array_index
            history_combination_information["mutation_type"] = update_mutation_types
            history_combination.append(history_combination_information)
            update_shapley_snippet(seed, message_index, snippets, update_mutation_array_index, update_mutation_types, response_update, cluster_round, restoreSeed)
        else:
            number_array.append(0)
            seed.M[message_index].raw["Content"] = tempMessage
            seed.M[message_index].snippet[cluster_round]["snippets"][mutation_index_array[i]]["shapley"] += 1000
            seed.number_interested += 1000
            seed.M[message_index].number_interested_message += 1000
            seed.M[message_index].snippet[cluster_round]["interested"] += 1000
            history_combination_information = {}
            history_combination_information["mutation_index"] = update_mutation_array_index
            history_combination_information["mutation_type"] = update_mutation_types
            history_combination.append(history_combination_information)
            update_shapley_snippet(seed, message_index, snippets, update_mutation_array_index, update_mutation_types, response_update, cluster_round, restoreSeed)
    return ""


def mutation_generation(seed, message_index, snippet, pick, length):
    logger.info("*mutation generation")
    message = seed.M[message_index].raw["Content"]
    value = len(message) - length
    snippet[0] = snippet[0] + value
    snippet[1] = snippet[1] + value

    if pick == 0:
        # ========  BitFlip  ========
        asc = ""
        for o in range(min(snippet[0],len(message) - 1), min(snippet[1] + 1, len(message) - 1)):
            asc = asc + (chr(255 - ord(message[o])))
        message = message[:snippet[0]] + asc + message[snippet[1] + 1:]
        seed.M[message_index].raw["Content"] = message
        snippet[0] = snippet[0] - value
        snippet[1] = snippet[1] - value
        return seed
    
    elif pick == 1:
        # ========  Empty ==========
        message = message[:snippet[0]] + message[snippet[1] + 1:]
        seed.M[message_index].raw["Content"] = message
        snippet[0] = snippet[0] - value
        snippet[1] = snippet[1] - value
        return seed
        
    elif pick == 2:
        # ========  Repeat ========
        t = random.randint(2, 5)
        message = message[:snippet[0]] + message[snippet[0]:snippet[1] + 1] * t + message[snippet[1] + 1:]
        seed.M[message_index].raw["Content"] = message
        snippet[0] = snippet[0] - value
        snippet[1] = snippet[1] - value
        return seed
        
    elif pick == 3:
        # ======== Random Bytes Flip ===========
        index_array = []
        for index in range(snippet[0], snippet[1] + 1):
            index_array.append(index)
        mutation_number = random.randint(1, snippet[1] - snippet[0] + 1)
        mutation_array = random.sample(index_array, mutation_number)
        asc = ""
        for o in range(min(snippet[0],len(message) - 1), min(snippet[1] + 1, len(message) - 1)):
            if o in mutation_array:
                asc = asc + (chr(255 - ord(message[o])))
            else:
                asc = asc + message[o]
        message = message[:snippet[0]] + asc + message[snippet[1] + 1:]
        seed.M[message_index].raw["Content"] = message
        snippet[0] = snippet[0] - value
        snippet[1] = snippet[1] - value
        return seed
    
    elif pick == 4:
        # ========  Random Bytes delete ========
        index_array = []
        message_front = message[:snippet[0]]
        message_behind = message[snippet[1] + 1:]
        for index in range(snippet[0], snippet[1] + 1):
            index_array.append(index)
        mutation_number = random.randint(1, snippet[1] - snippet[0] + 1)
        mutation_array = random.sample(index_array, mutation_number)
        asc = ""
        for o in range(min(snippet[0],len(message) - 1), min(snippet[1] + 1, len(message) - 1)):
            if o in mutation_array:
                continue
            else:
                asc = asc + message[o]
        message = message_front + asc + message_behind
        seed.M[message_index].raw["Content"] = message
        snippet[0] = snippet[0] - value
        snippet[1] = snippet[1] - value
        return seed
    
    elif pick == 5:
        # ========  Random Bytes increase(Type one) ========
        index_array = []
        message_front = message[:snippet[0]]
        message_behind = message[snippet[1] + 1:]
        for index in range(snippet[0], snippet[1] + 1):
            index_array.append(index)
        mutation_number = random.randint(1, snippet[1] - snippet[0] + 1)
        mutation_array = random.sample(index_array, mutation_number)
        asc = ""
        mutation_type = random.randint(0, 2)
        if mutation_type == 0:
            for o in range(min(snippet[0],len(message) - 1), min(snippet[1] + 1, len(message) - 1)):
                if o in mutation_array:
                    asc = asc + message[o] + str(random.randint(0, 9))
                else:
                    asc = asc + message[o]
        elif mutation_type == 1:
            for o in range(min(snippet[0],len(message) - 1), min(snippet[1] + 1, len(message) - 1)):
                if o in mutation_array:
                    asc = asc + message[o] + random.choice(string.ascii_letters)
                else:
                    asc = asc + message[o]
        else:
            for o in range(min(snippet[0],len(message) - 1), min(snippet[1] + 1, len(message) - 1)):
                if o in mutation_array:
                    asc = asc + message[o] + random.choice(string.punctuation)
                else:
                    asc = asc + message[o]
        message = message_front + asc + message_behind
        seed.M[message_index].raw["Content"] = message
        snippet[0] = snippet[0] - value
        snippet[1] = snippet[1] - value
        return seed
    
    elif pick == 6:
        # ========  Random Bytes increase(Type one) ========
        index_array = []
        message_front = message[:snippet[0]]
        message_behind = message[snippet[1] + 1:]
        for index in range(snippet[0], snippet[1] + 1):
            index_array.append(index)
        mutation_number = random.randint(1, snippet[1] - snippet[0] + 1)
        mutation_array = random.sample(index_array, mutation_number)
        mutation_types = []
        for _ in range(mutation_number):
            mutation_types.append(random.randint(0,2))
        asc = ""
        for o in range(min(snippet[0],len(message) - 1), min(snippet[1] + 1, len(message) - 1)):
            if o in mutation_array:
                asc = asc + message[0]
                index_char = mutation_array.index(o)
                if mutation_types[index_char] == 0:
                    asc = asc + str(random.randint(0, 9))
                elif mutation_types[index_char] == 1:
                    asc = asc + random.choice(string.ascii_letters)
                else:
                    asc = asc + random.choice(string.punctuation)
            else:
                asc = asc + message[o]
        message = message_front + asc + message_behind
        seed.M[message_index].raw["Content"] = message
        snippet[0] = snippet[0] - value
        snippet[1] = snippet[1] - value
        return seed
    return seed


def advanced_mutate(queue, restoreSeed):
    global history_combination, path_score, device_type
    m = Messenger(restoreSeed, device_type)
    ratio_potential = cfg.POTENTIAL_SEED_RADIO
    ratio_number = cfg.NUMBER_SEED_RADIO
    ratio_interval = cfg.INTERVAL_SEED_RADIO
    
    potential = seed_potential(queue)
    number = []
    interval = []
    priority = []
    probability = []
    seed_index_array = []
    for i in range(len(queue)):
        seed_index_array.append(i)
    
    # process the number of times the seed has been used
    for seed in queue:
        number.append(seed.number_used)
    max_number = max(number)
    min_number = min(number)
    if max_number == min_number:
        for i in range(len(number)):
            number[i] = 1
    else:
        increase_number = 1 / (max_number - min_number)
        for i in range(len(number)):
            number[i] = 1 - increase_number * (number[i] - min_number)

    # process the number of times since the seed was last used
    for seed in queue:
        interval.append(seed.interval)
    max_interval = max(interval)
    min_interval = min(interval)
    if max_interval == min_interval:
        for i in range(len(interval)):
            interval[i] = 1
    else:
        increase_interval = 1 / (max_interval - min_interval)
        for i in range(len(interval)):
            interval[i] = increase_interval * (interval[i] - min_interval)      
        
    # caculate the priority of seed
    all_priority = 0
    for i in range(len(queue)):
        priority_seed = ratio_potential * potential[i] + ratio_number * number[i] + ratio_interval * interval[i]
        priority.append(priority_seed)
        all_priority += priority_seed
        
    # caculate the probability of seed
    for i in range(len(queue)):
        probability.append(priority[i] / all_priority)
    
    # choose a seed according to probability
    seed_index = random.choices(seed_index_array, weights = probability, k = 1)[0]
    seed = queue[seed_index]
    
    # update the information of seed
    queue[seed_index].number_used += 1
    queue[seed_index].interval = 0
    for i in range(len(queue)):
        if i != seed_index:
            queue[i].interval += 1

    # 消息的优先性选择
    ratio_potential_message = cfg.POTENTIAL_MSG_RADIO
    ratio_number_message = cfg.NUMBER_MSG_RADIO
    ratio_interval_message = cfg.INTERVAL_MSG_RADIO
    
    potential_message = message_potential(seed)
    number_message = []
    interval_message = []
    priority_message = []
    probability_message = []
    message_index_array = []
    for i in range(len(seed.M)):
        message_index_array.append(i)
    
    # process the number of times the message has been used
    for i in range(len(seed.M)):
        number_message.append(seed.M[i].number_used_message)
    max_number_message = max(number_message)
    min_number_message = min(number_message)
    if max_number_message == min_number_message:
        for i in range(len(number_message)):
            number_message[i] = 1
    else:
        increase_number_message = 1 / (max_number_message - min_number_message)
        for i in range(len(number_message)):
            number_message[i] = 1 - increase_number_message * (number_message[i] - min_number_message)
        
    # process the number of times since the message was last used
    for i in range(len(seed.M)):
        interval_message.append(seed.M[i].interval_message)
    max_interval_message = max(interval_message)
    min_interval_message = min(interval_message)
    if max_interval_message == min_interval_message:
        for i in range(len(interval_message)):
            interval_message[i] = 1
    else:
        increase_interval_message = 1 / (max_interval_message - min_interval_message)
        for i in range(len(interval_message)):
            interval_message[i] = increase_interval_message * (interval_message[i] - min_interval_message)
    
    # calculate the priority of message
    all_priority_message = 0
    for i in range(len(seed.M)):
        priority_message_single = ratio_potential_message * potential_message[i] + ratio_number_message * number_message[i] + ratio_interval_message * interval_message[i]
        priority_message.append(priority_message_single)
        all_priority_message += priority_message_single
    
    # calculate the probability of message
    for i in range(len(priority_message)):
        probability_message.append(priority_message[i] / all_priority_message)
        
    # choose a message according to probability
    message_index = random.choices(message_index_array, weights = probability_message, k = 1)[0]
    message = seed.M[message_index]
    
    # update the information of message
    queue[seed_index].M[message_index].number_used_message += 1
    queue[seed_index].M[message_index].interval_message = 0
    for i in range(len(seed.M)):
        if i != message_index:
            queue[seed_index].M[i].interval_message += 1
   
    # 聚类轮次的优先性选择
    ratio_cluster_interested = cfg.INTEREST_CLUSTER_RADIO
    ratio_cluster_number = cfg.NUMBER_CLUSTER_RADIO
    
    cluster_interested = []
    cluster_number = []
    priority_cluster = []
    probability_cluster = []
    cluster_index_array = []
    temp_array = []
    for i in range(len(message.snippet)):
        cluster_index_array.append(i)
        temp_array.append(1)
        
    # process the interested number of a cluster
    for i in range(len(message.snippet)):
        cluster_interested.append(message.snippet[i]["interested"])
    max_cluster_interested = max(cluster_interested)
    min_cluster_interested = min(cluster_interested)
    if max_cluster_interested == min_cluster_interested:
        for i in range(len(message.snippet)):
            cluster_interested[i] = 1
    else:
        cluster_interested_increase = 1 / (max_cluster_interested - min_cluster_interested)
        for i in range(len(message.snippet)):
            cluster_interested[i] = cluster_interested_increase * (cluster_interested[i] - min_cluster_interested)
    
    # process the number of times snippet has been used
    for i in range(len(message.snippet)):
        cluster_number.append(message.snippet[i]["number"])
    max_cluster_number = max(cluster_number)
    min_cluster_number = min(cluster_number)
    if max_cluster_number == min_cluster_number:
        for i in range(len(message.snippet)):
            cluster_number[i] = 1
    else:
        cluster_number_increase = 1 / (max_cluster_number - min_cluster_number)
        for i in range(len(message.snippet)):
            cluster_number[i] = 1 - cluster_number_increase * (cluster_number[i] - min_cluster_number)
    
    # calculate the priority of snippet
    all_priority_cluster = 0
    for i in range(len(message.snippet)):
        priority_cluster_single = ratio_cluster_interested * cluster_interested[i] + ratio_cluster_number * cluster_number[i]
        priority_cluster.append(priority_cluster_single)
        all_priority_cluster += priority_cluster_single
        
    # calculate the probability of message
    for i in range(len(message.snippet)):
        probability_cluster.append(priority_cluster[i] / all_priority_cluster)
    
    if (len(temp_array) == len(cluster_interested) and all(x == y for x, y in zip(temp_array, cluster_interested))) and (len(temp_array) == len(cluster_number) and all(x == y for x, y in zip(temp_array, cluster_number))):
        # setting the parameters of the binomial distribution
        n = len(message.snippet) - 1
        p = 0.5
        
        x = np.arange(0, n + 1)
        pmf = binom.pmf(x, n, p)
        
        cluster_round = random.choices(x, weights = pmf, k = 1)[0]
        snippets = message.snippet[cluster_round]["snippets"]
    else:
        cluster_round = random.choices(cluster_index_array, weights = probability_cluster, k = 1)[0]
        snippets = message.snippet[cluster_round]["snippets"]
        
    # update the information of cluster
    message.snippet[cluster_round]["number"] += 1
    
    # randomly selecting the number of mutation segments
    number_mutation_segment = discrete_exponential(0.35, len(snippets))[0]
    
    # 基于Shapley分析以及CMAB的优先性选择
    ratio_shapley = cfg.SHAPLEY_VALUE_RADIO
    ratio_number_snippet = cfg.NUMBER_SNIPPET_RADIO
    
    shapley_snippet = []
    number_snippet = []
    priority_snippet = []
    probability_snippet = []
    snippet_index_array = []
    for i in range(len(snippets)):
        snippet_index_array.append(i)
    
    # process the shapley value of a snippet
    for i in range(len(snippets)):
        shapley_snippet.append(snippets[i]["shapley"])
    max_shapley = max(shapley_snippet)
    min_shapley = min(shapley_snippet)
    if max_shapley == min_shapley:
        for i in range(len(shapley_snippet)):
            shapley_snippet[i] = 1
    else:
        shapley_increase = 1 / (max_shapley - min_shapley)
        for i in range(len(shapley_snippet)):
            shapley_snippet[i] = shapley_increase * (shapley_snippet[i] - min_shapley)
    
    # process the number of times snippet has been used
    for i in range(len(snippets)):
        number_snippet.append(snippets[i]["number"])
    max_number_snippet = max(number_snippet)
    min_number_snippet = min(number_snippet)
    if max_number_snippet == min_number_snippet:
        for i in range(len(number_snippet)):
            number_snippet[i] = 1
    else:
        number_increase_snippet = 1 / (max_number_snippet - min_number_snippet)
        for i in range(len(number_snippet)):
            number_snippet[i] = 1 - number_increase_snippet * (number_snippet[i] - min_number_snippet) 
        
    # calculate the priority of snippet
    all_priority_snippet = 0
    for i in range(len(snippets)):
        priority_snippet_single = ratio_shapley * shapley_snippet[i] + ratio_number_snippet * number_snippet[i]
        priority_snippet.append(priority_snippet_single)
        all_priority_snippet += priority_snippet_single
        
    # calculate the probability of message
    for i in range(len(snippets)):
        probability_snippet.append(priority_snippet[i] / all_priority_snippet)
    
    Regeneration = True

    while Regeneration == True:
        # Select the fragments that need to be mutated according to the weight
        mutation_index_array = []
        mutation_number = 0   
        while (1) :
            mutation_index = int(random.choices(snippet_index_array, weights = probability_snippet, k = 1)[0])
            while mutation_index in mutation_index_array:
                mutation_index = int(random.choices(snippet_index_array, weights = probability_snippet, k = 1)[0])
            mutation_index_array.append(mutation_index)
            mutation_number += 1
            if mutation_number == number_mutation_segment:
                break
        mutation_index_array = sorted(mutation_index_array)        
        
        # Random selection of mutation types
        mutation_types = []
        for _ in range(number_mutation_segment):
            mutation_types.append(random.randint(0, 6))
                
        history_combination_information = {}
        history_combination_information["mutation_index"] = mutation_index_array
        history_combination_information["mutation_type"] = mutation_types
        
        if history_combination_information in history_combination:
            Regeneration = True
        else:
            Regeneration = False


    # update the number information of snippet
    for mutation_index in mutation_index_array:
        queue[seed_index].M[message_index].snippet[cluster_round]["snippets"][mutation_index]["number"] += 1
    
    # performing the corresponding mutation process
    tempMessage = queue[seed_index].M[message_index].raw["Content"]
    length_message = len(tempMessage)
    for i in range(number_mutation_segment):
        queue[seed_index] = mutation_generation(queue[seed_index], message_index, snippets[mutation_index_array[i]]["fragment"], mutation_types[i], length_message)
    
    flag = True
    
    if responseHandle(queue[seed_index], m.SnippetMutationSend(queue[seed_index], message_index, path_score)) == False:
        
        # update the history combination
        history_combination_information = {}
        history_combination_information["mutation_index"] = mutation_index_array
        history_combination_information["mutation_type"] = mutation_types
        history_combination.append(history_combination_information)
        flag = False
        
        response = m.sendMessage(queue[seed_index].M[message_index])
        queue[seed_index].M[message_index].raw["Content"] = tempMessage
        # update the interested situation of seed
        queue[seed_index].number_interested += 1000
        # update the interested situation of message
        queue[seed_index].M[message_index].number_interested_message += 1000
        # update the interested situation of cluster
        queue[seed_index].M[message_index].snippet[cluster_round]["interested"] += 1000
        # update the shapley value of snippet
        logger.info(f"{Fore.BLUE}Start to update the shapley value!{Fore.RESET}")
        update_shapley_snippet(queue[seed_index], message_index, snippets, mutation_index_array, mutation_types, response, cluster_round, restoreSeed)
    
    queue[seed_index].M[message_index].raw["Content"] = tempMessage
    logger.info(f"{Fore.BLUE}There is no interseted in this advance mutation process!{Fore.RESET}")
        
    return flag


def snipleyfuzz():
    global queue, restoreSeed, outputfold, number_array, round, device_type, device_name
    init()
    # 通过参数指定不同设备与文件路径，便于在 Yeelight、小米等多种设备之间复用统一的主程序。
    parser = argparse.ArgumentParser(
        description=("IoT Fuzz main program: specify paths for restorefile, inputfold, "
                     "outputfold, probe_fold, recordfile, device_type, and device_name. "
                     "Use -h/--help to see all options."
        ),
        add_help=True
    )
    parser.add_argument(
        "--restorefile",
        type=str,
        required=True,
        help="Path to the restore seed file."
    )
    parser.add_argument(
        "--outputfold",
        type=str,
        required=True,
        help="Directory for storing crash samples."
    )
    parser.add_argument(
        "--inputfold",
        type=str,
        required=True,
        help="Directory of initial seed inputs."
    )
    parser.add_argument(
        "--devicetype",
        type=str,
        required=True,
        help="Device type (e.g., 'yeelight', 'xiaomi'); passed to Messenger for low-level communication."
    )
    parser.add_argument(
        "--devicename",
        type=str,
        required=True,
        help="Device name (e.g., 'YLDP05YL', 'YLDP13YL'); passed to construct the output file name."
    )
    # 互斥组：--recordfile 和 --probefold 二选一，且不能同时出现
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--recordfile",
        type=str,
        help="Probe record file path; if exists, Probe phase will be skipped."
    )
    group.add_argument(
        "--probefold",
        type=str,
        help="Directory to store Probe-phase record files."
    )
    args = parser.parse_args()
    # 首先进行参数的检查
    if not os.path.isfile(args.restorefile):
        parser.error(f"[Error] --restorefile '{args.restorefile}' does not exist or is not a regular file")
    if not os.path.isdir(args.inputfold):
        parser.error(f"[Error] --inputfold '{args.inputfold}' does not exist or is not a directory")
    output_dir = Path(args.outputfold)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    elif not output_dir.is_dir():
        parser.error(f"[Error] --outputfold '{args.outputfold}' already exists but is not a directory")
    if args.recordfile is not None:
        if not os.path.exists(args.recordfile):
            parser.error(f"[Error] --recordfile '{args.recordfile}' does not exist")
        record_path = Path(args.recordfile)
        # It is invalid if the given path is an existing directory
        if not record_path.is_file():
            parser.error(f"[Error] --recordfile '{args.recordfile}' is not a regular file")
    if args.probefold is not None:
        probe_dir = Path(args.probefold)
        if not probe_dir.exists():
            probe_dir.mkdir(parents=True, exist_ok=True)
        elif not probe_dir.is_dir():
            parser.error(f"[Error] --probefold '{args.probefold}' already exists but is not a directory")
    # 进行参数传递
    restorefile = args.restorefile
    outputfold = args.outputfold
    recordfile = args.recordfile
    inputfold = args.inputfold
    probefold = args.probefold
    device_type = args.devicetype
    device_name = args.devicename
    # 进行模糊测试
    restoreSeed = readInputFile(restorefile)
    logger.info(f"{Fore.BLUE}Successful read from the restorefile!{Fore.RESET}")
    queue = readInputFold(inputfold)
    if recordfile:
        logger.info(f"{Fore.BLUE}ProbeRecord file exists and Probe process has been ignored!{Fore.RESET}")
        queue = readRecordFile(recordfile)
        for seed in queue:
            update_path_score(seed)
        thread = threading.Thread(target = info)
        thread.start()
        for seed in queue:
            seed.display()
        if (dryRun(queue)):  
            logger.error('#### Dry run failed, check the inputs or connection.')
            sys.exit()
    else:
        logger.info(f"{Fore.BLUE}Start to exec Probe process!{Fore.RESET}")
        queue = readInputFold(inputfold)
        if (dryRun(queue)):  
            logger.error('#### Dry run failed, check the inputs or connection.')
            sys.exit()
        for i in range(len(queue)):
            queue[i].display()
            logger.info(f"{Fore.BLUE}Start to exec Probe process for seed{i}! {i + 1}/{len(queue)}{Fore.RESET}")
            queue[i] = Probe(queue[i])
        # update the information
        thread = threading.Thread(target = info)
        thread.start()
        writeRecord(queue, probefold)
    # 开始进行变异测试
    skip = False
    number = 0
    while (1):
        if not skip:
            i = 0
            while i < len(queue):
                if not queue[i].isMutated:
                    queue[i].number_used += 1
                    queue[i].interval = 0
                    for j in range(len(queue)):
                        if j != i:
                            queue[j].interval += 1
                    logger.info(f"{Fore.BLUE}Start to exec SnippetMutate process for seed{i}! {i+1}/{len(queue)}{Fore.RESET}")
                    SnippetMutate(queue[i], restoreSeed)
                    queue[i].isMutated = True
                i = i + 1
        skip = True
        number += 1
        logger.info(f"{Fore.BLUE}Start to exec advanced_mutate process! the {round}th round({number}) {Fore.RESET}")
        skip = advanced_mutate(queue, restoreSeed)
        if skip == False:
            round += 1
            number_array.append(number)
            number = 0
            
if __name__ == '__main__':
    snipleyfuzz()
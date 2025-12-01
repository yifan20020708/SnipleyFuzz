# -*- coding: utf-8 -*-
import cfg
from typing import List


# Define Snippet and ClusterRound classes to hold context info
class Snippet:
    def __init__(self, message, start_index, end_index, data_bytes):
        self.message = message               # parent Message reference
        self.start = start_index
        self.end = end_index
        self.original_data = data_bytes      # original bytes of this snippet
        # Initialize snippet context statistics
        self.shapley = 0.0
        self.last_update_round = -1          # last round Shapley updated (optional)
        self.total_contrib_count = 0         # count of Shapley contribution updates
        self.frequency = 0                  # freq_{j,t}: times snippet selected
        self.recency = 0                    # recency_{j,t}: rounds since snippet last selected
        self.streak = 0                     # streak_{j,t}: consecutive rounds since snippet last yielded utility
        self.recent_rewards = [0] * cfg.REWARD_WINDOW_H  # reward history of last h rounds (for short-term reward)
        self.recent_reward = 0.0             # R_{j,t,h}: weighted recent reward aggregate


class ClusterRound:
    snippets: List[Snippet]
    def __init__(self, level):
        self.level = level                  # clustering round index (0 = finest segmentation)
        self.snippets = []                  # list of Snippet objects for this cluster level
        self.interested = 0                 # NRS_cum(C_k): cumulative novel responses found at this level
        self.number_used = 0                # times this clustering level selected
        self.last_used_round = -1           # last round this level was selected
        

class Message:
    headers = []  # Header List
    raw = {}  # Header and corresponding content
    clusters: List[ClusterRound]

    def __init__(self) -> None:
        self.headers = []
        self.raw = {}
        self.response_div = 0.0      # Response diversity for this message (distinct response clusters count)
        self.nrs_cum = 0            # Cumulative NRS count triggered by this message
        self.number_used = 0        # Times this message has been selected for mutation
        self.interval = 0           # Rounds since this message was last selected
        self.redundancy_score = 0.0 # (Optional) similarity redundancy score for this message
        # prepare structure for cluster rounds (to be populated in advanced mutation)
        self.clusters = []          # List of ClusterRound objects for this message (initialized later)

    def append(self, line) -> None:
        if ":" in line:
            sp = line.split(":")
            if sp[0] in self.headers:
                print("Error. Message headers '", sp[0], "' is duplicated.")
            else:
                self.headers.append(sp[0])
                self.raw[sp[0]] = line[(line.index(':') + 1):]

class Seed:
    M: List[Message]
    R = [] # Response List
    PR = [] # Probe message response pool 
    PS = [] # Probe message response self-similarity scores (the set)
    PI = [] # the index of response for every probe message
    isMutated = False
    number_used = 0
    interval = 1
    number_interested = 0
    ClusterList = []  # the final cluster result
    Snippet = [] # the final message snippet
    
    def __init__(self) -> None:
        self.M = []
        self.R = []
        self.PR = []
        self.PS = []
        self.PI = []
        self.isMutated = False
        self.ClusterList = []
        self.Snippet = []
        self.response_div = 0.0      # Response diversity metric (distinct response clusters count)
        self.nrs_cum = 0            # Cumulative NRS count (novel responses triggered by this seed)
        self.number_used = 0        # Times this seed has been selected
        self.interval = 0           # Rounds since this seed was last selected
        self.redundancy_score = 0.0 # (Optional) similarity redundancy score for this seed (for ρ(M) in formula (3))
        
    def append(self, message):
        self.M.append(message)
        
    def response(self, response):
        self.R.append(response)
        
    def display(self):
        print("**** Seed Information ****")
        print("Is Mutated: ", self.isMutated)
        print("Number of Messages: ", len(self.M))
        print("**** Message Information ****")
        for i in range(0, len(self.M)):
            print("=== Message index: ", i + 1)
            for header in self.M[i].headers:
                print(header, ":", self.M[i].raw[header])
            print('Response : ' + self.R[i])
            if self.PR and self.PS and self.PI:
                print('Probe Result:')
                print('PI')
                print(self.PI[i])
                print('PR and PS')
                for n in range(len(self.PR[i])):
                    print("(" + str(n) + ") " + self.PR[i][n])
                    print(self.PS[i][n])
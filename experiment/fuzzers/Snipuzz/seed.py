class Seed:
    M = []  # Message List - message type
    R = []  # Response List - string

    PR = []  # Probe message response pool - 2d list
    PS = []  # Probe message response similarity scores - 2d list
    PI = []

    isMutated = False

    ClusterList = []

    Snippet = []


    def __init__(self) -> None:
        self.M = []
        self.R = []
        self.PR = []
        self.PS = []
        self.PI = []
        self.isMutated = False
        self.ClusterList = []
        self.Snippet = []



    def append(self, message):
        self.M.append(message)

    def response(self, response):
        self.R.append(response)

    def display(self):
        for i in range(0, len(self.M)):
            print("Message index: ", i + 1)
            for header in self.M[i].headers:
                print(header, " : ", self.M[i].raw[header])
            print('Response:')
            print(self.R[i])
            if self.PR and self.PS and self.PI:
                print('Probe Result:')
                print('PI')
                print(self.PI[i])
                print('PR and PS')
                for n in range(len(self.PR[i])):
                    print("(" + str(n) + ") " + self.PR[i][n])
                    print(self.PS[i][n])


class Message:
    headers = []  # Header List
    raw = {}  # Header and corresponding content

    def __init__(self) -> None:
        self.headers = []
        self.raw = {}

    def append(self, line) -> None:
        if ":" in line:
            sp = line.split(":")
            if sp[0] in self.headers:
                print("Error. Message headers '", sp[0], "' is duplicated.")
            else:
                self.headers.append(sp[0])
                self.raw[sp[0]] = line[(line.index(':') + 1):]
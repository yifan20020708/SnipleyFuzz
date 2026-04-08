# -*- coding: utf-8 -*-
import time
import socket
import ast
import json
import sys
import signal
import logging
from pathlib import Path
from colorama import init, Fore

# 构建相对路径指向 simple-mi-home 目录
current_dir = Path(__file__).parent
mi_home_dir = current_dir.parent.parent.parent / "device" / "mihome" / "simple-mi-home"
sys.path.append(str(mi_home_dir))

try:
    from MiApi.service import MiService
except Exception:  # 环境引入失败
    MiService = None
    
YEELIGHT_TIMEOUT_TIMES = 2
YEELIGHT_MAX_RETRY = 3
XIAOMI_TIMEOUT_TIMES = 5

logger = logging.getLogger(__name__)

# 超时处理（小米接口使用）
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Timeout occurred")

# 字符串编辑距离与相似度
def EditDistanceRecursive(str1, str2):
    edit = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            edit[i][j] = min(
                edit[i - 1][j] + 1,     
                edit[i][j - 1] + 1,     
                edit[i - 1][j - 1] + d   
            )
    return edit[len(str1)][len(str2)]

# 根据编辑距离计算相似度分数（0~100）
def SimilarityScore(str1, str2):
    ED = EditDistanceRecursive(str1, str2)
    return round((1 - (ED / max(len(str1), len(str2)))) * 100, 2)


# 消息发送器（device_type = "yeelight" or "xiaomi"）
class Messenger:
    SocketSender = socket.socket()
    restore = []  # restoring message sequence (保持初始状态的恢复序列)

    def __init__(self, restoreSeed, device_type="yeelight"):
        self.SocketSender = None
        self.restore = restoreSeed
        self.device_type = device_type.lower().strip()

    # 公共序列发送接口
    def DryRunSend(self, sequence):
        for message in sequence.M:
            response = self.sendMessage(message)
            if response == "#error":
                return True
            sequence.R.append(response)
        # 发送恢复序列，保证设备状态回滚
        for message in self.restore.M:
            response = self.sendMessage(message)
            if response == "#error":
                return True
        return sequence

    def ProbeSend(self, sequence, index):
        for i in range(len(sequence.M)):
            response = self.sendMessage(sequence.M[i])
            if response == "#error":
                return "#error"
            elif response == '#crash':
                return '#crash'
            if i == index:
                res = response
        # 发送恢复序列，保持设备状态稳定
        for i in range(len(self.restore.M)):
            resotreResponse = self.sendMessage(self.restore.M[i])
            if resotreResponse == "#error":
                return "#error"
            elif response == '#crash':
                return '#crash'
        return res

    def SnippetMutationSend(self, sequence, index, path_score):
        for i in range(len(sequence.M)):
            response = self.sendMessage(sequence.M[i])
            if response == "#error":
                return "#error"
            elif response == '#crash':
                return '#crash'
            if i == index:
                res = response
                content = sequence.M[i].raw["Content"].strip()
                logger.info(f"{Fore.BLUE}[Message Content]{content}{Fore.RESET}")
                logger.info(res)
        for i in range(len(self.restore.M)):
            restoreResponse = self.sendMessage(self.restore.M[i])
            if restoreResponse == "#error":
                return "#error"
            elif response == '#crash':
                return '#crash'
        pool = []
        scores = []
        for j in range(len(path_score)):
            pool.append(path_score[j]["response"])
            scores.append(path_score[j]["score"])
        for i in range(len(pool)):
            c = SimilarityScore(pool[i].strip(), res.strip())
            if c >= scores[i]:
                return ""
        return "#interesting-"+str(index)

    # 核心发送接口
    def sendMessage(self, message, timeout_time=0, retrytime=0):
        if self.device_type == "yeelight":
            return self._send_yeelight(message, timeout_time, retrytime)
        elif self.device_type == "xiaomi":
            if MiService is None:
                raise RuntimeError("MiService module not found. Please check the environment setup.")
            return self._send_xiaomi(message, timeout_time)
        else:
            logger.error(f"{Fore.RED}[ERROR] Unsupported device_type: {self.device_type}{Fore.RESET}")
            return "#error"

    # Yeelight 设备实现
    def _send_yeelight(self, message, timeout_time=0, retrytime=0):
        if "IP" in message.headers and "Port" in message.headers:
            ip = message.raw["IP"].strip()
            port = int(message.raw["Port"])
            content = message.raw["Content"] + "\r\n"
            notification_string = '"method":"props"'
            response = ''
            timeout = 5
            max_retries = YEELIGHT_MAX_RETRY
            localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
            init()

            try:
                # 建立 TCP 连接
                self.SocketSender = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.SocketSender.connect((ip, port))
                self.SocketSender.send(content.encode('utf8'))
                self.SocketSender.settimeout(timeout)
                # 接收响应
                response = self.SocketSender.recv(1024).decode('utf8')
                if notification_string in response:
                    # 过滤掉通知类数据，再收一次真正的响应
                    response = self.SocketSender.recv(1024).decode('utf8')
                logger.info(f"{Fore.GREEN}[+]{localtime}:Successful receive response from yeelight!{Fore.RESET}")
                self.SocketSender.close()
            # 超时重试逻辑
            except socket.timeout:
                logger.error(f"{Fore.RED}[ERROR]{localtime}:Time out during the tcp process! Retrying ({timeout_time + 1}/{3}).{Fore.RESET}")
                if timeout_time < YEELIGHT_TIMEOUT_TIMES:
                    self.SocketSender.close()
                    time.sleep(0.5)
                    response = self.sendMessage(message, timeout_time + 1, retrytime)
                else:
                    return "#crash"
            # 其他异常重试
            except Exception as e:
                if retrytime < max_retries:
                    logger.error(f"{Fore.RED}[ERROR]{localtime}:{str(e)}! Retrying ({retrytime + 1}/{max_retries}).{Fore.RESET}")
                    self.SocketSender.close()
                    time.sleep(0.5)
                    response = self.sendMessage(message, timeout_time, retrytime + 1)
                else:
                    return "#crash"

            return response
        else:
            logger.error("Error : IP and Port of target should be included in input files")
            return "#error"

    # 小米设备实现
    def _send_xiaomi(self, message, timeout_time=0):
        if "did" in message.headers and "uri" in message.headers:
            did = '"' + message.raw["did"].strip() + '"'
            uri =  message.raw["uri"].strip()
            content =  message.raw["content"].strip()
            response = ''
            command = {"uri": uri, "content": content}
            localtime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
            timeout = 7
            init()

            try:
                mi = MiService()
                device = mi.use_device(did)
                bind_id = '"did": ' + did 
                if bind_id not in content:
                    response = "Fail to bind device!" + "\r\n"
                else:
                    # 使用 signal 做超时控制
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(timeout)
                        response = device.send(command)
                    except TimeoutError:
                        if timeout_time < XIAOMI_TIMEOUT_TIMES:
                            logger.error(f"{Fore.RED}[ERROR]{localtime}:Timeout occurred! Retrying ({timeout_time + 1}/{5}).{Fore.RESET}")
                            signal.alarm(0)
                            response = self.sendMessage(message, timeout_time + 1)
                        else:
                            signal.alarm(0)
                            logger.error("Crash!")
                            return "#crash"
                    # 关闭闹钟
                    signal.alarm(0)
                    if "updateTime" in response:
                        data = ast.literal_eval(response)
                        for item in data['result']:
                            item.pop('updateTime', None)
                        response = json.dumps(data, ensure_ascii = False) + "\r\n"
                        
                logger.info(f"{Fore.GREEN}[+]{localtime}:Successful receive response from xiaomi!{Fore.RESET}")

            # 增强鲁棒性的异常处理
            except Exception as e:
                if timeout_time < XIAOMI_TIMEOUT_TIMES:
                    logger.error(f"{Fore.RED}[ERROR]{localtime}:{str(e)}! Retrying ({timeout_time + 1}/{5}).{Fore.RESET}")
                    time.sleep(0.5)
                    response = self.sendMessage(message, timeout_time + 1)
                else:
                    logger.error("Crash!")
                    return "#crash"

            return response
        else:
            logger.error("Error : device_id (did) and uri of target should be included in input files")
            return "#error"
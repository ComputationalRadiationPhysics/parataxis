import os

COLOR_NAMES = ['grey', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']
COLOR_CODES = ['\033[' + str(i) + 'm' for i in range(30, 38)]
COLORS = dict(list(zip(COLOR_NAMES, COLOR_CODES)))

RESET_COLOR = '\033[0m'

def cprint(text, color = None):
    """Print the text with the optional color"""
    
    if color != None and os.getenv('ANSI_COLORS_DISABLED') is None:
        text = COLORS[color] + text + RESET_COLOR
    print(text)

def thumbsUp():

    cprint("""
...........,_
........../.(|
..........\..\
........___\..\,. ~~~~~~
.......(__)_)...\
......(__)__)|...|
......(__)__)|.__|
.......(__)__)___/~~~~~~""", "green")

def thumbsDown():

    print("""
........_________
.......(__)__).__\~~~~~~
......(__)__)|...|
......(__)__)|...|
.......(__)_)..,/
.........../../.. ~~~~~~
........../../
..........\_(|""", "red")


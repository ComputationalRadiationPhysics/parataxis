# Copyright 2015-2016 Alexander Grund, Axel Hübl
#
# This file is part of ParaTAXIS.
#
# ParaTAXIS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ParaTAXIS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ParaTAXIS.  If not, see <http://www.gnu.org/licenses/>.

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
..........\..\\
........___\..\,. ~~~~~~
.......(__)_)...\\
......(__)__)|...|
......(__)__)|.__|
.......(__)__)___/~~~~~~""", "green")

def thumbsDown():

    cprint("""
........_________
.......(__)__).__\~~~~~~
......(__)__)|...|
......(__)__)|...|
.......(__)_)..,/
.........../../.. ~~~~~~
........../../
..........\_(|""", "red")

if __name__ == '__main__':
    for c in COLOR_NAMES:
        cprint(c, c)
    thumbsUp()
    thumbsDown()


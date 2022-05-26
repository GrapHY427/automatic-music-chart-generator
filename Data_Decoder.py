import numpy as np
import json
from pydub import AudioSegment
from aubio import source, onset, pitch
import random as rd
import bpm_get

#####################################################################
name = "溢彩华庭"  # 音乐名，请保持音乐和谱面名称相同，音乐请使用mp3格式
hold_con = "auto"  # 推荐值:1；越大，生成的hold越多；若赋值为"auto"，将根据音乐智能决定该值大小
chain_con = "auto"  # 推荐值:0.1；越大，生成的chain越多；若赋值为"auto"，将根据音乐智能决定该值大小
bpm = 120   # 113.41  # 若赋值为"auto":将利用内置函数自动测量bpm，否则请在此手动输入bpm
#####################################################################
song_name = name + ".mp3"
if bpm == "default":
    bpm = round(bpm_get.get_file_bpm(song_name), 2)
song = AudioSegment.from_mp3(song_name)
song_time = song.duration_seconds
tick_total = song_time * bpm * 8
print(bpm)  # cylheim初始化谱面时需填写该信息

win_s = 512  # fft size
hop_s = win_s // 2  # hop size

filename = name + ".mp3"

samplerate = 0
s = source(filename, samplerate, hop_s)
samplerate = s.samplerate

o = onset("default", win_s, hop_s, samplerate)

# list of onsets, in samples
my_onsets = []

# total number of frames read
total_frames = 0
while True:
    samples, read = s()
    if o(samples):
        my_onsets.append(float(o.get_last_s()))
    total_frames += read
    if read < hop_s:
        break
my_onsets.append(float("inf"))
my_onsets = [0] + my_onsets
print(my_onsets)


def get_onsets_chains(chain_con, my_onsets):
    onsets = []
    chains = []
    for i in range(1, len(my_onsets) - 1):
        if my_onsets[i] - my_onsets[i - 1] < chain_con or my_onsets[i + 1] - my_onsets[i] < chain_con:
            chains.append(my_onsets[i])
        else:
            onsets.append(my_onsets[i])
        now = o
    return onsets, chains


filename = name + ".mp3"

downsample = 1
samplerate = 44100 // downsample

win_s = 4096 // downsample  # fft size
hop_s = 512 // downsample  # hop size

s = source(filename, samplerate, hop_s)
samplerate = s.samplerate

tolerance = 0.8

pitch_o = pitch("yin", win_s, hop_s, samplerate)
pitch_o.set_unit("midi")
pitch_o.set_tolerance(tolerance)

# total number of frames read
pitches = []
frames = []
total_frames = 0
while True:
    samples, read = s()
    my_pitch = pitch_o(samples)[0]
    pitches.append(my_pitch)
    frames.append(total_frames)
    total_frames += read
    if read < hop_s:
        break


def get_starts_ends(hold_con):
    starts = []
    ends = []
    start = 0
    end = 0
    count = 0
    for i in range(1, len(pitches)):
        if pitches[i] != 0 and abs(pitches[i] - pitches[i - 1]) <= hold_con:
            count += 1
            end = frames[i] / float(samplerate)
        else:
            if count >= 30:
                starts.append(start)
                ends.append(end)
            start = frames[i] / float(samplerate)
            count = 1
    return starts, ends


hold_std = 0.5   # 0.06733389961960336
chain_std = 0.2  # 0.2198323300217897 * 2
if chain_con == "auto":
    chain_con = 0.1
    delta = None
    while True:
        onsets, chains = get_onsets_chains(chain_con, my_onsets)
        ratio = len(chains) / (len(onsets) + len(chains))
        if delta == None:
            delta = 0.005 if ratio < chain_std else -0.005
        elif delta * (ratio - chain_std) >= 0:
            [onsets, chains] = [onsets, chains] if abs(ratio - chain_std) < abs(pre_ratio - chain_std) else [pre_onsets,
                                                                                                             pre_chains]
            break
        pre_onsets = onsets
        pre_chains = chains
        pre_ratio = ratio
        chain_con += delta
else:
    onsets, chains = get_onsets_chains(chain_con)

if hold_con == "auto":
    hold_con = 1
    delta = None
    while True:
        starts, ends = get_starts_ends(hold_con)
        ratio = len(starts) / (len(onsets) + len(chains))
        if delta == None:
            delta = 0.05 if ratio < hold_std else -0.05
        elif delta * (ratio - hold_std) >= 0:
            [starts, ends] = [starts, ends] if abs(ratio - hold_std) < abs(pre_ratio - hold_std) else [pre_starts,
                                                                                                       pre_ends]
            break
        pre_starts = starts
        pre_ends = ends
        pre_ratio = ratio
        hold_con += delta
else:
    starts, ends = get_starts_ends(hold_con)

i, j = 0, 0
while True:
    try:
        if starts[j] < onsets[i]:
            j += 1
        else:
            if starts[j] - onsets[i] <= 0.2:
                onsets.pop(i)
            else:
                i += 1
    except IndexError:
        break

print("click: ", len(onsets))
print("hold: ", len(starts))
print("chain: ", len(chains))

pre_x = float("inf")


def produce_x(p):
    x = rd.random()
    if p % 2 == 0:
        while not (0.1 <= x <= 0.2 or 0.55 <= x <= 0.7):
            x = rd.random()
    else:
        while not (0.3 <= x <= 0.45 or 0.8 <= x <= 0.9):
            x = rd.random()
    return x


def produce_n(t, c, my_id, hold_tick=0):
    tick = round(c * tick_total / song_time)
    page_index = int(tick / 960)
    n = {'hold_tick': hold_tick * tick_total / song_time,
         'type': t,
         'next_id': -1,
         'x': produce_x(page_index),
         'tick': tick,
         'page_index': page_index,
         'id': my_id,
         'has_sibling': False,
         'is_forward': False}
    return n


def produce_0(n):
    return n


def produce_1(n):
    global pre_x
    if n['page_index'] != int((n['tick'] + n['hold_tick']) / 960):
        n['type'] = 2
    p = n["page_index"]
    while abs(n['x'] - pre_x) < 0.1:
        n['x'] = produce_x(p)
    pre_x = n['x']
    return n


def produce_3(n):
    return n


print(starts)
note_name = name + ".json"
note = json.loads(open(note_name, encoding="utf-8").read())
note_list = []
count = 0
while starts != [] or onsets != [] or chains != []:
    try:
        start = starts[0]
    except IndexError:
        start = float("inf")
    try:
        onset = onsets[0]
    except IndexError:
        onset = float("inf")
    try:
        chain = chains[0]
    except IndexError:
        chain = float("inf")
    min_v = min(start, onset, chain)
    if onset == min_v:
        onsets.pop(0)
        n = produce_n(0, onset, count)
        if n['page_index'] == 0:
            continue
        n = produce_0(n)
    elif start == min_v:
        starts.pop(0)
        n = produce_n(1, start, count, hold_tick=ends.pop(0) - start)
        if n['page_index'] == 0:
            continue
        n = produce_1(n)
    else:
        chains.pop(0)
        n = produce_n(3, chain, count)
        if n['page_index'] == 0:
            continue
        n = produce_3(n)
        if count != 0 and (note_list[-1]['type'] == 3 or note_list[-1]['type'] == 4) and n['tick'] - note_list[-1][
            'tick'] < 0.5 * tick_total / song_time:
            note_list[-1]['next_id'] = count
            n['type'] = 4
    note_list.append(n)
    count += 1

if note_list[0]["type"] == 3 and note_list[1]["type"] != 4:
    note_list[0]["type"] = 0
for i in range(1, len(note_list) - 1):
    if note_list[i]["type"] == 3 and note_list[i + 1]["type"] != 4:
        j = 1
        while note_list[i + j]["type"] == 1 or note_list[i + j]["type"] == 2:
            j += 1
        note_list[i + j]["type"] = 4
        note_list[i]["next_id"] = note_list[i + j]["id"]
end = None
now_x = None
i = 0
while i < len(note_list):
    if 1 <= note_list[i]["type"] <= 2:
        end = note_list[i]["tick"] + note_list[i]["hold_tick"]
        now_x = note_list[i]["x"]
        if note_list[i - 1]["type"] == 3:
            while (note_list[i - 1]['x'] < 0.5) == (note_list[i]['x'] < 0.5):
                p = note_list[i - 1]['page_index']
                note_list[i - 1]['x'] = produce_x(p)
    else:
        if end == None:
            i += 1
            continue
        while (note_list[i]['x'] < 0.5) == (now_x < 0.5):
            p = note_list[i]['page_index']
            note_list[i]['x'] = produce_x(p)
        if note_list[i]["tick"] >= end:
            end = None
            i -= 1
    i += 1
for i in range(1, len(note_list)):
    if note_list[i]["next_id"] != -1:
        next_id = note_list[i]["next_id"]
        if note_list[next_id]["page_index"] != note_list[i]["page_index"]:
            try:
                tag = (note_list[i]["x"] - note_list[i - 1]["x"]) / abs(note_list[i]["x"] - note_list[i - 1]["x"])
            except ZeroDivisionError:
                tag = 1
            note_list[next_id]["x"] = abs(max(0.05, min((rd.random() / 10 + 0.05) * tag + note_list[i]["x"], 0.95)))
        else:
            note_list[next_id]["x"] = max(0.05, min((rd.random() - 0.5) / 5 + note_list[i]["x"], 0.95))
note['note_list'] = note_list
json.dump(note, open(note_name, "w", encoding="utf-8"))
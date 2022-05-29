import json
import torch
from pydub import AudioSegment
from aubio import source, tempo
from numpy import median, diff
import torchaudio
import torchvision.transforms as transforms


"""
    Macro
"""
MAX_NOTE = 1800
MAX_DURATION = 5  # unit: minute


# #####################################################################
# name = "周深 - 亲爱的旅人啊"  # 音乐名，请保持音乐和谱面名称相同，音乐请使用mp3格式
# hold_con = "auto"  # 推荐值:1；越大，生成的hold越多；若赋值为"auto"，将根据音乐智能决定该值大小
# chain_con = "auto"  # 推荐值:0.1；越大，生成的chain越多；若赋值为"auto"，将根据音乐智能决定该值大小
# bpm = 113.41  # 若赋值为"auto":将利用内置函数自动测量bpm，否则请在此手动输入bpm
# #####################################################################


class ChartParser:
    """  Parsing existing charts and transform the data to tensor for training
    """

    def __init__(self, filename: str):
        self.filename = filename + '.json'
        with open(self.filename, "r", encoding="utf-8") as f:
            chart_dict: dict = json.load(f)

        self.page_list = chart_dict["page_list"]
        self.tempo_list = chart_dict["tempo_list"]
        self.event_order_list = chart_dict["event_order_list"]
        self.note_list = chart_dict["note_list"]
        self.temp_list = []

    def create_note_list_tensor(self):
        """
            The format of the tensor:
                0: '1.0' if there exist a Note, else '0.0'
                1 - 9 : The map of 'Note' class's information
        """

        total_note = 0
        for each in self.note_list:
            first_dimension = [1,
                               each['page_index'],
                               each['type'],
                               each['id'],
                               each['tick'],
                               each['x'],
                               1 if each['has_sibling'] else 0,
                               each['hold_tick'],
                               each['next_id'],
                               1 if each['is_forward'] else 0]
            self.temp_list.append(first_dimension)
            total_note += 1
        page_list_tensor = torch.Tensor(self.temp_list)
        page_list_tensor = torch.cat([page_list_tensor, torch.Tensor(MAX_NOTE - total_note, 10)])
        return page_list_tensor


class ChartGenerator:
    """ Receiving the tensor from the neural network,
        generating the chart of the music with Cytus2 format
    """

    def __init__(self, music_name: str, format_version: int = 0,
                 time_base: int = 480, start_offset_time: float = 0, page_tick: int = 960):
        self.music_path = music_name + '.wav'
        self.filename = music_name + '.json'

        song = AudioSegment.from_wav(self.music_path)
        self.duration_time = song.duration_seconds
        self.bpm = self.get_file_bpm()
        self.tick_total = round(self.duration_time * self.bpm * time_base / 60)

        self.chart_head = {"format_version": format_version,
                           "time_base": time_base,
                           "start_offset_time": start_offset_time}
        self.page_tick = page_tick
        self.page_list = []
        self.tempo_list = [{"tick": 0,
                            "value": 60000000 / self.bpm}]
        self.event_order_list = []
        self.note_list = []

    def get_file_bpm(self):
        """
        Calculate the beats per minute (bpm) of a given file.
        """

        # default:
        sample_rate, win_s, hop_s = 44100, 1024, 512
        # manual settings

        s = source(self.music_path, sample_rate, hop_s)
        sample_rate = s.samplerate
        o = tempo("specdiff", win_s, hop_s, sample_rate)
        # List of beats, in samples
        beats = []
        # Total number of frames read
        total_frames = 0

        while True:
            samples, read = s()
            is_beat = o(samples)
            if is_beat:
                this_beat = o.get_last_s()
                beats.append(this_beat)
                # if o.get_confidence() > .2 and len(beats) > 2.:
                #    break
            total_frames += read
            if read < hop_s:
                break

        def beats_to_bpm(beat_times, path):
            """
            If enough beats are found, convert to periods then to bpm
            """
            if len(beat_times) > 1:
                if len(beat_times) < 4:
                    print("few beats found in {:s}".format(path))
                bpm = 60. / diff(beat_times)
                return median(bpm)
            else:
                print("not enough beats found in {:s}".format(path))
                return 0

        return beats_to_bpm(beats, self.music_path)

    def generate_page_list(self):
        """
        Fill the page_list with formatted dictionary
        """
        start_tick = 0
        end_tick = self.page_tick
        scan_line_direction = 1
        while start_tick < (self.tick_total + self.page_tick):
            page = {"start_tick": start_tick,
                    "end_tick": end_tick,
                    "scan_line_direction": scan_line_direction}
            start_tick = end_tick
            end_tick += self.page_tick
            scan_line_direction *= -1
            self.page_list.append(page)

    def generate_event_order_list(self):
        pass

    def generate_note_list(self):
        pass

    def generate_chart(self):
        with open(self.filename, 'w+', encoding="utf-8") as f:
            json.dump(self.chart_head, f)
            json.dump({"page_list": self.page_list}, f)
            json.dump({"tempo_list": self.tempo_list}, f)
            json.dump({"event_order_list": self.event_order_list}, f)
            json.dump({"note_list": self.note_list}, f)


class Note:
    """  Class that used to let the 'Note' data struct
         clearer and easier to understand
    """

    def __init__(self, page_index: int, chart_type: int, chart_id: int, tick: int, x: float,
                 has_sibling: bool, hold_tick: int, next_id: int, is_forward: bool):
        self.dictionary = {
            'page_index': page_index,
            'type': chart_type,
            'id': chart_id,
            'tick': tick,
            'x': x,
            'has_sibling': has_sibling,
            'hold_tick': hold_tick,
            'next_id': next_id,
            'is_forward': is_forward}


class EventOrder:
    """
        Class that used to let the "Event order" data struct
        clearer and easier to understand
    """

    def __init__(self, tick: int, event_type: int, args: int):
        arg_list = ['W', 'R', 'G']
        self.dictionary = {
            "tick": tick,
            "event_list": [
                {
                    "type": event_type,
                    "args": arg_list[args]
                }
            ]}


class DatasetGenerator:

    def __init__(self, path):
        self.path = path + '.wav'
        self.chart_generator = ChartGenerator(path)
        self.spectrogram = self.spectrogram_transformer()

    def spectrogram_transformer(self):
        waveform, sample_rate = torchaudio.load(self.path)
        spectrogram = torchaudio.transforms.Spectrogram()(waveform)
        spectrogram = spectrogram.permute(0, 2, 1)
        spectrogram = spectrogram[0]
        transformer1 = transforms.ToPILImage()
        transformer2 = transforms.Resize((round(self.chart_generator.duration_time * 20), 200))
        transformer3 = transforms.ToTensor()
        spectrogram = transformer1(spectrogram)
        spectrogram = transformer2(spectrogram)
        spectrogram = transformer3(spectrogram)
        spectrogram = torch.squeeze(spectrogram, 0)
        spectrogram = torch.cat([spectrogram, torch.Tensor(1200 * MAX_DURATION -
                                                           round(self.chart_generator.duration_time * 20), 200)])
        return spectrogram

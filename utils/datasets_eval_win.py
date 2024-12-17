import json
import os
from typing import Dict, Any, Union, Tuple, Optional

import torch
import numpy as np
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from utils.audio import load_audio_file, slice_padded_array
from utils.tokenizer import EventTokenizerBase, NoteEventTokenizer
from utils.note2event import slice_multiple_note_events_and_ties_to_bundle
from utils.note_event_dataclasses import Note, NoteEvent, NoteEventListsBundle
from utils.task_manager import TaskManager
from config.config import shared_cfg
from config.config import audio_cfg as default_audio_cfg

UNANNOTATED_PROGRAM = 129

class AudioFileDataset(Dataset):
    """
    🎧 AudioFileDataset for validation/test:
    
    This dataset class is designed to be used ONLY with `batch_size=None` and 
    returns sliced audio segments and unsliced notes and sliced note events for
     a single song when `__getitem__` is called.

    Args:
        file_list (Union[str, bytes, os.PathLike], optional):
            Path to the file list. e.g. "../../data/yourmt3_indexes/slakh_validation_file_list.json"
        task_manager (TaskManager, optional): TaskManager instance. Defaults to TaskManager().
        fs (int, optional): Sampling rate. Defaults to 16000.
        seg_len_frame (int, optional): Segment length in frames. Defaults to 32767.
        seg_hop_frame (int, optional): Segment hop in frames. Defaults to 32767.
        sub_batch_size (int, optional): Sub-batch size that will be used in 
            generation of tokens. Defaults to 32.
        max_num_files (int, optional): Maximum number of files to be loaded. Defaults to None.
        
    
    Variables:
        file_list:
            '{dataset_name}_{split}_file_list.json' has the following keys:
            {
                'index':
                    {
                        'mtrack_id': mtrack_id,
                        'n_frames': n of audio frames
                        'stem_file': Dict of stem audio file info
                        'mix_audio_file': mtrack.mix_path,
                        'notes_file': available only for 'validation' and 'test'
                        'note_events_file': available only for 'train' and 'validation'
                        'midi_file': mtrack.midi_path
                    }
            }
            
    __getitem__(index) returns:

        audio_segment:
            torch.FloatTensor: (nearest_N_divisable_by_sub_batch_size, 1, seg_len_frame)

        notes_dict:
            {
                'mtrack_id': int,
                'program': List[int],
                'is_drum': bool, 
                'duration_sec': float, 
                'notes': List[Note], 
            }
            
        token_array:
            torch.LongTensor: (n_segments, seg_len_frame)

    """

    def __init__(
            self,
            file_list: Union[str, bytes, os.PathLike],
            task_manager: TaskManager = TaskManager(),
            #  tokenizer: Optional[EventTokenizerBase] = None,
            fs: int = 16000,
            seg_len_frame: int = 32767,
            seg_hop_frame: int = 32767,
            max_num_files: Optional[int] = None) -> None:
        
        # Move file list loading to a separate method
        self.file_list = self._load_file_list(file_list, max_num_files)
        self.fs = fs
        self.seg_len_frame = seg_len_frame
        self.seg_len_sec = seg_len_frame / fs
        self.seg_hop_frame = seg_hop_frame
        self.task_manager = task_manager

    @staticmethod
    def _load_file_list(file_list_path, max_num_files=None):
        with open(file_list_path, 'r') as f:
            fl = json.load(f)
        file_list = {int(key): value for key, value in fl.items()}
        if max_num_files:
            return dict(list(file_list.items())[:max_num_files])
        return file_list
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict, NoteEventListsBundle]:
        # get metadata
        metadata = self.file_list[index]
        audio_file = metadata['mix_audio_file']
        notes_file = metadata['notes_file']
        note_events_file = metadata['note_events_file']

        # load the audio
        audio = load_audio_file(audio_file, dtype=np.int16)  # returns bytes
        audio = audio / 2**15
        audio = audio.astype(np.float32)
        audio = audio.reshape(1, -1)
        audio_segments = slice_padded_array(
            audio,
            self.seg_len_frame,
            self.seg_hop_frame,
            pad=True,
        )  # (n_segs, seg_len_frame)
        audio_segments = rearrange(audio_segments, 'n t -> n 1 t').astype(np.float32)
        num_segs = audio_segments.shape[0]

        # load all notes and from a file (of a single song)
        notes_dict = np.load(notes_file, allow_pickle=True, fix_imports=False).tolist()

        # TODO: add midi_file path in preprocessing instead of here
        notes_dict['midi_file'] = metadata['midi_file']

        # tokenize note_events
        note_events_dict = np.load(note_events_file, allow_pickle=True, fix_imports=False).tolist()

        if self.task_manager.tokenizer is not None:
            # not using seg_len_sec to avoid accumulated rounding errors
            start_times = [i * self.seg_hop_frame / self.fs for i in range(num_segs)]
            note_event_segments = slice_multiple_note_events_and_ties_to_bundle(
                note_events_dict['note_events'],
                start_times,
                self.seg_len_sec,
            )

            # Support for multi-channel decoding
            if UNANNOTATED_PROGRAM in notes_dict['program']:
                has_unannotated_segments = [True] * num_segs
            else:
                has_unannotated_segments = [False] * num_segs

            token_array = self.task_manager.tokenize_note_events_batch(note_event_segments,
                                                                       start_time_to_zero=False,
                                                                       sort=True)
            return torch.from_numpy(audio_segments), notes_dict, torch.from_numpy(token_array).long()
    def __len__(self) -> int:
        return len(self.file_list)
    
class EvalDataLoader:
    """Wrapper class to make the dataloader picklable"""
    def __init__(
        self,
        dataset_name: str,
        split: str = 'validation',
        dataloader_config: Dict = {"num_workers": 0},
        task_manager: TaskManager = TaskManager(),
        max_num_files: Optional[int] = None,
        audio_cfg: Optional[Dict] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.dataloader_config = dataloader_config
        self.task_manager = task_manager
        self.max_num_files = max_num_files
        self.audio_cfg = audio_cfg or default_audio_cfg
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
     
    def __call__(self) -> DataLoader:
        if self.dataloader_config is None:
            self.dataloader_config = {"num_workers": 0}

        data_home = shared_cfg["PATH"]["data_home"]
        file_list = f"{data_home}/yourmt3_indexes/{self.dataset_name}_{self.split}_file_list.json"
        
        ds = AudioFileDataset(
            file_list,
            task_manager=self.task_manager,
            seg_len_frame=int(self.audio_cfg["input_frames"]),
            seg_hop_frame=int(self.audio_cfg["input_frames"]),
            max_num_files=self.max_num_files
        )
        # Force single process for validation
        self.dataloader_config["num_workers"] = 0
        dl = DataLoader(ds, batch_size=None, collate_fn=lambda k: k, **self.dataloader_config)
        return dl
        
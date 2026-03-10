import os
from typing import Iterator, List, Tuple, Optional

import tensorflow as tf

from app.utils import load_video as grid_load_video, load_alignments, char_to_num


class BaseDataLoader:
    def __init__(self, root: Optional[str] = None) -> None:
        self.root = root

    def get_video_paths(self, split: str = 'train') -> List[str]:
        raise NotImplementedError

    def load_video(self, path: str) -> tf.Tensor:
        raise NotImplementedError

    def get_transcript(self, path: str) -> tf.Tensor:
        raise NotImplementedError

    def iter_samples(self, split: str = 'train') -> Iterator[Tuple[tf.Tensor, tf.Tensor]]:
        for vp in self.get_video_paths(split):
            yield self.load_video(vp), self.get_transcript(vp)


class GRIDLoader(BaseDataLoader):
    def __init__(self, root: Optional[str] = None, speaker: str = 's1') -> None:
        super().__init__(root)
        self.speaker = speaker

    def get_video_paths(self, split: str = 'train') -> List[str]:
        # For simplicity, ignore split and list all videos for a speaker
        data_dir = os.path.join('..', 'data', self.speaker)
        if self.root:
            data_dir = os.path.join(self.root, self.speaker)
        if not os.path.isdir(data_dir):
            return []
        return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.mpg')]

    def load_video(self, path: str) -> tf.Tensor:
        return grid_load_video(path)

    def get_transcript(self, path: str) -> tf.Tensor:
        file_name = os.path.basename(path).split('.')[0]
        align_dir = os.path.join('..', 'data', 'alignments', self.speaker)
        if self.root:
            align_dir = os.path.join(self.root, 'alignments', self.speaker)
        align_path = os.path.join(align_dir, f'{file_name}.align')
        return load_alignments(align_path)


class IndianEnglishLoader(BaseDataLoader):
    def __init__(self, root: str) -> None:
        super().__init__(root)

    def get_video_paths(self, split: str = 'train') -> List[str]:
        # Expects structure: root/{split}/videos/*.mp4 and transcripts in root/{split}/transcripts/*.txt
        video_dir = os.path.join(self.root, split, 'videos')
        if not os.path.isdir(video_dir):
            return []
        return [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mpg'))]

    def load_video(self, path: str) -> tf.Tensor:
        return grid_load_video(path)

    def get_transcript(self, path: str) -> tf.Tensor:
        base = os.path.splitext(os.path.basename(path))[0]
        txt_path = os.path.join(self.root, 'train', 'transcripts', f'{base}.txt')
        if not os.path.exists(txt_path):
            txt_path = os.path.join(self.root, 'test', 'transcripts', f'{base}.txt')
        if not os.path.exists(txt_path):
            # Fallback: empty transcript
            return char_to_num(tf.constant(list('')))
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read().strip().lower()
        return char_to_num(tf.reshape(tf.strings.unicode_split(text, input_encoding='UTF-8'), (-1)))



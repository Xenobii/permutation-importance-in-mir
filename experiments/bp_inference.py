import sys
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple

import torch
from torch import nn
import numpy as np
import pretty_midi

# Resolve the path because someone didn't acount for using basic_pitch as a submodule
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models" / "basic_pitch"))

from models.basic_pitch.basic_pitch_torch import note_creation as infer
from models.basic_pitch.basic_pitch_torch.model import BasicPitchTorch
from models.basic_pitch.basic_pitch_torch.inference import unwrap_output, get_audio_input, predict
from models.basic_pitch.basic_pitch_torch.constants import (
    AUDIO_SAMPLE_RATE,
    AUDIO_N_SAMPLES,
    FFT_HOP
)

from experiments.permutations import CQTRandPerm, CQTMicrotonalPerm, CQTHighFreqPerm
from experiments.maskers import FundamentalMasking, HarmonicMasking, SoftFundamentalMasking


def hooked_inference(
        audio_path: Union[Path, str],
        model: nn.Module,
        hook_module: nn.Module
) -> Dict[str, np.array]:
    # Run model on the input audio path
    n_overlapping_frames = 30
    overlap_len = n_overlapping_frames * FFT_HOP
    hop_size = AUDIO_N_SAMPLES - overlap_len

    audio_windowed, _, audio_original_length = get_audio_input(audio_path, overlap_len, hop_size)
    audio_windowed = torch.from_numpy(audio_windowed).T
    if torch.cuda.is_available():
        audio_windowed = audio_windowed.cuda()

    # Hook permutations
    def bp_pre_hook(
            module: nn.Module,
            inputs: Tuple[torch.Tensor, ...],
    ) -> Tuple[torch.Tensor, ...]:
        (x,) = inputs
        x_perm = hook_module(x)
        return (x_perm,)
    handle = model.hs.register_forward_pre_hook(bp_pre_hook)
    
    # Run inference
    output = model(audio_windowed)

    """ 
    Output: {
        'onset'  : [B, T, P],
        'frame'  : [B, T, P],
        'contour': [B, T, 3*P]
    }
    """
    unwrapped_output = {k: unwrap_output(output[k], audio_original_length, n_overlapping_frames) for k in output}

    return unwrapped_output


def permuted_predict(
        audio_path: Union[Path, str],
        model_path: Union[Path, str],
        onset_threshold: float = 0.5, 
        frame_threshold: float = 0.3,
        minimum_note_length: float = 127.70,
        minimum_frequency: Optional[float] = None,
        maximum_frequency: Optional[float] = None,
        multiple_pitch_bends: bool = False,
        melodia_trick: bool = True,
        midi_tempo: float = 120,
) -> Tuple[Dict[str, np.array], pretty_midi.PrettyMIDI, List[Tuple[float, float, int, float, Optional[List[int]]]],]:
    # Load model
    model = BasicPitchTorch()
    model.load_state_dict(torch.load(str(model_path)))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # Load permuter
    permuter = CQTHighFreqPerm(start_bin=250, seed=0)
    # permuter = CQTRandPerm(p=0.1, seed=0)
    # permuter = CQTMicrotonalPerm(bins_per_semitone=3, seed=0)

    print(f"Predicting MIDI for {audio_path}...")
    
    model_output = hooked_inference(audio_path, model, permuter)
    
    # Convert to midi
    min_note_len = int(np.round(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    midi_data, note_events = infer.model_output_to_notes(
        model_output,
        onset_thresh         = onset_threshold,
        frame_thresh         = frame_threshold,
        min_note_len         = min_note_len,
        min_freq             = minimum_frequency,
        max_freq             = maximum_frequency,
        multiple_pitch_bends = multiple_pitch_bends,
        melodia_trick        = melodia_trick,
        midi_tempo           = midi_tempo,
    )

    return model_output, midi_data, note_events


def masked_predict(
        audio_path: Union[Path, str],
        midi_path: Union[Path, str],
        model_path: Union[Path, str],
        onset_threshold: float = 0.5, 
        frame_threshold: float = 0.3,
        minimum_note_length: float = 127.70,
        minimum_frequency: Optional[float] = None,
        maximum_frequency: Optional[float] = None,
        multiple_pitch_bends: bool = False,
        melodia_trick: bool = True,
        midi_tempo: float = 120,
) -> Tuple[Dict[str, np.array], pretty_midi.PrettyMIDI, List[Tuple[float, float, int, float, Optional[List[int]]]],]:
    model = BasicPitchTorch()
    model.load_state_dict(torch.load(str(model_path)))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # Load masker and required label
    masker = FundamentalMasking()
    masker.load_label(midi_path)

    print(f"Predicting MIDI for {audio_path}...")
    
    model_output = hooked_inference(audio_path, model, masker)

    # Convert to midi
    min_note_len = int(np.round(minimum_note_length / 1000 * (AUDIO_SAMPLE_RATE / FFT_HOP)))
    midi_data, note_events = infer.model_output_to_notes(
        model_output,
        onset_thresh         = onset_threshold,
        frame_thresh         = frame_threshold,
        min_note_len         = min_note_len,
        min_freq             = minimum_frequency,
        max_freq             = maximum_frequency,
        multiple_pitch_bends = multiple_pitch_bends,
        melodia_trick        = melodia_trick,
        midi_tempo           = midi_tempo,
    )

    return model_output, midi_data, note_events



if __name__ == "__main__":
    
    wav_path    = "test_data/maps_1.wav"
    midi_path   = "test_data/maps_1.mid"
    # wav_path    = "test_data/test_audio.wav"
    # midi_path   = "test_data/test_midi.MID"
    model_path  = "models/basic_pitch/assets/basic_pitch_pytorch_icassp_2022.pth"
    output_path = "test_data/test_output.mid"
    
    # model_out, midi_data, note_events = permuted_predict(
    #     wav_path,
    #     model_path=str(model_path))
    model_out, midi_data, note_events = masked_predict(
        wav_path,
        midi_path,
        model_path=str(model_path)
    )
    midi_data.write(str(output_path))
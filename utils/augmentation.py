"""
Data Augmentation Module for Audio Event Detection
Implements various augmentation techniques for audio data
"""

import numpy as np
import librosa
import torch
import random
from typing import Tuple, Optional
import yaml


class AudioAugmentor:
    """
    Audio data augmentation for training robustness
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize augmentor with configuration
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.aug_config = self.config['augmentation']
        self.enabled = self.aug_config['enabled']
        self.techniques = self.aug_config['techniques']
    
    def time_stretch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Time stretching (speed up or slow down)
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Time-stretched audio
        """
        if not self.techniques['time_stretch']['enabled']:
            return audio
        
        rate_range = self.techniques['time_stretch']['rate_range']
        rate = random.uniform(rate_range[0], rate_range[1])
        
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Pitch shifting
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Pitch-shifted audio
        """
        if not self.techniques['pitch_shift']['enabled']:
            return audio
        
        n_steps_range = self.techniques['pitch_shift']['n_steps_range']
        n_steps = random.uniform(n_steps_range[0], n_steps_range[1])
        
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    
    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise
        
        Args:
            audio: Audio array
            
        Returns:
            Noisy audio
        """
        if not self.techniques['add_noise']['enabled']:
            return audio
        
        noise_level = self.techniques['add_noise']['noise_level']
        noise = np.random.randn(len(audio))
        
        return audio + noise_level * noise
    
    def time_shift(self, audio: np.ndarray) -> np.ndarray:
        """
        Shift audio in time (circular shift)
        
        Args:
            audio: Audio array
            
        Returns:
            Time-shifted audio
        """
        if not self.techniques['time_shift']['enabled']:
            return audio
        
        shift_range = self.techniques['time_shift']['shift_range']
        shift = int(random.uniform(shift_range[0], shift_range[1]) * len(audio))
        
        return np.roll(audio, shift)
    
    def spec_augment(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        SpecAugment: Frequency and time masking on spectrogram
        
        Args:
            mel_spec: Mel-spectrogram (n_mels, time_frames)
            
        Returns:
            Augmented mel-spectrogram
        """
        if not self.techniques['spec_augment']['enabled']:
            return mel_spec
        
        spec = mel_spec.copy()
        n_mels, n_frames = spec.shape
        
        # Frequency masking
        freq_mask_param = self.techniques['spec_augment']['freq_mask_param']
        n_freq_masks = self.techniques['spec_augment']['n_freq_masks']
        
        for _ in range(n_freq_masks):
            f = random.randint(0, freq_mask_param)
            f0 = random.randint(0, n_mels - f)
            spec[f0:f0+f, :] = 0
        
        # Time masking
        time_mask_param = self.techniques['spec_augment']['time_mask_param']
        n_time_masks = self.techniques['spec_augment']['n_time_masks']
        
        for _ in range(n_time_masks):
            t = random.randint(0, time_mask_param)
            t0 = random.randint(0, n_frames - t)
            spec[:, t0:t0+t] = 0
        
        return spec
    
    def mixup(self, 
              audio1: np.ndarray, 
              audio2: np.ndarray, 
              label1: int, 
              label2: int,
              num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mixup augmentation: mix two audio samples
        
        Args:
            audio1: First audio array
            audio2: Second audio array
            label1: Label for first audio
            label2: Label for second audio
            num_classes: Total number of classes
            
        Returns:
            Tuple of (mixed_audio, mixed_label)
        """
        if not self.techniques['mixup']['enabled']:
            return audio1, self._to_one_hot(label1, num_classes)
        
        alpha = self.techniques['mixup']['alpha']
        lam = np.random.beta(alpha, alpha)
        
        # Mix audio
        mixed_audio = lam * audio1 + (1 - lam) * audio2
        
        # Mix labels (one-hot encoded)
        label1_oh = self._to_one_hot(label1, num_classes)
        label2_oh = self._to_one_hot(label2, num_classes)
        mixed_label = lam * label1_oh + (1 - lam) * label2_oh
        
        return mixed_audio, mixed_label
    
    def _to_one_hot(self, label: int, num_classes: int) -> np.ndarray:
        """Convert label to one-hot encoding"""
        one_hot = np.zeros(num_classes)
        one_hot[label] = 1
        return one_hot
    
    def augment_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply random audio augmentations
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Augmented audio
        """
        if not self.enabled:
            return audio
        
        # Randomly apply augmentations
        if random.random() > 0.5:
            audio = self.time_stretch(audio, sr)
        
        if random.random() > 0.5:
            audio = self.pitch_shift(audio, sr)
        
        if random.random() > 0.5:
            audio = self.add_noise(audio)
        
        if random.random() > 0.5:
            audio = self.time_shift(audio)
        
        return audio
    
    def augment_spectrogram(self, mel_spec: np.ndarray) -> np.ndarray:
        """
        Apply spectrogram augmentations
        
        Args:
            mel_spec: Mel-spectrogram
            
        Returns:
            Augmented mel-spectrogram
        """
        if not self.enabled:
            return mel_spec
        
        return self.spec_augment(mel_spec)


class SpecAugment(torch.nn.Module):
    """
    PyTorch module for SpecAugment
    Can be used directly in training pipeline
    """
    
    def __init__(self, 
                 freq_mask_param: int = 15,
                 time_mask_param: int = 35,
                 n_freq_masks: int = 2,
                 n_time_masks: int = 2):
        """
        Initialize SpecAugment module
        
        Args:
            freq_mask_param: Maximum frequency mask width
            time_mask_param: Maximum time mask width
            n_freq_masks: Number of frequency masks
            n_time_masks: Number of time masks
        """
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment
        
        Args:
            spec: Spectrogram tensor (batch, channels, freq, time)
            
        Returns:
            Augmented spectrogram
        """
        spec = spec.clone()
        
        # Get dimensions
        _, _, n_freq, n_time = spec.shape
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, n_freq - f)
            spec[:, :, f0:f0+f, :] = 0
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, n_time - t)
            spec[:, :, :, t0:t0+t] = 0
        
        return spec


def test_augmentation():
    """Test augmentation functions"""
    import matplotlib.pyplot as plt
    
    print("Testing audio augmentation...")
    
    # Create dummy audio
    sr = 22050
    duration = 4
    t = np.linspace(0, duration, sr * duration)
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Initialize augmentor
    augmentor = AudioAugmentor(config_path="../audio_event_detection/configs/config.yaml")
    
    # Test augmentations
    aug_audio = augmentor.augment_audio(audio, sr)
    
    print(f"Original audio shape: {audio.shape}")
    print(f"Augmented audio shape: {aug_audio.shape}")
    
    # Test spectrogram augmentation
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    aug_mel_spec = augmentor.augment_spectrogram(mel_spec)
    
    print(f"Original spectrogram shape: {mel_spec.shape}")
    print(f"Augmented spectrogram shape: {aug_mel_spec.shape}")
    
    print("Augmentation test complete!")


if __name__ == "__main__":
    test_augmentation()

import numpy as np
import librosa
import librosa.display
from typing import Tuple, Dict, Union
from scipy import signal

class SpeechFeatureExtractor:
    def __init__(self,
                 sr: int = 22050,
                 n_mfcc: int = 13,
                 hop_length: int = 512,
                 n_fft: int = 2048):
        """
        Initialize the speech feature extractor with configurable parameters.
        
        Args:
            sr: Sampling rate
            n_mfcc: Number of MFCC coefficients
            hop_length: Hop length for STFT
            n_fft: FFT window size
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.hop_length = hop_length
        self.n_fft = n_fft
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample if necessary.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio signal, sampling rate)
        """
        y, sr = librosa.load(file_path, sr=self.sr)
        return y, sr
    
    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """
        Extract Mel-frequency cepstral coefficients (MFCC).
        
        Args:
            y: Audio signal
            
        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(y=y, 
                                  sr=self.sr,
                                  n_mfcc=self.n_mfcc,
                                  hop_length=self.hop_length,
                                  n_fft=self.n_fft)
        return mfcc
    
    def extract_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        Extract spectrogram features.
        
        Args:
            y: Audio signal
            
        Returns:
            Spectrogram features
        """
        D = librosa.stft(y, 
                        hop_length=self.hop_length,
                        n_fft=self.n_fft)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        return S_db
    
    def extract_temporal_features(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal features including pitch contours and phoneme transitions.
        
        Args:
            y: Audio signal
            
        Returns:
            Dictionary of temporal features
        """
        # Extract pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=self.sr)
        
        # Extract onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        
        # Extract tempo
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)
        
        # Extract zero-crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        
        # Extract spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        
        features = {
            'mean_pitch': np.mean(pitches),
            'std_pitch': np.std(pitches),
            'tempo': tempo,
            'mean_zcr': np.mean(zcr),
            'std_zcr': np.std(zcr),
            'mean_spectral_centroid': np.mean(spectral_centroid),
            'std_spectral_centroid': np.std(spectral_centroid)
        }
        
        return features
    
    def extract_phoneme_transitions(self, y: np.ndarray) -> np.ndarray:
        """
        Extract phoneme transition features using energy-based segmentation.
        
        Args:
            y: Audio signal
            
        Returns:
            Phoneme transition features
        """
        # Compute energy envelope
        energy = librosa.feature.rms(y=y)[0]
        
        # Find peaks in energy envelope
        peaks, _ = signal.find_peaks(energy)
        
        # Compute transition features
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks)
            transition_features = {
                'mean_interval': np.mean(peak_intervals),
                'std_interval': np.std(peak_intervals),
                'num_transitions': len(peaks) - 1
            }
        else:
            transition_features = {
                'mean_interval': 0,
                'std_interval': 0,
                'num_transitions': 0
            }
            
        return transition_features
    
    def extract_all_features(self, y: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """
        Extract all features from the audio signal.
        
        Args:
            y: Audio signal
            
        Returns:
            Dictionary containing all extracted features
        """
        features = {}
        
        # Extract MFCC features
        features['mfcc'] = self.extract_mfcc(y)
        
        # Extract spectrogram
        features['spectrogram'] = self.extract_spectrogram(y)
        
        # Extract temporal features
        features['temporal'] = self.extract_temporal_features(y)
        
        # Extract phoneme transitions
        features['phoneme_transitions'] = self.extract_phoneme_transitions(y)
        
        return features
    
    def preprocess_audio(self, y: np.ndarray) -> np.ndarray:
        """
        Preprocess the audio signal for feature extraction.
        
        Args:
            y: Audio signal
            
        Returns:
            Preprocessed audio signal
        """
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Trim silence
        y, _ = librosa.effects.trim(y)
        
        return y 
import os
from typing import List, Optional, Union
from pathlib import Path
import time
from infer_rvc_python import BaseLoader

class AudioConverterError(Exception):
    """Custom exception for audio converter errors"""
    pass

def validate_audio_files(files: Union[str, List[str]]) -> List[str]:
    """
    Validate and normalize audio file inputs.
    
    Args:
        files: Single audio file path or list of paths
        
    Returns:
        List of validated file paths
        
    Raises:
        AudioConverterError: If files are invalid or missing
    """
    if not files:
        raise AudioConverterError("No audio files provided")
        
    if isinstance(files, str):
        files = [files]
        
    valid_files = []
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            raise AudioConverterError(f"Audio file not found: {file_path}")
        valid_files.append(str(path.absolute()))
        
    return valid_files

def generate_random_tag() -> str:
    """Generate a unique random tag for conversion"""
    return f"USER_{random.randint(10000000, 99999999)}"

def configure_converter(converter, config_params: dict) -> None:
    """
    Configure converter with the given parameters.
    
    Args:
        converter: Converter instance
        config_params: Dictionary containing configuration parameters
    """
    sample_rate = 44100 if config_params['audio_files'][0].endswith('.mp3') else 0
    converter.apply_conf(**config_params, resample_sr=sample_rate)
    time.sleep(0.1)  # Small delay for configuration settling

def mainpipe(
    model: str,
    audio_files: Union[str, List[str]],
    pitch_alg: str,
    pitch_lvl: float,
    index_inf: float,
    r_m_f: bool,
    e_r: float,
    c_b_p: bool,
    converter: BaseLoader
) -> dict:
    """
    Main pipeline function for audio conversion.
    
    Args:
        model: Model name to use for conversion
        audio_files: Audio file(s) to convert
        pitch_alg: Pitch algorithm to use
        pitch_lvl: Pitch level parameter
        index_inf: Index influence parameter
        r_m_f: Respiration median filtering flag
        e_r: Envelope ratio parameter
        c_b_p: Consonant breath protection flag
        converter: Converter instance
        
    Returns:
        Conversion result dictionary
        
    Raises:
        AudioConverterError: On validation or conversion errors
    """
    try:
        # Validate and normalize inputs
        audio_files = validate_audio_files(audio_files)
        
        # Get duration of first file
        duration_base = librosa.get_duration(filename=audio_files[0])
        print(f"Duration: {duration_base:.2f} seconds")
        
        # Generate unique tag
        random_tag = generate_random_tag()
        
        # Find model configuration
        model_config = next(
            (m for m in MODELS if m["model_name"] == model),
            None
        )
        
        if not model_config:
            raise AudioConverterError(f"Model not found: {model}")
            
        if not model_config["model"].endswith(".pth"):
            raise AudioConverterError("Invalid model file (must be .pth)")
            
        # Prepare configuration parameters
        config_params = {
            'tag': random_tag,
            'file_model': model_config["model"],
            'pitch_algo': pitch_alg,
            'pitch_lvl': pitch_lvl,
            'file_index': model_config["index"],
            'index_influence': index_inf,
            'respiration_median_filtering': r_m_f,
            'envelope_ratio': e_r,
            'consonant_breath_protection': c_b_p,
            'audio_files': audio_files
        }
        
        # Configure and execute conversion
        configure_converter(converter, config_params)
        result = converter(audio_files, random_tag, overwrite=False, parallel_workers=8)
        
        print(f"Conversion completed. Result: {result}")
        return result[0]
        
    except Exception as e:
        raise AudioConverterError(f"Conversion failed: {str(e)}")

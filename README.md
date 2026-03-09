# SCoT-SER
We proposes a novel framework, SCoT-SER, which combines a structured Chain-of-Thought(CoT) paradigm with a self-distillation mechanism.

## 1) Environment Setup

### Install
```bash
pip install -r requirements.txt
```

## 2) Prepare Data

### Data pipeline overview
Training JSON files must contain these fields per sample:
- `file_id`
- `audio_path`
- `original_emotion`
- `transcription`
- `analysis_output`
- `model_output_raw_reasoning`

### IEMOCAP

```bash
python dataset/IEMOCAP_preprocess.py --path /path/to/IEMOCAP --prodir /path/to/IEMOCAP
```

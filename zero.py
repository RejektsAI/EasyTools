# Modified from https://github.com/RVC-Boss/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
import os
from faster_whisper import WhisperModel
from scipy.io.wavfile import write
from datetime import datetime
import argparse, torch

parser = argparse.ArgumentParser(description='Generate speech from text')
parser.add_argument('--input_file', type=str, help='Path to the input audio file')
parser.add_argument('--audio_lang', type=str, help='Language of the input audio')
parser.add_argument('--text', type=str, help='Text to be processed')
parser.add_argument('--text_lang', type=str, help='Language to translate the text to')
parser.add_argument('--whisper_model', type=str, help='Whisper Model to use', default="medium.en")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

whisper_model = WhisperModel(args.whisper_model, device=device, compute_type="auto")

gpt_path = os.environ.get(
    "gpt_path", "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
)
sovits_path = os.environ.get("sovits_path", "pretrained_models/s2G488k.pth")
cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "pretrained_models/chinese-roberta-wwm-ext-large"
)

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]


import gradio as gr
import librosa
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from feature_extractor import cnhubert

cnhubert.cnhubert_base_path = cnhubert_base_path
from time import time as ttime
import datetime

from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from module.mel_processing import spectrogram_torch
from module.models import SynthesizerTrn
from my_utils import load_audio
from text import cleaned_text_to_sequence
from text.cleaner import clean_text

is_half = eval(
    os.environ.get("is_half", "True" if torch.cuda.is_available() else "False")
)

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)


# bert_model=bert_model.to(device)
def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)  #####输入是long不用管精度问题，精度随bert_model
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # if(is_half==True):phone_level_feature=phone_level_feature.half()
    return phone_level_feature.T


n_semantic = 1024
dict_s2 = torch.load(sovits_path, map_location="cpu")
hps = dict_s2["config"]


class DictToAttrRecursive:
    def __init__(self, input_dict):
        for key, value in input_dict.items():
            if isinstance(value, dict):
                # 如果值是字典，递归调用构造函数
                setattr(self, key, DictToAttrRecursive(value))
            else:
                setattr(self, key, value)


hps = DictToAttrRecursive(hps)
hps.model.semantic_frame_rate = "25hz"
dict_s1 = torch.load(gpt_path, map_location="cpu")
sovits_config = dict_s1["config"]
ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)

vq_model = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model,
)
if is_half == True:
    vq_model = vq_model.half().to(device)
else:
    vq_model = vq_model.to(device)
vq_model.eval()
print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
hz = 50
max_sec = sovits_config["data"]["max_sec"]
# t2s_model = Text2SemanticLightningModule.load_from_checkpoint(checkpoint_path=gpt_path, config=config, map_location="cpu")#########todo
t2s_model = Text2SemanticLightningModule(sovits_config, "ojbk", is_train=False)
t2s_model.load_state_dict(dict_s1["weight"])
if is_half == True:
    t2s_model = t2s_model.half()
t2s_model = t2s_model.to(device)
t2s_model.eval()
total = sum([param.nelement() for param in t2s_model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


dict_language = {"Chinese": "zh", "English": "en", "Japanese": "ja"}

splits = {
    "，",
    "。",
    "？",
    "！",
    ",",
    ".",
    "?",
    "!",
    "~",
    ":",
    "：",
    "—",
    "…",
}

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # There must be punctuation at the end, so you can jump out directly. The last paragraph has been added last time
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts

def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 3))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx] : split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)


def transcribe(inf_ref):
    file = inf_ref
    segments, _ = whisper_model.transcribe(audio=file,beam_size=1,vad_filter=True,best_of=1)
    sentences = [sentence.text for sentence in segments]
    transcription = " ".join(sentences)
    return transcription

def show(path,ext='',on_error=None):
    try:
        audios = list(filter(lambda x: x.endswith(ext), os.listdir(path)))
        audio_paths = []
        for audio in audios:
            audio_paths.append(os.path.join(path,audio))
        return audio_paths
    except:
        return on_error
    
def return_to(element):
    return element
    
def upload_audio(numpy):
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.wav'
    sample_rate, numpy_array = numpy
    try:
        write(filename=f"audios/{name}",data=numpy_array,rate=sample_rate)
    except Exception as e:
        print(f"Could not write audio because of error: {e}")
    return os.path.join("audios",name),{"choices":show("audios"),"__type__":"update","value":os.path.join("audios",name)}

# Load audio file and get transcription
def get_tts_wav(ref_wav_path, prompt_language, text, text_language):
    prompt_text = transcribe(ref_wav_path)
    if prompt_text != "":
        prompt_text = cut1(prompt_text)
        t0 = ttime()
        prompt_text = prompt_text.strip("\n")
        prompt_language, text = prompt_language, text.strip("\n")
        with torch.no_grad():
            wav16k, sr = librosa.load(ref_wav_path, sr=16000)  # Load 16kHz audio
            wav16k = torch.from_numpy(wav16k)
            if is_half == True:
                wav16k = wav16k.half().to(device) 
            else:
                wav16k = wav16k.to(device)
            ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
                "last_hidden_state"
            ].transpose(
                1, 2
            )  # Get SSL model output
            codes = vq_model.extract_latent(ssl_content) # Vector quantize SSL output
            prompt_semantic = codes[0, 0] # Get prompt semantic code
        t1 = ttime()
        prompt_language = dict_language[prompt_language] 
        text_language = dict_language[text_language]
        phones1, word2ph1, norm_text1 = clean_text(prompt_text, prompt_language) # Clean prompt text
        phones1 = cleaned_text_to_sequence(phones1) 
        texts = text.split("\n")
        audio_opt = []
        zero_wav = np.zeros(
            int(hps.data.sampling_rate * 0.3),  
            dtype=np.float16 if is_half == True else np.float32,
        )
        for text in texts:
            phones2, word2ph2, norm_text2 = clean_text(text, text_language) # Clean target text
            phones2 = cleaned_text_to_sequence(phones2)
            if prompt_language == "zh":
                bert1 = get_bert_feature(norm_text1, word2ph1).to(device) # Get BERT embedding for prompt
            else:
                bert1 = torch.zeros(  
                    (1024, len(phones1)),
                    dtype=torch.float16 if is_half == True else torch.float32,
                ).to(device)
            if text_language == "zh":
                bert2 = get_bert_feature(norm_text2, word2ph2).to(device) # Get BERT embedding for target
            else:
                bert2 = torch.zeros((1024, len(phones2))).to(bert1) 
            bert = torch.cat([bert1, bert2], 1) # Concatenate BERT embeddings

            all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0) # Phoneme IDs
            bert = bert.to(device).unsqueeze(0) 
            all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device) # Length of phoneme sequence
            prompt = prompt_semantic.unsqueeze(0).to(device) # Prompt semantic code
            t2 = ttime()
            with torch.no_grad():
                # Generate semantic code
                pred_semantic, idx = t2s_model.model.infer_panel(  
                    all_phoneme_ids,
                    all_phoneme_len,
                    prompt,
                    bert,
                    top_k=sovits_config["inference"]["top_k"],
                    early_stop_num=hz * max_sec,
                )
            t3 = ttime()
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0) # Get relevant part of semantic code
            refer = get_spepc(hps, ref_wav_path) # Get spectrogram
            if is_half == True:
                refer = refer.half().to(device)
            else:
                refer = refer.to(device)
            audio = (
                vq_model.decode( # Decode audio
                    pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
                )
                .detach()
                .cpu()
                .numpy()[0, 0] 
            )
            audio_opt.append(audio)
            audio_opt.append(zero_wav)
            t4 = ttime()
        print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))
        
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = f"../audios/spoken_{current_datetime}.wav"
        write(audio_path, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16), hps.data.sampling_rate)
        return audio_path, True
        #yield hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16)
    else:
        print("Prompt text empty.")
        return None, False

if "__name__" == "__main__":
    get_tts_wav(args.input_file, args.audio_lang, args.text, args.text_lang)
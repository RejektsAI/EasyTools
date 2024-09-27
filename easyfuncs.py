import os, subprocess
import gradio as gr
import shutil, time, torch, gc
from mega import Mega
from datetime import datetime
import pandas as pd
import os, sys, subprocess,  numpy as np
from pydub import AudioSegment
try: 
    from whisperspeech.pipeline import Pipeline as TTS
    whisperspeak_on = True   
except:
    whisperspeak_on = False

# Class to handle caching model urls from a spreadsheet
class CachedModels:
    def __init__(self):
        csv_url = "https://docs.google.com/spreadsheets/d/1tAUaQrEHYgRsm1Lvrnj14HFHDwJWl0Bd9x0QePewNco/export?format=csv&gid=1977693859"
        if os.path.exists("spreadsheet.csv"):
            self.cached_data = pd.read_csv("spreadsheet.csv") 
        else:
            self.cached_data = pd.read_csv(csv_url)
            self.cached_data.to_csv("spreadsheet.csv", index=False)
        # Cache model urls        
        self.models = {}
        for _, row in self.cached_data.iterrows():
            filename = row['Filename']
            url = None
            for value in row.values:
                if isinstance(value, str) and "huggingface" in value:
                    url = value
                    break
            if url:
                self.models[filename] = url
    # Get cached model urls    
    def get_models(self):
        return self.models
        
def show(path,ext,on_error=None):
    try:
        return list(filter(lambda x: x.endswith(ext), os.listdir(path)))
    except:
        return on_error
    
def run_subprocess(command):
    try:
        subprocess.run(command, check=True)
        return True, None
    except Exception as e:
        return False, e
        
def download_from_url(url=None, model=None):
    if not url:
        try:
            url = model[f'{model}']
        except:
            gr.Warning("Failed")
            return ''
    if model == '':
        try:
            model = url.split('/')[-1].split('?')[0]
        except:
            gr.Warning('Please name the model')
            return
    model = model.replace('.pth', '').replace('.index', '').replace('.zip', '')
    url = url.replace('/blob/main/', '/resolve/main/').strip()

    for directory in ["downloads", "unzips","zip"]:
        #shutil.rmtree(directory, ignore_errors=True)
        os.makedirs(directory, exist_ok=True)

    try:
        if url.endswith('.pth'):
            subprocess.run(["wget", url, "-O", f'assets/weights/{model}.pth'])
        elif url.endswith('.index'):
            os.makedirs(f'logs/{model}', exist_ok=True)
            subprocess.run(["wget", url, "-O", f'logs/{model}/added_{model}.index'])
        elif url.endswith('.zip'):
            subprocess.run(["wget", url, "-O", f'downloads/{model}.zip'])
        else:
            if "drive.google.com" in url:
                url = url.split('/')[0]
                subprocess.run(["gdown", url, "--fuzzy", "-O", f'downloads/{model}'])
            elif "mega.nz" in url:
                Mega().download_url(url, 'downloads')
            else:
                subprocess.run(["wget", url, "-O", f'downloads/{model}'])

        downloaded_file = next((f for f in os.listdir("downloads")), None)
        if downloaded_file:
            if downloaded_file.endswith(".zip"):
                shutil.unpack_archive(f'downloads/{downloaded_file}', "unzips", 'zip')
                for root, _, files in os.walk('unzips'):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file.endswith(".index"):
                            os.makedirs(f'logs/{model}', exist_ok=True)
                            shutil.copy2(file_path, f'logs/{model}')
                        elif file.endswith(".pth") and "G_" not in file and "D_" not in file:
                            shutil.copy(file_path, f'assets/weights/{model}.pth')
            elif downloaded_file.endswith(".pth"):
                shutil.copy(f'downloads/{downloaded_file}', f'assets/weights/{model}.pth')
            elif downloaded_file.endswith(".index"):
                os.makedirs(f'logs/{model}', exist_ok=True)
                shutil.copy(f'downloads/{downloaded_file}', f'logs/{model}/added_{model}.index')
            else:
                gr.Warning("Failed to download file")
                return 'Failed'

        gr.Info("Done")
    except Exception as e:
        gr.Warning(f"There's been an error: {str(e)}")
    finally:
        shutil.rmtree("downloads", ignore_errors=True)
        shutil.rmtree("unzips", ignore_errors=True)
        shutil.rmtree("zip", ignore_errors=True)
        return 'Done'
        
def speak(audio, text):
    print(f"({audio}, {text})")
    current_dir = os.getcwd()
    os.chdir('./gpt_sovits_demo')
    process = subprocess.Popen([
        "python", "./zero.py",
        "--input_file", audio,
        "--audio_lang", "English", 
        "--text", text,
        "--text_lang", "English"
    ], stdout=subprocess.PIPE, text=True)
    
    for line in process.stdout:
        line = line.strip()
        if "All keys matched successfully" in line:
            continue
        if line.startswith("(") and line.endswith(")"):
            path, finished = line[1:-1].split(", ")
            if finished:
                os.chdir(current_dir)
                return path
    os.chdir(current_dir)
    return None

def whisperspeak(text, tts_lang, cps=10.5):
    if whisperspeak_on is None: return None
    if not "tts_pipe" in locals(): tts_pipe = TTS(t2s_ref='whisperspeech/whisperspeech:t2s-v1.95-small-8lang.model', s2a_ref='whisperspeech/whisperspeech:s2a-v1.95-medium-7lang.model')
    from fastprogress.fastprogress import master_bar, progress_bar
    master_bar.update = lambda *args, **kwargs: None
    progress_bar.update = lambda *args, **kwargs: None
    
    output = f"audios/tts_audio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    tts_pipe.generate_to_file(output, text, cps=cps, lang=tts_lang)
    return os.path.abspath(output)

def stereo_process(audio1,audio2,choice):
    audio = audio1 if choice == "Input" else audio2
    print(audio)
    sample_rate, audio_array = audio
    if len(audio_array.shape) == 1:
        audio_bytes = audio_array.tobytes()
        segment = AudioSegment(
            data=audio_bytes,
            sample_width=audio_array.dtype.itemsize,  # 2 bytes for int16
            frame_rate=sample_rate,  # Use the sample rate from your tuple
            channels=1  # Adjust if your audio has more channels
            )
        samples = np.array(segment.get_array_of_samples())
        delay_samples = int(segment.frame_rate * (0.6 / 1000.0))
        left_channel = np.zeros_like(samples)
        right_channel = samples
        left_channel[delay_samples:] = samples[:-delay_samples]
        stereo_samples = np.column_stack((left_channel, right_channel))
        return (sample_rate, stereo_samples.astype(np.int16))
    else:
        return audio
    
def sr_process(audio1, audio2, choice):
    torch.cuda.empty_cache()
    gc.collect()
    if "tts_pipe" in locals(): del tts_pipe
    audio = audio1 if choice == "Input" else audio2
    sample_rate, audio_array = audio
    audio_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_array.dtype.itemsize,
        channels=1 if len(audio_array.shape) == 1 else 2
    )
    temp_file = os.path.join('TEMP', f'{choice}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav')
    audio_segment.export(temp_file, format="wav")
    output_folder = "SR"
    model_name = "speech"
    suffix = "_ldm"
    guidance_scale = 2.7
    ddim_steps = 50
    venv_dir = "audiosr"

    def split_audio(input_file, output_folder, chunk_duration=5.12):
        if os.path.exists(output_folder): shutil.rmtree(output_folder)
        os.makedirs(output_folder, exist_ok=True)
        ffmpeg_command = f"ffmpeg -i {input_file} -f segment -segment_time {chunk_duration} -c:a pcm_s16le {output_folder}/out%03d.wav"
        subprocess.run(ffmpeg_command, shell=True, check=True)

    def create_file_list(output_folder):
        file_list = os.path.join(output_folder, "file_list.txt")
        with open(file_list, "w") as f:
            for filename in sorted(os.listdir(output_folder)):
                if filename.endswith(".wav"):
                    f.write(os.path.join(output_folder, filename) + "\n")
        return file_list

    def run_audiosr(file_list, model_name, suffix, guidance_scale, ddim_steps, output_folder, venv_dir):
        command = f"{venv_dir}/bin/python -m audiosr --input_file_list {file_list} --model_name {model_name} --suffix {suffix} --guidance_scale {guidance_scale} --ddim_steps {ddim_steps} --save_path {output_folder}"
        try:
            subprocess.run(command, shell=True, check=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error running audiosr: {e.stderr.decode()}")


    split_audio(temp_file, output_folder)
    file_list = create_file_list(output_folder)
    run_audiosr(file_list, model_name, suffix, guidance_scale, ddim_steps, output_folder, venv_dir)

    output_file = None
    time.sleep(1)
    processed_chunks = []
    for root, dirs, files in os.walk(output_folder):
        for file in sorted(files):
            if file.startswith("out") and file.endswith(f"{suffix}.wav"):
                chunk_file = os.path.join(root, file)
                processed_chunks.append(AudioSegment.from_wav(chunk_file))

    if processed_chunks:
        merged_audio = sum(processed_chunks)
        output_file = os.path.join(output_folder, f"{choice}_merged{suffix}.wav")
        merged_audio.export(output_file, format="wav")
        
        display_file = AudioSegment.from_file(output_file)
        sample_rate = display_file.frame_rate
        audio_array = np.array(display_file.get_array_of_samples())
        return (sample_rate, audio_array)
    else:
        print(f"Error: Could not find any processed audio chunks in {output_folder}")
        return None

import os, io
import openai
import torch
import torchaudio
from transformers import pipeline
from dotenv import load_dotenv
from denoiser import pretrained
from denoiser.dsp import convert_audio
from temp_file import create_temp_audio, delete_temp_audio
from faster_whisper import WhisperModel

torch.set_num_threads(1)
torchaudio.set_audio_backend("soundfile")
load_dotenv()
PATH = os.getcwd().split('MaCALL')[0] + 'MaCALL/'


class SpeechtoText:
    def __init__(self, path_voicerecords: str, sample_rate: int = 16000):
        openai.api_key = os.getenv('OPEN_AI_API_KEY')

        self.path_voicerecords = path_voicerecords
        if not os.path.exists(self.path_voicerecords):
            os.makedirs(self.path_voicerecords)

        self.sample_rate = sample_rate
        self.denoiser_model = pretrained.dns64().cuda() if torch.cuda.is_available() else pretrained.dns64()
        self.DURACOES_AUDIOS = []
        print('----- IS GPU ENABLED: -----')
        print(torch.cuda.is_available())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.faster_whisper = None
        self.IFwhisper = None

    async def denoise_audio(self, audio) -> bytes:
        temp_audio_path = await create_temp_audio(audio, audio.filename)
        wav, sample_rate = torchaudio.load(temp_audio_path)
        wav = wav.to(self.device)

        assert wav.shape[0] == 1 and len(wav.shape) == 2, "Audio must be mono and have 2 dimensions"

        if sample_rate != 16000:  # caso o sample rate não seja 16000, converte para 16000
            wav = convert_audio(wav.cuda(), sample_rate,
                                self.denoiser_model.sample_rate, self.denoiser_model.chin)

        denoised_audio = torch.tensor([], device=self.device)  # tensor vazio na GPU para receber os trechos denoised
        with torch.no_grad():
            # dividindo o audio em trechos de 120 segundos e aplicando o denoiser em cada trecho
            trecho = 120
            for i in range(0, wav.shape[1], sample_rate * trecho):
                denoised_audio = torch.cat((denoised_audio, self.denoiser_model(wav[:, i:i + sample_rate * trecho])[0]),
                                           dim=1)
            del wav

            denoised_audio = denoised_audio.unsqueeze(0)

            buffer = io.BytesIO()
            torchaudio.save(buffer, denoised_audio[0].cpu(), sample_rate, format="ogg")
            buffer.seek(0)

        return buffer.read()

    async def get_local_faster_whisper_transcription(self, audio) -> str:
        transcript = ""
        # caso o modelo não tenha sido carregado ainda
        if self.faster_whisper is None:
            self.faster_whisper = WhisperModel("large", device=self.device, compute_type="int8")

        assert ".wav" in audio.filename or ".mp3" in audio.filename, "The file must have an .wav or .mp3 extension"
        temp_audio_path = await create_temp_audio(audio, audio.filename)
        segments, _ = self.faster_whisper.transcribe(temp_audio_path, vad_filter=True,
                                                     vad_parameters=dict(min_silence_duration_ms=2200,threshold = 0.7,min_speech_duration_ms = 250,window_size_samples = 1024,speech_pad_ms = 400), language='pt')
        result = []
        for segment in segments:
            result.append([segment.start, segment.end, segment.text])

        delete_temp_audio(audio.filename)
        return result

    async def get_local_insanely_fast_whisper_transcription(self, audio) -> dict:
        assert ".wav" in audio.filename or ".mp3" in audio.filename, "The file must have an .wav or .mp3 extension"
        temp_audio_path = await create_temp_audio(audio, audio.filename)

        transcript = ""
        # verificando se o modelo não foi carregado ainda
        if self.IFwhisper is None:
            self.faster_whisper = None
            self.IFwhisper = pipeline("automatic-speech-recognition",
                                      "openai/whisper-large-v2",
                                      torch_dtype=torch.float16,
                                      device="cuda:0")

            self.IFwhisper.model = self.IFwhisper.model.to_bettertransformer()
        # gerando os outputs
        outputs = self.IFwhisper(temp_audio_path,
                                 chunk_length_s=30,
                                 batch_size=24,
                                 generate_kwargs={"language": "portuguese"},
                                 return_timestamps=True)

        return outputs
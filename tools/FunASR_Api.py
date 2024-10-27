# FunASR_Api.py

import soundfile as sf
import speech_recognition as sr
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

class SpeechToText:
    def __init__(self):
        # 初始化FunASR模型
        model_dir = "iic\SenseVoiceSmall"  # 替换为你使用的模型路径
        self.model = AutoModel(
            model=model_dir,
            trust_remote_code=False,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",  # 如果没有GPU，请使用 "cpu"
            disable_update=True  # 禁用更新检查
        )
        self.recognizer = sr.Recognizer()

    def recognize_speech(self):
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 500  # 可以调整阈值
        recognizer.pause_threshold = 1  # 静音时 1 秒后自动停止录音

        print("请说话...")

        with sr.Microphone() as source:
            audio = recognizer.listen(source)

        # 保存音频到临时文件并处理
        with open("temp.wav", "wb") as f:
            f.write(audio.get_wav_data())

        try:
            # 调用 FunASR 模型进行语音识别
            res = self.model.generate(
                input="temp.wav",
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )

            # 后处理得到文本
            text = rich_transcription_postprocess(res[0]["text"])

            # 将识别到的文本保存到 .txt 文件
            with open("output_text.txt", "w", encoding="utf-8") as file:
                file.write(text)
            
            return "output_text.txt"
        except Exception as e:
            print("识别时发生错误:", str(e))
            return None


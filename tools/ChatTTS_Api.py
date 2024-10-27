# ChatTTS_Api.py

import ChatTTS
import torch
import torchaudio

class TextToSpeech:
    def __init__(self):
        # 初始化 ChatTTS 模型
        self.chat = ChatTTS.Chat()
        self.chat.load(compile=False)

    def convert_text_to_speech(self, text_file):
        # 从文本文件读取文本
        #with open(text_file, "r") as file:
            #text = file.read()

        with open(text_file, "r", encoding="utf-8") as file:
            text = file.read()


        print("正在将文本转换为语音...")

        texts = [text]  # 把单个文本转化为列表
        wavs = self.chat.infer(texts)  # 进行语音合成

        # 将生成的语音保存为 .wav 文件
        for i in range(len(wavs)):
            wav_filename = f"output_speech_{i}.wav"
            try:
                torchaudio.save(wav_filename, torch.from_numpy(wavs[i]).unsqueeze(0), 24000)
            except:
                torchaudio.save(wav_filename, torch.from_numpy(wavs[i]), 24000)
            
            print(f"语音文件保存为 {wav_filename}")
        return wav_filename

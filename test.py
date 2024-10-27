# main.py

from FunASR_Api import SpeechToText
from ChatTTS_Api import TextToSpeech
import time

def main():
    # 初始化语音识别和语音合成系统
    print("初始化系统中，请稍候...")
    stt_system = SpeechToText()
    tts_system = TextToSpeech()
    
    print("系统初始化完成。")

    # 步骤 1：调用语音识别功能
    print("准备进行语音识别，请说话...")
    start_time = time.time()  # 记录开始时间
    text_file = stt_system.recognize_speech()
    
    if text_file:
        elapsed_time = time.time() - start_time  # 计算识别所用时间
        print(f"语音识别已完成，文本保存在 {text_file}，用时 {elapsed_time:.2f} 秒")
    
        # 步骤 2：调用语音合成功能，将识别到的文本转换为语音
        print("正在进行语音合成，请稍候...")
        start_time = time.time()
        wav_file = tts_system.convert_text_to_speech(text_file)
        elapsed_time = time.time() - start_time
        print(f"语音合成已完成，音频文件保存在 {wav_file}，用时 {elapsed_time:.2f} 秒")
    else:
        print("语音识别失败，请重试。")

if __name__ == "__main__":
    main()

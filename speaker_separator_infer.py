import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import time
import keyboard
import mysql.connector
from scipy.spatial.distance import cosine
import speech_recognition as sr
import os
from sklearn.preprocessing import normalize

from profile_utils import extract_user_profile, save_user_profile_json, load_user_profile_json

# 설정
SR = 16000
CHANNELS = 1
BLOCK_DURATION = 0.5
SIMILARITY_THRESHOLD = 0.95
USER_PROFILE_PATH = "C:/Users/jyn36/Desktop/학교/캡스톤1/0016_G1A2E7_KYM_000001_join.wav"
PROFILE_JSON = "user_profile.json"
TEMP_WAV = "temp_segmented.wav"

def connect_to_mysql():
    return mysql.connector.connect(
        host='localhost',
        database='test',
        user='root',
        password='3672'
    )

def create_session(conn):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO sessions () VALUES ()")
    conn.commit()
    return cursor.lastrowid

def insert_segment(conn, session_id, speaker_type, start_sec, end_sec, text, similarity):
    cursor = conn.cursor()
    query = """
    INSERT INTO speech_segments (session_id, speaker_type, start_time, end_time, text, similarity)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    start_str = time.strftime("%H:%M:%S", time.gmtime(start_sec)) + f".{int((start_sec % 1)*1000):03d}"
    end_str = time.strftime("%H:%M:%S", time.gmtime(end_sec)) + f".{int((end_sec % 1)*1000):03d}"

    cursor.execute(query, (session_id, speaker_type, start_str, end_str, text, float(similarity)))
    conn.commit()

    print(f">>> INSERT: {speaker_type}, {start_str} ~ {end_str}, text='{text}', sim={similarity:.3f}")

def extract_mfcc_vector(segment_y, sr=SR):
    if len(segment_y) < 0.3 * sr:
        return None
    mfcc = librosa.feature.mfcc(y=segment_y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.concatenate((mfcc, delta, delta2), axis=0)
    vec = np.mean(combined.T, axis=0)
    return normalize(vec.reshape(1, -1))[0]

def fmt(t):
    return time.strftime("%M:%S", time.gmtime(t)) + f".{int((t % 1)*10):01d}"

def transcribe_google(filename):
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = r.record(source)
    try:
        return r.recognize_google(audio, language="ko-KR")
    except:
        return ""

def diarize_and_store(user_vec):
    print("🕹️ 'a' 누르면 시작 / 다시 누르면 종료")
    while not keyboard.is_pressed("a"):
        time.sleep(0.1)
    print("🎤 녹음 중... 다시 'a' 누르면 종료됩니다.")

    q = []
    segments = []
    start_time = time.time()

    with sd.InputStream(samplerate=SR, channels=CHANNELS, dtype='float32') as stream:
        while not keyboard.is_pressed("a"):
            block = stream.read(int(SR * BLOCK_DURATION))[0].flatten()
            q.append(block)

            now = time.time() - start_time
            s, e = max(0, now - BLOCK_DURATION), now

            if np.abs(block).mean() > 0.01:
                vec = extract_mfcc_vector(block, sr=SR)
                if vec is None:
                    continue
                sim = 1 - cosine(user_vec, vec)
                label = "self" if sim >= SIMILARITY_THRESHOLD else "other"

                if segments and segments[-1]["label"] == label:
                    segments[-1]["end"] = e
                else:
                    segments.append({"label": label, "start": s, "end": e, "sim": sim})

                print(f"[{fmt(s)} ~ {fmt(e)}] {label} (sim:{sim:.3f})")

    print("🛑 녹음 종료")
    audio = np.concatenate(q)
    sf.write(TEMP_WAV, audio, SR)

    print("\n📝 전체 스크립트:")
    text = transcribe_google(TEMP_WAV)
    words = text.strip().split()
    if not words:
        print("(인식된 텍스트 없음)")
        return

    conn = connect_to_mysql()
    session_id = create_session(conn)
    print(f"✅ 세션 저장됨: session_id = {session_id}")

    total_duration = sum(seg["end"] - seg["start"] for seg in segments)
    total_words = len(words)
    index = 0
    for seg in segments:
        ratio = (seg["end"] - seg["start"]) / total_duration
        count = round(ratio * total_words)
        chunk = words[index:index + count]
        sent = " ".join(chunk)
        index += count
        print(f"{seg['label']}: {sent}")
        insert_segment(conn, session_id, seg["label"], seg["start"], seg["end"], sent, seg["sim"])

    conn.close()
    print("✅ 모든 segment 저장 완료")
if __name__ == "__main__":
    if os.path.exists(PROFILE_JSON):
        print("📥 저장된 사용자 프로파일 불러오는 중...")
        user_vec = load_user_profile_json(PROFILE_JSON)
    else:
        print("🔍 사용자 프로파일 생성 중...")
        user_vec = extract_user_profile(USER_PROFILE_PATH)
        save_user_profile_json(user_vec, PROFILE_JSON)

    diarize_and_store(user_vec)

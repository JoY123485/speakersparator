import librosa
import numpy as np
import json
from sklearn.preprocessing import normalize

SR = 16000

def extract_user_profile(wav_path, sr=SR):
    y, _ = librosa.load(wav_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.concatenate((mfcc, delta, delta2), axis=0)
    vec = np.mean(combined.T, axis=0)
    norm_vec = normalize(vec.reshape(1, -1))[0]
    return norm_vec.tolist()

def save_user_profile_json(vec, path="user_profile.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vec, f)
    print(f"✅ 사용자 프로파일 저장됨: {path}")

def load_user_profile_json(path="user_profile.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data)

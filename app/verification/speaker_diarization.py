import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from app.verification.constants import ReasonCode, CLUSTER_SHARE_MINIMUM, DIARIZATION_DISTANCE_THRESHOLD, TARGET_SAMPLE_RATE

def check_multiple_speakers(audio: np.ndarray, speech_regions: list[dict], ecapa_model) -> tuple[bool, int, list, str, str, dict]:
    """
    Checks for multiple speakers by clustering embeddings.
    """
    if not speech_regions or not ecapa_model:
        return True, 1, [1.0], None, None, {}

    chunk_size = 2.0
    embeddings = []
    chunk_durations = []
    
    tensor_audio = torch.from_numpy(audio).float()

    for r in speech_regions:
        start = r["start"]
        end = r["end"]
        
        curr = start
        while curr < end:
            c_end = min(curr + chunk_size, end)
            dur = c_end - curr
            
            if dur >= 0.5:
                s_idx = int(curr * TARGET_SAMPLE_RATE)
                e_idx = int(c_end * TARGET_SAMPLE_RATE)
                chunk_tensor = tensor_audio[s_idx:e_idx].unsqueeze(0)
                
                with torch.no_grad():
                    emb = ecapa_model.encode_batch(chunk_tensor)
                    embeddings.append(emb.squeeze(0).squeeze(0).numpy())
                chunk_durations.append(dur)
                
            curr += chunk_size

    if not embeddings:
        return True, 1, [1.0], None, None, {}

    embeddings = np.array(embeddings)
    
    if len(embeddings) == 1:
        return True, 1, [1.0], None, None, {"num_clusters": 1, "cluster_shares": [1.0]}

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=DIARIZATION_DISTANCE_THRESHOLD
    )
    labels = clustering.fit_predict(embeddings)
    
    num_clusters = len(set(labels))
    total_duration = sum(chunk_durations)
    
    cluster_shares = []
    for c in range(num_clusters):
        c_dur = sum(d for i, d in enumerate(chunk_durations) if labels[i] == c)
        cluster_shares.append(c_dur / total_duration)

    cluster_shares.sort(reverse=True)
    significant_clusters = [s for s in cluster_shares if s >= CLUSTER_SHARE_MINIMUM]
    
    passed = len(significant_clusters) <= 1
    
    reason_code = ReasonCode.MULTIPLE_SPEAKERS_DETECTED.value if not passed else None
    message = "This recording appears to contain more than one speaker. Please provide audio of a single person's voice only." if not passed else None
    
    return passed, len(significant_clusters), cluster_shares, reason_code, message, {
        "num_clusters": len(significant_clusters),
        "cluster_shares": [round(s, 3) for s in cluster_shares]
    }

def get_audio_embedding(audio: np.ndarray, ecapa_model) -> np.ndarray:
    """Returns a single embedding for the whole audio, for batch cross-file check"""
    tensor_audio = torch.from_numpy(audio).float().unsqueeze(0)
    with torch.no_grad():
        emb = ecapa_model.encode_batch(tensor_audio)
    return emb.squeeze(0).squeeze(0).numpy()

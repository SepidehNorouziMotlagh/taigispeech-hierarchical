from collections import Counter

import torch
from datasets import Audio, load_dataset
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor

from utils import set_seed

BACKBONE_NAME = "microsoft/wavlm-base-plus"
SAMPLE_RATE = 16000

EMERGENCY_INTENTS = {
    "SOS_CALL",
    "FALL_HELP",
    "BREATHING_CHEST_EMERG",
    "PAIN_GENERAL",
}


def load_taigispeech():
    ds = load_dataset("TaigiSpeech/TaigiSpeech")
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    return ds


def get_label_names(ds):
    feature = ds["train"].features["intent"]
    if hasattr(feature, "names"):
        return list(feature.names)
    return sorted(set(ds["train"]["intent"]))


def resolve_intent(item, feature):
    val = item["intent"]
    if isinstance(val, str):
        return val
    return feature.int2str(val)


def to_binary(intent_name: str) -> int:
    return 1 if intent_name in EMERGENCY_INTENTS else 0


class TaigiDataset(Dataset):
    def __init__(self, hf_split, feature_extractor, label2id):
        self.hf_split = hf_split
        self.feature_extractor = feature_extractor
        self.label2id = label2id
        self.intent_feature = hf_split.features["intent"]

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        item = self.hf_split[idx]
        audio = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]
        assert sr == SAMPLE_RATE, f"expected {SAMPLE_RATE}, got {sr}"

        inputs = self.feature_extractor(
            audio, sampling_rate=sr, return_tensors="pt"
        )
        input_values = inputs.input_values.squeeze(0)

        intent_name = resolve_intent(item, self.intent_feature)
        return {
            "input_values": input_values,
            "intent": torch.tensor(self.label2id[intent_name], dtype=torch.long),
            "emergency": torch.tensor(to_binary(intent_name), dtype=torch.long),
        }


def collate_fn(batch):
    max_len = max(b["input_values"].shape[0] for b in batch)
    input_values = torch.zeros(len(batch), max_len)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_values"].shape[0]
        input_values[i, :L] = b["input_values"]
        attention_mask[i, :L] = 1
    return {
        "input_values": input_values,
        "attention_mask": attention_mask,
        "intent": torch.stack([b["intent"] for b in batch]),
        "emergency": torch.stack([b["emergency"] for b in batch]),
    }


def build_datasets():
    set_seed(42)
    ds = load_taigispeech()
    label_names = get_label_names(ds)
    label2id = {name: i for i, name in enumerate(label_names)}
    feature_extractor = AutoFeatureExtractor.from_pretrained(BACKBONE_NAME)
    splits = {
        split: TaigiDataset(ds[split], feature_extractor, label2id)
        for split in ["train", "val", "test"]
    }
    return splits, label_names, label2id


if __name__ == "__main__":
    set_seed(42)
    ds = load_taigispeech()
    label_names = get_label_names(ds)
    print("Label names:", label_names)
    print("Emergency labels:", sorted(EMERGENCY_INTENTS))

    for split in ["train", "val", "test"]:
        n = len(ds[split])
        speakers = set(ds[split]["speaker_id"])
        print(f"{split}: {n} utterances, {len(speakers)} speakers")

    train_s = set(ds["train"]["speaker_id"])
    val_s = set(ds["val"]["speaker_id"])
    test_s = set(ds["test"]["speaker_id"])
    assert train_s.isdisjoint(val_s), "train/val speaker overlap"
    assert train_s.isdisjoint(test_s), "train/test speaker overlap"
    assert val_s.isdisjoint(test_s), "val/test speaker overlap"
    print("Speaker-disjoint: OK")

    for split in ["train", "val", "test"]:
        feature = ds[split].features["intent"]
        names = [resolve_intent(ds[split][i], feature) for i in range(len(ds[split]))]
        counts = dict(Counter(names))
        binary = sum(counts.get(k, 0) for k in EMERGENCY_INTENTS)
        print(f"{split} emergency={binary}, non-emergency={sum(counts.values()) - binary}")
        print(f"{split} per-intent: {counts}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(BACKBONE_NAME)
    item = ds["train"][0]
    out = feature_extractor(item["audio"]["array"], sampling_rate=SAMPLE_RATE, return_tensors="pt")
    print("Feature shape:", tuple(out.input_values.shape))

    splits, _, label2id = build_datasets()
    sample = splits["train"][0]
    print("Dataset sample keys:", sample.keys())
    print("input_values:", sample["input_values"].shape, sample["input_values"].dtype)
    print("intent id:", sample["intent"].item(), "| emergency:", sample["emergency"].item())
    print("label2id:", label2id)
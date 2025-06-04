import h5py
import os
import pandas as pd
import numpy as np
import torch

from pre_calculate import LLM
from keypoints import KeypointProcessing

main_directory = "/home/giorgio6846/Code/Sign-AI"

WRITE = True

def save_keypoints(hdf5Group, videoFolderPath):
    if WRITE:
        group = hdf5Group.require_group("keypoints")

    for video_idx, videoName  in enumerate(sorted(os.listdir(videoFolderPath))):
        name, _ = os.path.splitext(videoName)
        videoPath = os.path.join(videoFolderPath, videoName)

        if not metaDF.loc[metaDF["id"]==name].shape[0] == 1:
            continue
        
        keypoint = keypoint_tool.process_keypoints(videoPath)

        if WRITE:
            group.create_dataset(str(video_idx), data=keypoint, compression="gzip", compression_opts=4)

def save_embeddings(hdf5Group, videoFolderPath):
    if WRITE:
        group = hdf5Group.require_group("embeddings")

    for video_idx, videoName  in enumerate(sorted(os.listdir(videoFolderPath))):
        name, _ = os.path.splitext(videoName)

        if not metaDF.loc[metaDF["id"]==name].shape[0] == 1:
            continue

        label = metaDF.loc[metaDF["id"]==name]["label"].values[0]
        print(label)
        embedding = llm.run(label)
        embedding = embedding.cpu().numpy()

        if WRITE:
            group.create_dataset(str(video_idx), data=embedding, compression="gzip", compression_opts=4)

def save_labels(hdf5Group, videoFolderPath):
    if WRITE:
        group = hdf5Group.require_group("labels")

    for video_idx, videoName  in enumerate(sorted(os.listdir(videoFolderPath))):
        name, _ = os.path.splitext(videoName)

        if not metaDF.loc[metaDF["id"]==name].shape[0] == 1:
            continue

        label = metaDF.loc[metaDF["id"]==name]["label"].values[0]

        if WRITE:
            dt = h5py.string_dtype(encoding='utf-8')
            group.create_dataset(str(video_idx), data=[label], dtype=dt, compression="gzip")

if __name__ == "__main__":
    dataPath = os.path.join(main_directory, "data")
    keypoint_tool = KeypointProcessing()
    llm = LLM(main_directory)
    f = h5py.File(os.path.join(dataPath, "dataset.hdf5"), 'a')

    for Folder in os.listdir(dataPath):
        folderPath = os.path.join(dataPath, Folder)

        if not os.path.isdir(folderPath):
            continue 

        if Folder in f:
            group = f[Folder]
        else:
            group = f.require_group(Folder)

        videoFolderPath = os.path.join(folderPath, "videos")
        metaDF = pd.read_csv(os.path.join(folderPath, "meta.csv"))

        if "keypoints" not in group:
            save_keypoints(group, videoFolderPath)

        if "embeddings" not in group:
            save_embeddings(group, videoFolderPath)

        if "labels" not in group:
            save_labels(group, videoFolderPath)
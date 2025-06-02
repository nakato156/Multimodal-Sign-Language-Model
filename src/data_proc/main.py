import h5py
import os
os.environ["AUX_USE_SYMLINKS"] = "True"

import pandas as pd
import numpy as np

from llm_tools import LLMTools
from keypoints import KeypointProcessing

cd = "/home/giorgio6846/Code/Sign-AI/Sign-Giorgio/src/data_proc"

if __name__ == "__main__":
    dataPath = os.path.join(cd, os.pardir, os.pardir, os.pardir, os.pardir, "data")
    embedding_tool = LLMTools()
    keypoint_tool = KeypointProcessing()
    f = h5py.File(os.path.join(dataPath, "dataset.hdf5"), 'a')

    for Folder in os.listdir(dataPath):
        folderPath = os.path.join(dataPath, Folder)

        embeddings = []
        keypoints = []
        labels = []

        if os.path.isdir(folderPath):
            videoFolderPath = os.path.join(folderPath, "videos")
            metaDF = pd.read_csv(os.path.join(folderPath, "meta.csv"))

            for videoName in os.listdir(videoFolderPath):
                name, _ = os.path.splitext(videoName)
                videoPath = os.path.join(videoFolderPath, videoName)

                if metaDF.loc[metaDF["id"]==name].shape[0] == 1:
                    label = metaDF.loc[metaDF["id"]==name]["label"].values[0]
                    embedding = embedding_tool.process_embedding(label)
                    keypoint = keypoint_tool.process_keypoints(videoPath)

                    labels.append(label)
                    embeddings.append(embedding)
                    labels.append(keypoint)
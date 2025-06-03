import h5py
import os
import pandas as pd
import numpy as np

from pre_calculate import LLM
from keypoints import KeypointProcessing

main_directory = "/home/giorgio6846/Code/Sign-AI"

WRITE = True

if __name__ == "__main__":
    dataPath = os.path.join(main_directory, "data")
    keypoint_tool = KeypointProcessing()
    llm = LLM(main_directory)
    f = h5py.File(os.path.join(dataPath, "dataset.hdf5"), 'a')

    for Folder in os.listdir(dataPath):

        if Folder in f:
            continue

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
                    embedding = llm.run(label)
                    keypoint = keypoint_tool.process_keypoints(videoPath)

                    labels.append(label)
                    embeddings.append(embedding)
                    keypoints.append(keypoint)

            if WRITE:
                embeddings = np.array(embeddings)    
                keypoints = np.array(keypoints)    
                labels = np.array(labels, dtype='S')    

                group = f.require_group(Folder)
                group.create_dataset("embeddings", data=embeddings, compression="gzip", compression_opts=4, chunks=True)
                group.create_dataset("keypoints", data=keypoints, compression="gzip", compression_opts=4)
                group.create_dataset("labels", data=labels, compression="gzip", compression_opts=4)
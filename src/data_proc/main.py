import h5py
import os
import pandas as pd

from pre_calculate import LLM
from keypoints import KeypointProcessing

main_directory = "/home/giorgio6846/Code/Sign-AI"


def save_keypoints(hdf5Group, videoFolderPath):
    group = hdf5Group.require_group("keypoints")
    videoList = sorted(os.listdir(videoFolderPath))

    print("Keypoints: ", len(videoList))
    for video_idx, videoName  in enumerate(videoList):
        print(video_idx)

        if str(video_idx) in group:
            continue
        
        name, _ = os.path.splitext(videoName)
        videoPath = os.path.join(videoFolderPath, videoName)

        if not metaDF.loc[metaDF["id"]==name].shape[0] == 1:
            continue
        
        keypoint = keypoint_tool.process_keypoints(videoPath)

        print(keypoint.shape)
        group.create_dataset(str(video_idx), data=keypoint, compression="gzip", compression_opts=4)

def save_embeddings(hdf5Group, videoFolderPath):
    group = hdf5Group.require_group("embeddings")
    videoList = sorted(os.listdir(videoFolderPath))

    print("Embeddings: ", len(videoList))
    for video_idx, videoName  in enumerate(videoList):
        print(video_idx)

        if str(video_idx) in group:
            continue

        name, _ = os.path.splitext(videoName)

        if not metaDF.loc[metaDF["id"]==name].shape[0] == 1:
            continue

        label = metaDF.loc[metaDF["id"]==name]["label"].values[0]
        print(label)
        embedding = llm.run(label)
        embedding = embedding.cpu().numpy()

        group.create_dataset(str(video_idx), data=embedding, compression="gzip", compression_opts=4)

def save_labels(hdf5Group, videoFolderPath):
    group = hdf5Group.require_group("labels")
    videoList = sorted(os.listdir(videoFolderPath))

    print("labels: ", len(videoList))
    for video_idx, videoName  in enumerate(videoList):
        print(video_idx)
        
        if str(video_idx) in group:
            continue

        name, _ = os.path.splitext(videoName)

        if not metaDF.loc[metaDF["id"]==name].shape[0] == 1:
            continue

        label = metaDF.loc[metaDF["id"]==name]["label"].values[0]

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

        keypoint_tool.load_model()        
        save_keypoints(group, videoFolderPath)
        f.flush()
        keypoint_tool.unload_model()        

        llm.load_model()
        save_embeddings(group, videoFolderPath)
        f.flush()
        llm.unload_model()

        save_labels(group, videoFolderPath)
        f.flush()
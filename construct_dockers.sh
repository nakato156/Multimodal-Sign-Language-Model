docker build -f Dockerfile.data -t sign-data:latest .
docker build -f Dockerfile.worker -t sign-worker:latest .

cd ../data
docker run -it --rm --shm-size=1g -v "$(pwd)/dataset.hdf5":/output/data/dataset.hdf5 -p 50051:50051 --name sign-ai-data-node sign-data:latest
#docker run -it --rm --name sign-ai-worker sign-worker:latest
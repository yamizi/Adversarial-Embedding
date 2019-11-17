export IMAGENET_ACCESS_KEY="f71790c5bd56628cd74635b9f5c4f56084452633"
export IMAGENET_USERNAME="yamizi"
#nohup bash download_imagenet.sh ./imagenet synsets.txt >& download.log &

python ./imagenet_loader.py
from bs4 import BeautifulSoup
import numpy as np
import requests
import cv2
import PIL.Image
import urllib
import os
from keras.preprocessing.image import save_img, load_img, img_to_array, array_to_img
import argparse



img_rows, img_cols = 256, 256 #number of rows and columns to convert the images to
input_shape = (img_rows, img_cols, 3)#format to store the images (rows, columns,channels) called channels last

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	# return the image
	return image

#n_of_training_images:the number of training images to use

def run(split_urls, img_class="", n_of_training_images=100):
    remaining = min(len(split_urls),n_of_training_images)

    for progress in range(remaining):#store all the images on a directory
        # Print out progress whenever progress is a multiple of 20 so we can follow the
        # (relatively slow) progress
        if(progress%20==0):
            print(progress)
        try:
            I = url_to_image(split_urls[progress])
            if (len(I.shape))==3: #check if the image has width, length and channels
                save_path = '/imagenet/train/{}/img{}.jpg'.format(img_class,str(progress))#create a name of each image
                save_img(save_path,I)

        except:
            None
            
            
    #Test data:

    remaining = min(len(split_urls)-n_of_training_images,50)
    for progress in range(remaining):#store all the images on a directory
        # Print out progress whenever progress is a multiple of 20 so we can follow the
        # (relatively slow) progress
        if(progress%20==0):
            print(progress)
        try:
            I = url_to_image(split_urls[n_of_training_images+progress])#get images that are different from the ones used for training
            if (len(I.shape))==3: #check if the image has width, length and channels
                save_path = '/imagenet/test/{}/img{}.jpg'.format(img_class,str(progress))#create a name of each image
                save_img(save_path,I)

        except:
            None







if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--u", "-u", help="username",
                        type=str, default=os.environ.get('IMAGENET_USERNAME',''))
    parser.add_argument("--k", "-k", help="access key", type=str, default=os.environ.get('IMAGENET_ACCESS_KEY',''))

    args = parser.parse_args()
    user_name = args.u
    access_key = args.k

    set_id = "n02834778"
    url = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}&username={}&accesskey={}".format(set_id,user_name,access_key)
    print(url)
    page = requests.get(url)#bicycle synset
    soup = BeautifulSoup(page.content, 'html.parser')#puts the content of the website into the soup variable, each url on a different line
    str_soup=str(soup)#convert soup to string so it can be split
    _split_urls=str_soup.split('\r\n')#split so each url is a different possition on a list

    run(_split_urls)
    
"""
Generating Steganography images using Adversarial attacks and comparing impact of data loss on recovery rate
using different models
"""



import sys
sys.path.append("./")
from experiments import logger, RANDOM_SEED

import os

import json

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy import stats



keras_values = {"rotate_recovery":0.5192,"crop_recovery":0.6217,"upscale_recovery":0.5513,"downscale_recovery":0.9872,"color_depth_recovery":0.4487,"compress90_recovery":0.8782,"compress75_recovery":0.5897,"compress50_recovery":0.35897}
keras_values = {**keras_values, "ssim":6.02e-2,"lpips":3.34e-3}
def runa(experiment_time="1571050357"):
    f = json.load(open("./experiments/results/{}.json".format(experiment_time)))
    
    models = []
    rotate_recovery =[]
    crop_recovery =[]
    upscale_recovery =[]
    compress_recovery =[]
    
    for i, combination in enumerate(f):
        rotate_recovery.append(combination["rotate_recovery"])
        crop_recovery.append(combination["crop_recovery"])
        upscale_recovery.append(combination["upscale_recovery"])
        compress_recovery.append(combination["compress_recovery"])
        models.append(combination["model"])

    rotate_recovery = np.array(rotate_recovery)
    crop_recovery = np.array(crop_recovery)
    upscale_recovery = np.array(upscale_recovery)
    compress_recovery = np.array(compress_recovery)

    suffix = "recovery rate ({} models)".format(len(f))
    n_bins = 20
    m = 0
    mx = 1
    metrics = {"rotate_recovery":("Post Rotation",rotate_recovery),"crop_recovery":("Post Cropping",crop_recovery),"upscale_recovery":("Post Upscaling",upscale_recovery),"compress75_recovery":("Post Compression (quality=75)",compress_recovery)}
    
    #metrics = {"ssim":("SSIM",ssim),"lpips":("LPIPS",lpips),"psnr":("PSNR",psnr),"color_depth_recovery":("Post Color Depth Reduction",color_recovery),"compress50_recovery":("Post Compression (quality=50)",compress50_recovery),"compress90_recovery":("Post Compression (quality=90)",compress90_recovery)}
    
    plot_metrics(metrics, suffix)

    # plt.figure()
    # plt.title("Post Rotation {}".format(suffix))
    # plt.hist(rotate_recovery, n_bins, weights=np.ones(len(rotate_recovery)) / len(rotate_recovery),rwidth=0.85)
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.xlabel('Recovery rate')
    # plt.ylabel('Count')

    # plt.figure()
    # plt.title("Post Cropping {}".format(suffix))
    # plt.hist(crop_recovery, n_bins, weights=np.ones(len(rotate_recovery)) / len(rotate_recovery))
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.xlabel('Recovery rate')
    # plt.ylabel('Count')

    # plt.figure()
    # plt.title("Post Upscaling {}".format(suffix))
    # plt.hist(upscale_recovery, n_bins, weights=np.ones(len(rotate_recovery)) / len(rotate_recovery))
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.xlabel('Recovery rate')
    # plt.ylabel('Count')
    

    # plt.figure()
    # plt.title("Post Compression (quality=75) {}".format(suffix))
    # n,bins,edges = plt.hist(compress_recovery, n_bins, weights=np.ones(len(rotate_recovery)) / len(rotate_recovery), color='#0504aa',
    #                         alpha=0.7, rwidth=0.9,edgecolor='k', cumulative=True)
    # plt.grid(axis='y', alpha=0.75)
    # plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.xlabel('Recovery rate')
    # plt.ylabel('Count')
    # m = min(compress_recovery)
    # mx = max(compress_recovery)
    # plt.xlim(m,mx)
    # b = bins+1/n_bins/2
    # plt.xticks(b,np.around(b,2))

    plt.show()


def boxplot(f, label, labels):
    fig, ax = plt.subplots()

    #f = f[:2]
    #labels = labels[:2]
    adv_map = np.transpose(np.array(f))
    adv_map = adv_map/adv_map.max()
    ax.boxplot(adv_map, labels=labels)
    ax.set_title("{} inputs".format(label))
    
    fig.tight_layout()


def plot_metrics(metrics, suffix):

    n_bins = 20
    m = 0
    mx = 1

    for k, (title, tbl) in metrics.items():
        plt.figure()

        indices = np.arange(len(tbl))
        #plt.scatter(indices,tbl)

        n,bins,edges = plt.hist(tbl, np.arange(0,n_bins+1)/n_bins, weights=np.ones(len(tbl)) / len(tbl), color='#0504aa',
                                 alpha=0.7, rwidth=0.9,edgecolor='k', cumulative=True)


        plt.grid(axis='y', alpha=0.75)

        if k=="ssim" or k=="lpips" or k=="psnr":
            plt.title("{}".format(title))
            plt.ylabel('Loss value')
            plt.xlabel('Model')
            plt.xticks(indices, [""]*len(tbl))

            b = bins
            lbls = np.around(b,2)
            plt.xticks(b,lbls)
        
        else:
            plt.title("{} {}".format(title,suffix))
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
            plt.xlabel('Recovery rate')
            plt.xlim(m,mx)
            plt.ylabel('Count')
        
        

        keras_value = keras_values.get(k)
        if keras_value:
            plt.axvline(x=keras_value, c='r')
        
       

    plt.show()

def runb(experiment_time="1572371338"):
    f = json.load(open("./experiments/results/{}.json".format(experiment_time)))
    
    models = []
    color_recovery =[]
    downscale_recovery =[]
    compress50_recovery =[]
    compress90_recovery =[]
    lpips = []
    ssim = []
    psnr = []
    

    for i, combination in enumerate(f):
        color_recovery.append(combination["color_depth_recovery"])
        downscale_recovery.append(combination["downscale_recovery"])
        compress50_recovery.append(combination["compress50_recovery"])
        compress90_recovery.append(combination["compress90_recovery"])

        lpips.append(combination["lpips"])
        ssim.append(combination["ssim"])
        psnr.append(combination["psnr"])

        models.append(combination["model"])

    color_recovery = np.array(color_recovery)
    downscale_recovery = np.array(downscale_recovery)
    compress50_recovery = np.array(compress50_recovery)
    compress90_recovery = np.array(compress90_recovery)

    tbl = lpips
    indices = np.arange(len(tbl))
    plt.scatter(indices,tbl,s=45)
    keras_value = keras_values.get("lpips")
    if keras_value:
        plt.axhline(y=keras_value, c='r')
    plt.title("LPIPS")
    plt.ylabel('Loss value')
    plt.xlabel('Model')
    plt.xticks(indices, [""]*len(tbl))
    plt.show()
    return 
    # boxplot([np.array(psnr)/100,np.array(lpips), np.array(ssim)],"Similarity metrics", ("psnr","lpips","ssim"))
    # plt.show()
    #return

    suffix = "recovery rate ({} models)".format(len(f))
    n_bins = 20
    m = 0
    mx = 1
    metrics = {"color_depth_recovery":("Post Color Depth Reduction",color_recovery),"compress50_recovery":("Post Compression (quality=50)",compress50_recovery),"compress90_recovery":("Post Compression (quality=90)",compress90_recovery)}
    #metrics = {"ssim":("SSIM",ssim),"lpips":("LPIPS",lpips),"psnr":("PSNR",psnr),**metrics}
    
    #metrics = {"compress90_recovery":("Post Compression (quality=90)",compress90_recovery)}
    plot_metrics(metrics, suffix)
    return 




def runc(experiment_time="1572371338"):
    f = json.load(open("./experiments/results/experimentSP9c/{}.json".format(experiment_time)))
    
    models = []
    decoding_recovery =[e["decoding_recovery"] for e in f][:100]

    plt.figure()
    title = "Decoding rate by third party models ({} models)".format(len(decoding_recovery))
    plt.title(title)
    n_bins = 20
    n,bins,edges = plt.hist(decoding_recovery, n_bins, weights=np.ones(len(decoding_recovery)) / len(decoding_recovery), color='#0504aa',
                            alpha=0.7, rwidth=0.9,edgecolor='k', cumulative=True)
    plt.grid(axis='y', alpha=0.75)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel('Decoding rate')
    plt.ylabel('Count')
    m = min(decoding_recovery)
    mx = max(decoding_recovery)
    plt.xlim(m,mx)
    b = bins+1/n_bins/2
    plt.xticks(b,np.around(b,2))

    plt.show()


if __name__ == "__main__":
    # runa(experiment_time="experimentSP9/1571050357 - Copie")
    # runa("experimentSP9/merged")
    # runb(experiment_time="experimentSP9b/merged")
    runc()
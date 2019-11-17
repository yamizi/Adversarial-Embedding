import os
import sys
import scipy
import logging
import numpy
import tempfile
import shutil
import numpy
import subprocess
import random

from scipy.io import savemat, loadmat
from scipy import misc
from PIL import Image

from aletheialib import utils

import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import cpu_count



M_BIN="octave -q --no-gui --eval"


# {{{ _embed()
def _embed(sim, path, payload, dst_path=None):
    
    payload = str(payload)
    for i in range(3):
        try:
            X=numpy.array([])

            im=Image.open(path)
            if (im.mode!='L' and sim in ["wow", "hugo", "hill", "s_uniward"]):
                print("Error,", sim, "must be used with grayscale images")
                return X

            currdir=os.path.dirname(__file__)
            basedir=os.path.abspath(os.path.join(currdir, os.pardir))
            m_path=os.path.join(basedir, 'external', 'octave')

            tmpdir=tempfile.mkdtemp()
            X_path=tmpdir+"/X.mat"

            m_code=""
            m_code+="cd "+tmpdir+";"
            m_code+="addpath('"+m_path+"');"
            m_code+="warning('off');"
            m_code+="pkg load image;"
            m_code+="pkg load signal;"

            if sim=='wow':
                m_code+="X=WOW('"+path+"',"+payload+");"
            elif sim=='hugo':
                m_code+="X=HUGO('"+path+"',"+payload+");"
            elif sim=='s_uniward':
                m_code+="X=S_UNIWARD('"+path+"',"+payload+");"
            elif sim=='hill':
                m_code+="X=HILL('"+path+"',"+payload+");"
            elif sim=='j_uniward':
                m_code+="J_UNIWARD('"+path+"',"+payload+",'"+dst_path+"');"
            elif sim=='j_uniward_color':
                m_code+="J_UNIWARD_COLOR('"+path+"',"+payload+",'"+dst_path+"');"
            elif sim=='nsf5':
                m_code+="NSF5('"+path+"',"+payload+",'"+dst_path+"');"
            elif sim=='nsf5_color':
                m_code+="NSF5_COLOR('"+path+"',"+payload+",'"+dst_path+"');"
            elif sim=='ebs':
                m_code+="EBS('"+path+"',"+payload+",'"+dst_path+"');"
            elif sim=='ebs_color':
                m_code+="EBS_COLOR('"+path+"',"+payload+",'"+dst_path+"');"
            elif sim=='ued':
                m_code+="UED('"+path+"',"+payload+",'"+dst_path+"');"
            elif sim=='ued_color':
                m_code+="UED_COLOR('"+path+"',"+payload+",'"+dst_path+"');"
            elif sim=='experimental':
                m_code+="X=EXPERIMENTAL('"+path+"',"+payload+");"
            elif sim=='jpeg_read_struct':
                m_code+="X=JPEG_READ_STRUCT('"+path+"');"

            if not dst_path:
                m_code+="save('-mat7-binary', '"+X_path+"','X');"
            m_code+="exit"

            p=subprocess.Popen(M_BIN+" \""+m_code+"\"", stdout=subprocess.PIPE, shell=True)
            #output, err = p.communicate()
            status = p.wait()

            if not dst_path:
                data=loadmat(X_path)
                X=data['X']
                shutil.rmtree(tmpdir)
                return X
             
            shutil.rmtree(tmpdir)
            return

        except:
            print("Error, retry!")
            continue
    return
# }}}

# {{{ _extract()
def _extract(extractor_name, path, params={}):
    fdir=os.path.dirname(__file__)
    basedir=os.path.abspath(os.path.join(fdir, os.pardir))
    m_path=os.path.join(basedir, 'external', 'octave')

    X=numpy.array([])
    im=Image.open(path)
    from aletheialib.feaext import FEAEXT_1CH, FEAEXT_3CH
    if ((im.mode=='L' and extractor_name in FEAEXT_1CH) or 
        (im.mode in ['RGB', 'RGBA', 'RGBX'] and extractor_name in FEAEXT_3CH)):
        tmpdir=tempfile.mkdtemp()
        try:
            os.chdir(tmpdir)
        except Exception as e:
            print("chdir:", str(e))

        channel = 1
        if "channel" in params:
            channel = params["channel"]

        data_path=tmpdir+"/data.mat"
        m_code=""
        m_code+="cd "+tmpdir+";"
        m_code+="addpath('"+m_path+"');"
        m_code+="warning('off');"
        m_code+="pkg load image;"
        m_code+="pkg load signal;"
        if extractor_name=="GFR":
            m_code+="data="+extractor_name+"('"+path+"'," \
                    +str(params["rotations"])+", "+str(params["quality"])+", "+str(channel)+");"
        else:
            m_code+="data="+extractor_name+"('"+path+"', "+str(channel)+");"
        m_code+="save('-mat7-binary', '"+data_path+"','data');"
        m_code+="exit"
        p=subprocess.Popen(M_BIN+" \""+m_code+"\"", stdout=subprocess.PIPE, shell=True)
        # output, err = p.communicate()
        status = p.wait()

        data=loadmat(data_path)
        shutil.rmtree(tmpdir)

        if extractor_name=="GFR":
            X = data["data"][0]
        else:
            for submodel in data["data"][0][0]:
                X = numpy.hstack((X,submodel.reshape((submodel.shape[1]))))

    else:
        print("Image mode/extractor not supported: ", im.mode, "/", extractor_name)
        print("")
        sys.stdout.flush()

    im.close()

    return X
# }}}

# {{{ _jpeg()
def _jpeg(fn_name, path):
    
    fn_names_with_return = ['jpeg_read_struct']

    for i in range(3):
        try:
            X=numpy.array([])
            im=Image.open(path)

            currdir=os.path.dirname(__file__)
            basedir=os.path.abspath(os.path.join(currdir, os.pardir))
            m_path=os.path.join(basedir, 'external', 'octave')

            tmpdir=tempfile.mkdtemp()
            X_path=tmpdir+"/X.mat"

            m_code=""
            m_code+="cd "+tmpdir+";"
            m_code+="addpath('"+m_path+"');"
            m_code+="warning('off');"
            m_code+="pkg load image;"
            m_code+="pkg load signal;"

            if fn_name=='jpeg_read_struct':
                m_code+="X=JPEG_READ_STRUCT('"+path+"');"

            if fn_name in fn_names_with_return:
                m_code+="save('-mat7-binary', '"+X_path+"','X');"
            m_code+="exit"

            p=subprocess.Popen(M_BIN+" \""+m_code+"\"", stdout=subprocess.PIPE, shell=True)
            #output, err = p.communicate()
            status = p.wait()

            if fn_name in fn_names_with_return:
                data=loadmat(X_path)
                X=data['X']
                shutil.rmtree(tmpdir)
                return X
             
            shutil.rmtree(tmpdir)
            return

        except:
            print("Error, retry!")
            continue
    return
# }}}



from pybooru import Danbooru
from dotenv import load_dotenv, find_dotenv
import os
import time
import json
import urllib.request
import threading
import cropy
import shutil
from tqdm import tqdm
from PIL import Image, ImageOps

load_dotenv(find_dotenv())
DL_DIRECTORY = os.environ.get("DL-DIRECTORY")
IMG_DIRECTORY = os.path.join(DL_DIRECTORY, 'img')
TEMPDIR = os.path.join(DL_DIRECTORY, 'temp')
TARGET_IMAGE_SIZE = (512, 512)

N = 100
client = Danbooru('danbooru')


def mapThreaded(func, *args):
    # perform threaded map
    threads = [threading.Thread(target=func, args=arg) for arg in zip(*args)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()


def getImgFiles():
    # List images besides index.json
    return [os.path.join(IMG_DIRECTORY, name)
            for name in os.listdir(IMG_DIRECTORY)]


def getPostList(N):
    return [p for p in client.post_list(tags='touhou 1girl', limit=N, random=True) if 'id' in p]


def initDir():
    # purge directory and make temp/ dir
    if not os.path.exists(TEMPDIR):
        os.makedirs(TEMPDIR)
    if not os.path.exists(IMG_DIRECTORY):
        os.makedirs(IMG_DIRECTORY)
    for name in os.listdir(DL_DIRECTORY):
        if os.path.isfile(os.path.join(DL_DIRECTORY, name)):
            os.remove(os.path.join(DL_DIRECTORY, name))
    for name in os.listdir(IMG_DIRECTORY):
        if os.path.isfile(os.path.join(IMG_DIRECTORY, name)):
            os.remove(os.path.join(IMG_DIRECTORY, name))


def exportToDirectory(postList):
    def downloadFile(post):
        try:
            urllib.request.urlretrieve(post['file_url'], os.path.join(TEMPDIR, str(post['id']) + '.jpg'))
        except KeyError:
            pass

    mapThreaded(downloadFile, postList)
    # with open(os.path.join(DL_DIRECTORY, 'index.json'), 'w') as f:
    #     json.dump(postList, f)
    return postList


def cropFaces(postList):
    def cropFile(source, target):
        cropy.detect(source, target)

    sourceUrls = [os.path.join(TEMPDIR, f"{str(post['id'])}.jpg") for post in postList]
    targetUrls = [os.path.join(IMG_DIRECTORY, f"{str(post['id'])}.jpg") for post in postList]

    mapThreaded(cropFile, sourceUrls, targetUrls)
    return [os.path.join(IMG_DIRECTORY, name)
            for name in os.listdir(IMG_DIRECTORY)
            if name != 'index.json' and os.path.isfile(os.path.join(IMG_DIRECTORY, name))]


def filterFiles(urls):
    def checkImage(url):
        img = Image.open(url)
        return img.size != (0, 0) and os.stat(url).st_size > 40000

    def filterImage(url):
        if os.path.exists(url) and not checkImage(url):
            os.remove(url)

    mapThreaded(filterImage, urls)
    return getImgFiles()


def upscaleImages(urls):
    def upscaleImage(url):
        ImageOps.fit(Image.open(url), (512, 512)).save(url)

    mapThreaded(upscaleImage, urls)
    return getImgFiles()


def pathFilename(path):
    return os.path.basename(path).split('.')[-2]


def writeJson(p):
    with open(os.path.join(DL_DIRECTORY, 'index.json'), 'w') as f:
        json.dump(p, f)


def cleanup():
    shutil.rmtree(TEMPDIR)


def main():
    postList = []
    # for _ in tqdm(range(N//100+2)):
    #     postList.extend(getPostList(N))
    postList = getPostList(N)

    n = len(postList)
    print(f"Downloading {n} posts from booru, preparing to download to {DL_DIRECTORY}...")
    initDir()
    start = time.time()
    exportToDirectory(postList)
    print(f'Done downloading files, elapsed time: {time.time() - start}')
    print('Cropping files..')
    urls = cropFaces(postList)
    n = len(urls)
    print(f'{n} cropped faces produced.\nNow filtering files...')
    urls = filterFiles(urls)
    print(f'{n - len(urls)} files filtered out. {len(urls)} files remaining\nUpscaling images...')
    n = len(urls)
    postList = [p for p in postList if p['id'] in [int(pathFilename(u)) for u in urls]]
    writeJson(postList)
    urls = upscaleImages(urls)
    print('Upscaled images.\nCleaning up...')
    cleanup()
    print('Fin')


main()

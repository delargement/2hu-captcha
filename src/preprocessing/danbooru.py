from pybooru import Danbooru
from dotenv import load_dotenv, find_dotenv
import os
import time
import json
import urllib.request
import threading
import cropy
from PIL import Image, ImageOps

load_dotenv(find_dotenv())
DL_DIRECTORY = os.environ.get("DL-DIRECTORY")
TEMPDIR = os.path.join(DL_DIRECTORY, 'temp')
TARGET_IMAGE_SIZE = (512, 512)

N = 30
client = Danbooru('safebooru')


def mapThreaded(func, *args):
    threads = [threading.Thread(target=func, args=arg) for arg in zip(*args)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()


def getImgFiles():
    return [os.path.join(DL_DIRECTORY, name)
            for name in os.listdir(DL_DIRECTORY)
            if name != 'index.json' and os.path.isfile(os.path.join(DL_DIRECTORY, name))]


def getPostList():
    return client.post_list(tags='touhou 1girl', limit=N, random=True)


def initDir():
    if not os.path.exists(TEMPDIR):
        os.makedirs(TEMPDIR)
    for name in os.listdir(DL_DIRECTORY):
        if os.path.isfile(os.path.join(DL_DIRECTORY, name)):
            os.remove(os.path.join(DL_DIRECTORY, name))


def exportToDirectory(postList):
    def downloadFile(post):
        try:
            urllib.request.urlretrieve(post['file_url'], os.path.join(TEMPDIR, str(post['id']) + '.jpg'))
        except KeyError:
            pass

    mapThreaded(downloadFile, postList)
    with open(os.path.join(DL_DIRECTORY, 'index.json'), 'w') as f:
        json.dump(postList, f)


def cropFaces(postList):
    def cropFile(source, target):
        cropy.detect(source, target)

    sourceUrls = [os.path.join(TEMPDIR, f"{str(post['id'])}.jpg") for post in postList]
    targetUrls = [os.path.join(DL_DIRECTORY, f"{str(post['id'])}.jpg") for post in postList]

    mapThreaded(cropFile, sourceUrls, targetUrls)
    return [os.path.join(DL_DIRECTORY, name)
            for name in os.listdir(DL_DIRECTORY)
            if name != 'index.json' and os.path.isfile(os.path.join(DL_DIRECTORY, name))]


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


def main():
    postList = getPostList()
    n = N
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
    urls = upscaleImages(urls)
    print('Upscaled images.')


main()

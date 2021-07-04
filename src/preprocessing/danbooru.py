from pybooru import Danbooru
from dotenv import load_dotenv, find_dotenv
import os
import time
import json
import urllib.request
import threading
import cropy

load_dotenv(find_dotenv())
DL_DIRECTORY = os.environ.get("DL-DIRECTORY")

N = 10
client = Danbooru('safebooru')


def getPostList():
    return client.post_list(tags='touhou 1girl', limit=N, random=True)


def exportToDirectory(postList):
    def downloadFile(post):
        urllib.request.urlretrieve(post['file_url'], os.path.join(DL_DIRECTORY, 'temp\\' + str(post['id']) + '.png'))

    threads = [threading.Thread(target=downloadFile, args=(post,)) for post in postList]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    with open(os.path.join(DL_DIRECTORY, 'index.json'), 'w') as f:
        json.dump(postList, f)


def cropFaces(postList):
    def cropFile(source, target):
        cropy.detect(source, target)

    sourceUrls = [os.path.join(DL_DIRECTORY, f"temp\\{str(post['id'])}.png") for post in postList]
    targetUrls = [os.path.join(DL_DIRECTORY, f"{str(post['id'])}.png") for post in postList]

    threads = [threading.Thread(target=cropFile, args=(s, t,)) for s, t in zip(sourceUrls, targetUrls)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()

def main():
    postList = getPostList()
    print(f"Downloaded {len(postList)} posts from booru, preparing to download to {DL_DIRECTORY}")
    start = time.time()
    exportToDirectory(postList)
    print(f'Done downloading files, elapsed time: {time.time() - start}')
    print('Preparing to crop files')
    cropFaces(postList)
    print('Done cropping files')

main()

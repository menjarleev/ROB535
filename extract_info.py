#! /usr/bin/python3
import numpy as np
from glob import glob
import csv


classes = np.loadtxt('classes.csv', skiprows=1, dtype=str, delimiter=',')
labels = classes[:, 2].astype(np.uint8)


def write_labels(path):
    files = glob('{}/*/*_bbox.bin'.format(path))
    files.sort()
    name = '{}/trainval_labels.csv'.format(path)
    with open(name, 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['guid/image', 'label'])

        for file in files:
            guid = file.split('/')[-2]
            idx = file.split('/')[-1].replace('_bbox.bin', '')

            bbox = np.fromfile(file, dtype=np.float32)
            bbox = bbox.reshape([-1, 11])
            found_valid = False
            for b in bbox:
                # ignore_in_eval
                if bool(b[-1]):
                    continue
                found_valid = True
                class_id = b[9].astype(np.uint8)
                label = labels[class_id]
            if not found_valid:
                label = 0
            
            writer.writerow(['{}/{}'.format(guid, idx), label])
    
    print('Wrote report file `{}`'.format(name))


def write_centroids(path):
    files = glob('{}/*/*_bbox.bin'.format(path))
    files.sort()
    name = '{}/trainval_centroids.csv'.format(path)

    with open(name, 'w') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(['guid/image/axis', 'value'])

        for file in files:
            guid = file.split('/')[-2]
            idx = file.split('/')[-1].replace('_bbox.bin', '')

            bbox = np.fromfile(file, dtype=np.float32)
            bbox = bbox.reshape([-1, 11])
            for b in bbox:
                # ignore_in_eval
                if bool(b[-1]):
                    continue
                xyz = b[3:6]
                for a, v in zip(['x', 'y', 'z'], xyz):
                    writer.writerow(['{}/{}/{}'.format(guid, idx, a), v])
    
    print('Wrote report file `{}`'.format(name))


if __name__ == '__main__':
    np.random.seed(0)
    for path in ['/home/ec2-user/deploy/trainval']:
        write_labels(path)
        write_centroids(path)
       
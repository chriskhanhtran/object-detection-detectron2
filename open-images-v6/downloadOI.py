# Author : Sunita Nayak, Big Vision LLC

#### Usage example: python3 downloadOI.py --classes 'Ice_cream,Cookie' --mode train

import argparse
import csv
import subprocess
import os
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool as thread_pool

cpu_count = multiprocessing.cpu_count()

parser = argparse.ArgumentParser(description="Download Class specific images from OpenImagesV4")
parser.add_argument("--mode", help="Dataset category - train, validation or test", required=True)
parser.add_argument("--classes", help="Names of object classes to be downloaded", required=True)
parser.add_argument("--nthreads", help="Number of threads to use", required=False, type=int, default=cpu_count * 2)
parser.add_argument("--occluded", help="Include occluded images", required=False, type=int, default=1)
parser.add_argument("--truncated", help="Include truncated images", required=False, type=int, default=1)
parser.add_argument("--groupOf", help="Include groupOf images", required=False, type=int, default=1)
parser.add_argument("--depiction", help="Include depiction images", required=False, type=int, default=1)
parser.add_argument("--inside", help="Include inside images", required=False, type=int, default=1)

args = parser.parse_args()

run_mode = args.mode

threads = args.nthreads

classes = []
for class_name in args.classes.split(","):
    classes.append(class_name)

# Read `class-descriptions-boxable.csv`
with open("./class-descriptions-boxable.csv", mode="r") as infile:
    reader = csv.reader(infile)
    dict_list = {rows[1]: rows[0] for rows in reader}  # rows[1] is ClassName, rows[0] is ClassCode

subprocess.run(["rm", "-rf", run_mode])
subprocess.run(["mkdir", run_mode])

pool = thread_pool(threads)
commands = []
cnt = 0

for ind in range(0, len(classes)):
    class_name = classes[ind]
    print("Class " + str(ind) + " : " + class_name)

    command = "grep " + dict_list[class_name.replace("_", " ")] + " ./" + run_mode + "-annotations-bbox.csv"
    class_annotations = subprocess.run(command.split(), stdout=subprocess.PIPE).stdout.decode("utf-8")
    class_annotations = class_annotations.splitlines()

    for line in class_annotations:
        line_parts = line.split(",")
        img_id = line_parts[0]
        save_path = os.path.join(run_mode, img_id + ".jpg")

        # If image exists, skip
        if os.path.exists(save_path):
            continue

        # Download options: IsOccluded, IsTruncated, IsGroupOf, IsDepiction, IsInside
        if args.occluded == 0 and int(line_parts[8]) > 0:
            print("Skipped %s", img_id)
            continue
        if args.truncated == 0 and int(line_parts[9]) > 0:
            print("Skipped %s", img_id)
            continue
        if args.groupOf == 0 and int(line_parts[10]) > 0:
            print("Skipped %s", img_id)
            continue
        if args.depiction == 0 and int(line_parts[11]) > 0:
            print("Skipped %s", img_id)
            continue
        if args.inside == 0 and int(line_parts[12]) > 0:
            print("Skipped %s", img_id)
            continue

        # Command to download
        command = f"aws s3 --no-sign-request --only-show-errors cp s3://open-images-dataset/'{run_mode}'/'{img_id}'.jpg {save_path}"
        commands.append(command)
        cnt += 1

print("Annotation Count : " + str(cnt))
commands = list(set(commands))
print("Number of images to be downloaded : " + str(len(commands)))

list(tqdm(pool.imap(os.system, commands), total=len(commands)))

pool.close()
pool.join()

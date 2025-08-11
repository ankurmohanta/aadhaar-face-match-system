import os

import random

import csv

def create_negative_pairs(data_folder, output_csv, num_pairs=461):

    person_dirs = [os.path.join(data_folder, d) for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]

    # Extract one image from each person folder

    person_images = {}

    for path in person_dirs:

        images = sorted([

            os.path.join(path, f)

            for f in os.listdir(path)

            if f.lower().endswith(('.jpg', '.jpeg', '.png'))

        ])

        if len(images) >= 2:

            person_images[path] = images

    keys = list(person_images.keys())

    pairs = set()

    while len(pairs) < num_pairs:

        a, b = random.sample(keys, 2)

        img1 = person_images[a][0]

        img2 = person_images[b][1]

        pairs.add((img1, img2, 0))

    # Write CSV

    with open(output_csv, 'w', newline='') as f:

        writer = csv.writer(f)

        writer.writerow(['img1', 'img2', 'label'])

        writer.writerows(list(pairs))

    print(f"✅ Saved {len(pairs)} negative test pairs to: {output_csv}")

# Usage

if __name__ == "__main__":

    create_negative_pairs(

        data_folder="/workspace/approaches/Glintr100_retrain_siamese_classifier/cropped_faces_15-19.06.25",

        output_csv= "/workspace/approaches/Glintr100_retrain_siamese_classifier/negative_15-19.06.25_testpairs.csv",

        num_pairs=461

    )
 
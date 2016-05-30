#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import cv2
import os
import re

height_std = 40.0
chars = [chr(x) for x in range(32, 127)]

def parse_bbox(image_full_path, text_path):
    print image_full_path, text_path
    labels = []
    features = []
    with open(text_path, 'r')  as f:
        for line in f.readlines():
            try:
                word = line.split('\n')[0]
                print word

                # process word
                word = np.array([ord(c) for c in word])
                if np.max(word) > ord(chars[-1]) or np.min(word) < ord(chars[0]):
                    print np.min(word), np.max(word), ord(chars[0]), ord(chars[-1])
                    print "Invalid chars"
                    continue
                else:
                    print "Valid chars"
                word = word - ord(chars[0]);

                # process image
                img = cv2.imread(image_full_path, 0)
                new_width = int(height_std / img.shape[0] * img.shape[1])
                sub = cv2.resize(img, (new_width, int(height_std)))

                features.append(sub)
                labels.append(word.tolist())
            except Exception as e:
                print(e)
                print("parse wrong, skip")
    return features, labels

if __name__ == "__main__":
    images_dir = os.path.expanduser('~/GitHub/clstm/IAM/clstm_data/')
    boxes_dir = os.path.expanduser("~/GitHub/clstm/IAM/clstm_data/")
    #images_dir  = os.path.expanduser('./test/')
    #boxes_dir = os.path.expanduser('./test/')
    out_imgs_dir = os.path.expanduser('./imgs/')
    out_train_imgs_list_path = os.path.expanduser('./train_img_list.txt')
    out_test_imgs_list_path = os.path.expanduser('./test_img_list.txt')

    features_all = []
    labels_all = []
    for root, dirs, files in os.walk(images_dir):
        for i, file in enumerate(files):
            if file.endswith('.png'):
                print("processing {}({}/{})".format(file, i + 1, len(files)))
                image_full_path = os.path.join(root, file)
                box_full_path = os.path.join(boxes_dir, file[0:-4] + '.gt.txt')
                features, labels = parse_bbox(image_full_path, box_full_path)
                labels_all.extend(labels)
                features_all.extend(features)

    print("total samples: {}".format(len(labels_all)))
    idxs = np.random.permutation(len(features_all))
    n_train_samples = int(0.9 * len(features_all))
    print("n_train_samples: {}, n_test_samples{}".format(n_train_samples, len(features_all) - n_train_samples))

    # for training
    print("saving to {}".format(out_train_imgs_list_path))
    with open(out_train_imgs_list_path, 'wb') as f:
        f.write("32 127\n") # chars range
        for i in idxs[:n_train_samples]:
            idx = idxs[i]
            img_save_path = "{}.jpg".format(i)
            img_save_full_path = os.path.join(out_imgs_dir, img_save_path)
            cv2.imwrite(img_save_full_path, features_all[idx])
            labels_str = [str(c) for c in labels_all[idx]]
            print labels_str
            record = img_save_path + " " + " ".join(labels_str) + "\n"
            f.write(record)

    # for testing
    print("saving to {}".format(out_test_imgs_list_path))
    with open(out_test_imgs_list_path, 'wb') as f:
        f.write("32 127\n") # chars range
        for i in idxs[n_train_samples:]:
            idx = idxs[i]
            img_save_path = "{}.jpg".format(i)
            img_save_full_path = os.path.join(out_imgs_dir, img_save_path)
            cv2.imwrite(img_save_full_path, features_all[idx])
            labels_str = [str(c) for c in labels_all[idx]]
            record = img_save_path + " " + " ".join(labels_str) + "\n"
            f.write(record)

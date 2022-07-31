from email.mime import image
import streamlit as st
import cv2
import os
from PIL import Image, ImageEnhance
import pixellib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pixellib.instance import instance_segmentation
from pixellib.tune_bg import alter_bg


@st.cache
def load_image(img):
    im = Image.open(img)


def main():

    st.title(" Nybsys Project | Blurring App ")
    st.text("Built with PixelLib and Streamlit")

    image_file = st.file_uploader("Upload Image", type=["jpeg"])

    if image_file is not None:
        file_path = './'
        file_name = 'sample.jpeg'
        our_image = Image.open(image_file)
        cv2.imwrite(os.path.join(file_path, file_name), np.asarray(our_image))
        st.text('Original Image')
        st.image(our_image)
        seg = instance_segmentation()
        seg.load_model("mask_rcnn_coco.h5")
        segmask, output = seg.segmentImage('./sample.jpeg', show_bboxes=True, output_image_name="output.jpg",
                                           extract_segmented_objects=True, save_extracted_objects=False)
        st.text('Detected Object with Mask')
        masked_image = Image.open('./output.jpg')
        st.image(masked_image)
        df = pd.read_csv('coco_labels.csv')
        labels = list(segmask['class_ids'])
        class_names = []
        for label in labels:
            filt = (df['Class_id'] == label)
            class_names.append(df.loc[filt, 'Class_Name'].values[0])


if __name__ == '__main__':
    main()

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
import warnings
warnings.filterwarnings("ignore")


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
        cv2.imwrite(os.path.join(file_path, file_name),
                    cv2.cvtColor(np.asarray(our_image), cv2.COLOR_BGR2RGB))

        st.subheader('Original Image')
        st.image(our_image)
        seg = instance_segmentation()
        seg.load_model("mask_rcnn_coco.h5")
        segmask, output = seg.segmentImage('./sample.jpeg', show_bboxes=True, output_image_name="output.jpg",
                                           extract_segmented_objects=True, save_extracted_objects=False)

        st.subheader('Detected Object with Mask')
        masked_image = Image.open('./output.jpg')
        st.image(masked_image)
        df = pd.read_csv('coco_labels.csv')
        labels = list(segmask['class_ids'])
        class_names = []
        for label in labels:
            filt = (df['Class_id'] == label)
            class_names.append(df.loc[filt, 'Class_Name'].values[0])
        class_count_dict = dict((i, labels.count(i)) for i in labels)
        num_objects = len(segmask['class_ids'])
        lookup_table = []

        for i in range(0, num_objects):
            current_label = labels[i]
            cv2.imwrite(os.path.join('./', class_names[i]+str(
                class_count_dict[current_label])+'.jpg'), segmask['extracted_objects'][i])
            lookup_table.append(
                class_names[i]+str(class_count_dict[current_label])+'.jpg')
            class_count_dict[current_label] = class_count_dict[current_label]-1

        # Show Extracted Objects

        st.subheader('Extracted Objects')
        for filename in os.listdir('./'):
            f = os.path.join('./', filename)
            # checking if it is a file
            if os.path.isfile(f) and filename != 'output.jpg' and filename != 'sample.jpeg' and filename != 'blur_img.jpg' and filename.endswith('.jpg'):
                extracted_objects = Image.open(filename)
                st.image(extracted_objects, caption=filename.split('.')[0])

        st.subheader('Blurring')

        options_to_select = [element.split('.')[0] for element in lookup_table]

        option = st.selectbox(
            'Select the type of object you want to keep unblurred', tuple(options_to_select
                                                                          ))
        st.write('You selected:', option)

        target_object = str(option)+'.jpg'
        filt2 = (df['Class_id'] == segmask['class_ids']
                 [lookup_table.index(target_object)])
        detect_class = df.loc[filt2, 'Class_Name'].values[0]
        change_bg = alter_bg(model_type="pb")
        change_bg.load_pascalvoc_model("xception_pascalvoc.pb")
        change_bg.blur_bg("sample.jpeg", low=True,
                          output_image_name="blur_img.jpg", detect=detect_class)

        blurred_image = Image.open('./blur_img.jpg')

        st.image(blurred_image)


if __name__ == '__main__':
    main()

import streamlit as st
import face_recognition
import cv2
import numpy as np
from PIL import Image


def converter(image_file):
    image = Image.open(image_file)
    image = np.array(image)
    return image

def train(faces_data):
    face_names = []
    face_encodings = []
    for face_name in faces_data:
        face_names.append(face_name)

        image = faces_data[face_name]
        encoding = face_recognition.face_encodings(image)[0]
        face_encodings.append(encoding)

    return [face_names, face_encodings]


def predict(persons_data, unknown_image):
    persons, persons_encodings = persons_data
    rgb_unknown_image = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_unknown_image)
    face_encodings = face_recognition.face_encodings(rgb_unknown_image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(persons_encodings, face_encoding)

        face_distances = face_recognition.face_distance(persons_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = persons[best_match_index]
        else:
            continue

        # Draw a box around the face
        cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 0, 255), 0)

        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(unknown_image, name, (left, top - 40), font, 1.0, (5, 0, 255), 1)

    return unknown_image
    # show the images


def main():
    st.title("Face recognition simple:")


    faces_data = {}

    face_image = st.file_uploader("Face image", type=['jpg', 'png', 'jpeg'], key = 0)
    if not face_image:
        return None
    face_name = st.text_input("Face name")
    faces_data[face_name] = converter(face_image)

    persons_data = train(faces_data)

    unknow_face_image = st.file_uploader("Recognize", type=['jpg', 'png', 'jpeg'], key="unknown")
    if not unknow_face_image:
        return None
    
    rec_image = predict(persons_data, converter(unknow_face_image))

    st.image(rec_image)


if __name__ == "__main__":
    main()

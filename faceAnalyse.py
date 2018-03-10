#!/usr/bin/env python
# -*- coding: utf-8 -*-

import face_recognition
import cv2
import os
import numpy as np
import json


class FaceDetector:
    def __init__(self):
        self.json_file_path = "data/known_faces.json"
        self.load_face()
        self.known_faces = {}
        self.tolerance = 0.9

    # jsonファイルから既知の顔のデータを読み込む
    def load_face(self):
        try:
            f = open(self.json_file_path, 'r')
            json_data = json.load(f)
            self.known_faces.update(json_data)
        except:
            pass

    def load_image_from_folder(self, path):
        for file in face_recognition.image_files_in_folder(path):
            image = face_recognition.load_image_file(file)
            basename = os.path.splitext(os.path.basename(file))[0]
            self.record_face(image, basename)

    def record_face(self, image, name):
        # faces dict empty check
        if not bool(self.known_faces):
            info = self.get_info_from_image(image, name)
            self.known_faces["faces"] = [info]
            self.save_to_json()
        else:
            # 情報が登録されているかチェック
            if (not name in [face_data["name"] for face_data in self.known_faces["faces"]]):
                info = self.get_info_from_image(image, name)
                self.known_faces["faces"].append(info)
                self.save_to_json()

    @staticmethod
    def get_info_from_image(image, name):
        encoding = face_recognition.face_encodings(image)[0]
        info = {
            "name": name,
            "encoding": encoding.tolist()
        }
        return info


    def save_to_json(self):
        dict = self.known_faces
        f = open(self.json_file_path, 'w')
        json.dump(dict, f, indent=4)


    # 画像を渡すと、顔の場所と顔の名前が入ったdictを返す
    def analyze_faces_in_image(self, image):
        encodings = face_recognition.face_encodings(image)
        locations = face_recognition.face_locations(image)
        faces = {}
        faces["face"] = []

        if len(self.known_faces) == 0:
            return

        if len(encodings) != 0:
            for encoding, location in zip(encodings, locations):
                compare_encodings = [face["encoding"] for face in self.known_faces["faces"]]
                distance = face_recognition.face_distance(compare_encodings, encoding)
                result = list(distance <= self.tolerance)  # しきい値でふるいにかける
                if True in result:
                    assert isinstance(distance, np.ndarray)
                    name = [face["name"] for face in self.known_faces["faces"]][distance.argmin()]
                    info = {
                        "name": name,
                        "info": {
                            "location": location,
                            "distance": min(distance)
                        }
                    }
                    faces["face"].append(info)
        print(faces)
        return faces

    def draw_rect(self, image, faces):
        assert isinstance(faces, dict)
        if len(faces["face"]) == 0:
            return image

        for f in faces["face"]:
            location = f["info"]["location"]
            cv2.rectangle(image, (location[3], location[0]), (location[1], location[2]), (0, 0, 255), 2)
            cv2.rectangle(image, (location[3], location[2]-25), (location[1], location[2]), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            prob = 100 - round(f["info"]["distance"], 3) * 100
            text = f["name"] + ":" + str(prob)
            cv2.putText(image, text, (location[3]+6, location[2]-6), font, 0.5, (255, 255, 255), 1)
        return image
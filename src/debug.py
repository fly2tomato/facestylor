import dlib
import os

pwd = os.getcwd()
predictor_file = pwd + '/work_dirs/lite-weights/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_file)
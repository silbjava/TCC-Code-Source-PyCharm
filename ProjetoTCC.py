import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import serial
import time

#Envia informação de sinais pela serial
def envSerial():
    time.sleep(.5)
    conex.write(info)
    atualInfo = info

def impControle():
    time.sleep(.5)
    ctrl = conex.readline().decode('utf8')
    return ctrl

#Mediapipe config
smaos = mp.solutions.hands
maos = smaos.Hands(max_num_hands=1, model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)

atualInfo = ''     #Variavel que salva informação atual

cam_cap = cv2.VideoCapture(0)


sinais = ['D', 'L', 'T', '1', '2', '3', 'FUNDO', 'A', 'B']
modelo = load_model('keras_model.h5')
dados = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
try:
    conex = serial.Serial('COM5', 9600, timeout=0.5)
    time.sleep(.5)
    conex.flushInput()
    print("Porta", conex.portstr, "Conectada")

except serial.SerialException:
    print("Porta USB não detectada!")
    pass

while True:
    visivel, imagem = cam_cap.read()
    if not visivel:
        break

    imagem.flags.writeable = False
    quadroRGB = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
    resultado = maos.process(quadroRGB)
    pontosMaos = resultado.multi_hand_landmarks
    a, l, _ = imagem.shape

    if pontosMaos != None:          # BoundBox em torno da mão
        for mao in pontosMaos:
            x_max = 0
            y_max = 0
            x_min = l
            y_min = a
            for lm in mao.landmark:
                x, y = int(lm.x * l), int(lm.y * a)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            cv2.rectangle(imagem, (x_min-50, y_min-50), (x_max+50, y_max+50), (0, 255, 0), 2)

            try:
                corteImg = imagem[y_min-50:y_max+50, x_min-50:x_max+50]
                corteImg = cv2.resize(corteImg, (224, 224))
                vetorImg = np.asarray(corteImg)
                vetor_img_normalizada = (vetorImg.astype(np.float32) / 127.0) - 1
                dados[0] = vetor_img_normalizada
                previsao = modelo.predict(dados)             # Predicão do Modelo
                valorInd = np.argmax(previsao)              # Valor maximo da predicao
                cv2.putText(imagem, sinais[valorInd], (x_min-50, y_min-65), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 50), 2)
                info = sinais[valorInd].encode()

                if (info != '' and atualInfo != info):
                    try:
                        envSerial()
                        print(impControle())
                    except:
                        continue

            except:
                continue

    cv2.imshow('Captura de Imagem', imagem)
    chave = cv2.waitKey(1)
    if chave == 27:
        break
cam_cap.release()
cv2.destroyAllWindows()
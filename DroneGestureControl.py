# Script para controle de drone DJI Tello via gestos manuais.
# Utiliza MediaPipe para detecção de mãos e OpenCV para processamento de imagem e faces.
# Funcionalidades: Decolagem, Pouso, Controle por Gestos, "Me Siga", "Scan de Rostos" e Captura de Fotos.
# autor: Hugo Gomes de la Fuente, 2025

# coding: utf-8
import cv2
import mediapipe as mp
from djitellopy import Tello
import time
import numpy as np
import os
from enum import Enum

# --- Estados do Programa ---
class EstadoPrograma(Enum):
    INICIALIZANDO = 0     # Conexão e preparação inicial.
    MENU = 1              # Menu principal para seleção de modos.
    CONTROLE_GESTAO = 2   # Controle direto do drone por gestos.
    MODO_COMANDO = 3      # Modos especiais (Me Siga, Scan de Rostos).

estado_atual_programa = EstadoPrograma.INICIALIZANDO

# --- Constantes Globais ---
# Comandos e Detecção
RC_SPEED = 25                   # Velocidade para comandos RC (0-100).
FINGER_UP_Y_OFFSET = 0.02       # Limiar normalizado para dedo "levantado".
ANGULO_ROTACAO_GRAUS = 45       # Ângulo de rotação para Scan de Rostos.

# Modo "Me Siga"
FOLLOW_SPEED_FB = 20                   # Velocidade frente/trás.
FOLLOW_SPEED_UD = 25                   # Velocidade cima/baixo.
FOLLOW_YAW_SPEED = 30                  # Velocidade de rotação (yaw).
CENTER_THRESHOLD_X_RATIO = 0.15        # Tolerância horizontal (% da largura) para centralização.
CENTER_THRESHOLD_Y_RATIO = 0.15        # Tolerância vertical (% da altura) para centralização.
DESIRED_FACE_WIDTH_SCREEN_RATIO = 0.15 # Proporção desejada do rosto na tela (distância).
FACE_LOST_HOVER_TIMEOUT_S = 3.0        # Tempo (s) pairando se perder o rosto.
FOLLOW_SEARCH_HOVER_DURATION_S = 2.0   # Duração (s) pairando na busca inicial.
FOLLOW_SEARCH_ROTATE_ANGLE_DEG = 45    # Ângulo (graus) da rotação na busca inicial.
FOLLOW_SEARCH_MAX_ROTATIONS = (360 // FOLLOW_SEARCH_ROTATE_ANGLE_DEG) + 2 # Máx. rotações na busca.
FOLLOW_SEARCH_STABILIZE_TIME_S = 0.5   # Tempo (s) para estabilizar após rotação na busca.


# Decolagem, Pouso, Foto
ATRASO_FINAL_ANTES_DECOLAGEM_SEGUNDOS = 3.0      # Contagem regressiva (s) decolagem inicial.
DURACAO_MAO_FECHADA_PARA_DECOLAGEM_S = 1.5       # Tempo (s) mão fechada para iniciar decolagem.
DURACAO_MAO_ABERTA_PARA_DECOLAGEM_MENU_S = 2.0   # Tempo (s) mão aberta para decolar do menu.
ATRASO_FINAL_ANTES_DECOLAGEM_MENU_SEGUNDOS = 3.0 # Contagem regressiva (s) decolagem do menu.
DURACAO_MAO_ABERTA_PARA_POUSAR_S = 1.5           # Tempo (s) mão aberta para confirmar pouso.
DURACAO_GESTO_FOTO_S = 1.0                       # Tempo (s) para segurar gesto de foto.


# --- Configurações do MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# IDs dos landmarks para detecção de dedos.
tip_ids = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
           mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
           mp_hands.HandLandmark.PINKY_TIP]
ref_joint_ids = [mp_hands.HandLandmark.THUMB_IP,
                 mp_hands.HandLandmark.INDEX_FINGER_MCP,
                 mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                 mp_hands.HandLandmark.RING_FINGER_MCP,
                 mp_hands.HandLandmark.PINKY_MCP]
finger_names = ["POLEGAR", "INDICADOR", "MEDIO", "ANELAR", "MINIMO"]

# --- Carregar Classificador Haar Cascade para Faces ---
haar_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
if not os.path.exists(haar_cascade_path):
    print("ALERTA: haarcascade_frontalface_default.xml não encontrado no caminho padrão do OpenCV.")
    # haar_cascade_path = 'haarcascade_frontalface_default.xml' 

face_cascade = None
if os.path.exists(haar_cascade_path):
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    print(f"Classificador de faces carregado de: {haar_cascade_path}")
else:
    print(f"ALERTA: Arquivo haarcascade_frontalface_default.xml NÃO encontrado.")
    print("Funcionalidades de detecção de face (Me Siga, Scan) estarão desabilitadas.")

# --- Inicialização da Câmera do Notebook ---
cap_notebook = cv2.VideoCapture(0)
if not cap_notebook.isOpened(): print("Erro: Não foi possível abrir a câmera do notebook."); exit()

# --- Inicialização e Conexão com o Drone Tello ---
tello = Tello()
drone_connected = False
frame_reader = None
battery_level = 0
drone_operational_status = "Aguardando conexao drone..."

# --- Variáveis de Controle ---
decolagem_executada = False
timestamp_inicio_atraso_decolagem = 0
mao_fechada_confirmada_para_decolagem = False
timestamp_mao_fechada_decolagem_detectada = 0

mao_aberta_confirmada_para_decolagem_menu = False
timestamp_mao_aberta_decolagem_menu_detectada = 0
timestamp_inicio_atraso_decolagem_menu = 0

last_keep_alive_time = time.time()
KEEP_ALIVE_INTERVAL = 0.1

feedback_overlay_text = ""
timestamp_feedback_overlay_end = 0
DURACAO_PADRAO_OVERLAY_S = 7.0
DURACAO_FOTO_OVERLAY_S = 3.0

scan_ativo_global = False
scan_fase_atual = "ocioso"
scan_angulo_corrente = 0
scan_total_rostos_acumulado = 0
scan_rostos_nesta_etapa = 0
controle_gestos_armado = False
timestamp_mao_fechada_detectada = 0
DURACAO_MAO_FECHADA_PARA_ARMAR_S = 1.0

follow_me_active = False
timestamp_last_face_seen_follow = 0
follow_me_target_acquired_this_session = False
follow_me_is_searching = False
follow_me_search_phase = "hover" 
timestamp_follow_search_phase_start = 0
follow_me_search_rotation_count = 0


timestamp_mao_aberta_pousar_menu_detectada = 0
timestamp_mao_aberta_pousar_controle_detectada = 0
timestamp_mao_aberta_pousar_comando_detectada = 0

timestamp_gesto_foto_detectado_controle = 0
foto_confirmada_para_disparo_controle = False

# Textos de display
comando_geral_display = drone_operational_status
info_modo_display = "Modo: Inicializando..."

# Conexão inicial com o drone Tello.
try:
    print("Conectando ao Tello..."); tello.connect(); drone_connected = True
    drone_operational_status = "Drone conectado. Aguardando video..."; print(drone_operational_status)
    tello.set_speed(30); battery_level = tello.get_battery(); print(f"Bateria: {battery_level}%")
    tello.streamoff(); tello.streamon(); frame_reader = tello.get_frame_read()
    if frame_reader is None: raise Exception("Nao foi possivel obter o frame_reader.")
    drone_operational_status = f"Video OK. Mostre MAO FECHADA ({DURACAO_MAO_FECHADA_PARA_DECOLAGEM_S:.1f}s) p/ decolar."
    print("Stream Tello OK.")
except Exception as e:
    error_msg = f"Falha Tello: {str(e)[:100]}"; print(error_msg)
    drone_operational_status = error_msg; drone_connected = False
print("Pressione 'q' para sair.")
last_battery_check_time = time.time()
battery_check_interval = 5

# --- Loop Principal ---
while True:
    current_time = time.time()

    # Captura e processamento da imagem da câmera do notebook (para gestos)
    success_notebook, image_notebook = cap_notebook.read()
    image_notebook_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    results_hand = None

    if success_notebook:
        image_notebook_rgb = cv2.cvtColor(image_notebook, cv2.COLOR_BGR2RGB)
        results_hand = hands.process(image_notebook_rgb)
        image_notebook_bgr = cv2.cvtColor(image_notebook_rgb, cv2.COLOR_RGB2BGR)
    else:
        cv2.putText(image_notebook_bgr, "Camera Notebook Falhou", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # Captura da imagem da câmera do drone
    frame_drone_local = None
    if drone_connected and frame_reader:
        try:
            frame_drone_local = frame_reader.frame
        except Exception:
            pass 

    # Lógica de Detecção de Gestos da Mão
    hand_info_text = "Nenhuma mao detectada"
    current_gesture_tuple = None

    if results_hand and results_hand.multi_hand_landmarks and results_hand.multi_handedness:
        for hand_idx, hand_landmarks in enumerate(results_hand.multi_hand_landmarks):
            if success_notebook:
                mp_drawing.draw_landmarks(image_notebook_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            handedness = results_hand.multi_handedness[hand_idx].classification[0].label
            fingers_status = [0] * 5 # 0=fechado, 1=aberto
            # Verifica se cada dedo está levantado comparando coordenada Y da ponta com junta de referência.
            if hand_landmarks.landmark[tip_ids[0]].y < hand_landmarks.landmark[ref_joint_ids[0]].y - FINGER_UP_Y_OFFSET:
                fingers_status[0] = 1
            for i in range(1, 5):
                if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[ref_joint_ids[i]].y - FINGER_UP_Y_OFFSET:
                    fingers_status[i] = 1
            dedos_levantados_nomes = [finger_names[i] for i, status in enumerate(fingers_status) if status == 1]
            if not dedos_levantados_nomes: hand_info_text = f"{handedness}: Mao Fechada"
            else: hand_info_text = f"{handedness}: {', '.join(dedos_levantados_nomes)}"
            current_gesture_tuple = tuple(fingers_status)
            break
    
    # --- Máquina de Estados Principal ---
    if estado_atual_programa == EstadoPrograma.INICIALIZANDO:
        info_modo_display = "MODO: INICIALIZANDO DRONE"
        # Reseta flags de confirmação e timers ao (re)iniciar.
        timestamp_mao_aberta_pousar_menu_detectada = 0
        timestamp_mao_aberta_pousar_controle_detectada = 0
        timestamp_mao_aberta_pousar_comando_detectada = 0
        timestamp_gesto_foto_detectado_controle = 0
        foto_confirmada_para_disparo_controle = False
        follow_me_is_searching = False 
        if not drone_connected or frame_reader is None:
            comando_geral_display = drone_operational_status
        elif not decolagem_executada: # Decolagem inicial.
            if not mao_fechada_confirmada_para_decolagem: # Aguarda confirmação do gesto de mão fechada.
                if not drone_operational_status.startswith("Video OK. Mostre MAO FECHADA"):
                    drone_operational_status = f"Video OK. Mostre MAO FECHADA ({DURACAO_MAO_FECHADA_PARA_DECOLAGEM_S:.1f}s) p/ decolar."
                if current_gesture_tuple == (0,0,0,0,0): # Mão fechada.
                    if timestamp_mao_fechada_decolagem_detectada == 0:
                        timestamp_mao_fechada_decolagem_detectada = current_time
                        print(f"INFO: Mao fechada detectada p/ decolagem. Mantenha por {DURACAO_MAO_FECHADA_PARA_DECOLAGEM_S:.1f}s.")
                        drone_operational_status = f"Confirmando mao fechada..."
                    elif current_time - timestamp_mao_fechada_decolagem_detectada >= DURACAO_MAO_FECHADA_PARA_DECOLAGEM_S:
                        mao_fechada_confirmada_para_decolagem = True
                        timestamp_inicio_atraso_decolagem = current_time
                        drone_operational_status = f"Mao confirmada. Decolar em {ATRASO_FINAL_ANTES_DECOLAGEM_SEGUNDOS:.0f}s..."
                        print(drone_operational_status)
                        timestamp_mao_fechada_decolagem_detectada = 0
                else: # Gesto não é mão fechada.
                    if timestamp_mao_fechada_decolagem_detectada != 0:
                        print("INFO: Mao nao esta (mais) fechada para decolagem. Resetando.")
                        drone_operational_status = f"Video OK. Mostre MAO FECHADA ({DURACAO_MAO_FECHADA_PARA_DECOLAGEM_S:.1f}s) p/ decolar."
                    timestamp_mao_fechada_decolagem_detectada = 0

            if mao_fechada_confirmada_para_decolagem: # Gesto confirmado, contagem regressiva para decolar.
                tempo_restante_decolagem = (timestamp_inicio_atraso_decolagem + ATRASO_FINAL_ANTES_DECOLAGEM_SEGUNDOS) - current_time
                if tempo_restante_decolagem <= 0: # Decolar
                    if drone_connected and not tello.is_flying:
                        try:
                            drone_operational_status = "Decolando agora..."
                            print(drone_operational_status); tello.takeoff(); time.sleep(2)
                            drone_operational_status = "Drone em voo."
                            print(drone_operational_status)
                            decolagem_executada = True; last_keep_alive_time = time.time()
                            estado_atual_programa = EstadoPrograma.MENU
                        except Exception as e:
                            error_msg = f"Erro decolagem: {str(e)[:50]}"; print(error_msg)
                            drone_operational_status = error_msg
                            mao_fechada_confirmada_para_decolagem = False
                    elif drone_connected and tello.is_flying:
                        drone_operational_status = "Drone ja voando."; print(drone_operational_status)
                        decolagem_executada = True; last_keep_alive_time = time.time()
                        estado_atual_programa = EstadoPrograma.MENU
                    else:
                        drone_operational_status = "Decolagem cancelada (nao conectado)."
                        mao_fechada_confirmada_para_decolagem = False
                elif not drone_operational_status.startswith("Decolando agora..."):
                    drone_operational_status = f"Mao confirmada. Decolar em {max(0, tempo_restante_decolagem):.0f}s..."
            comando_geral_display = drone_operational_status
        else: # Decolagem inicial já executada.
            estado_atual_programa = EstadoPrograma.MENU
            comando_geral_display = drone_operational_status

    elif estado_atual_programa == EstadoPrograma.MENU:
        info_modo_display = "MODO: MENU PRINCIPAL"
        follow_me_active = False; scan_ativo_global = False; follow_me_is_searching = False
        timestamp_mao_aberta_pousar_controle_detectada = 0
        timestamp_mao_aberta_pousar_comando_detectada = 0
        timestamp_gesto_foto_detectado_controle = 0 
        foto_confirmada_para_disparo_controle = False
        action_taken_this_menu_cycle = False

        # Lógica para decolar do MENU (drone pousado, decolagem inicial já feita).
        if drone_connected and not tello.is_flying and decolagem_executada:
            if mao_aberta_confirmada_para_decolagem_menu:
                tempo_restante_decolagem_menu = (timestamp_inicio_atraso_decolagem_menu + ATRASO_FINAL_ANTES_DECOLAGEM_MENU_SEGUNDOS) - current_time
                if tempo_restante_decolagem_menu <= 0:
                    try:
                        comando_geral_display = "MENU: Decolando agora..."
                        print(comando_geral_display); tello.takeoff(); time.sleep(2)
                        drone_operational_status = "Drone em voo."
                        print(drone_operational_status)
                        last_keep_alive_time = time.time()
                    except Exception as e:
                        error_msg = f"Erro decolagem (Menu): {str(e)[:50]}"; print(error_msg)
                        drone_operational_status = error_msg
                    comando_geral_display = drone_operational_status
                    mao_aberta_confirmada_para_decolagem_menu = False
                    timestamp_mao_aberta_decolagem_menu_detectada = 0
                else:
                    comando_geral_display = f"MENU: Decolar em {max(0, tempo_restante_decolagem_menu):.0f}s..."
                action_taken_this_menu_cycle = True
            elif timestamp_mao_aberta_decolagem_menu_detectada != 0: # Confirmando mão aberta.
                if current_gesture_tuple == (1,1,1,1,1):
                    if current_time - timestamp_mao_aberta_decolagem_menu_detectada >= DURACAO_MAO_ABERTA_PARA_DECOLAGEM_MENU_S:
                        mao_aberta_confirmada_para_decolagem_menu = True
                        timestamp_inicio_atraso_decolagem_menu = current_time
                        comando_geral_display = f"MENU: Mao aberta confirmada. Decolar em {ATRASO_FINAL_ANTES_DECOLAGEM_MENU_SEGUNDOS:.0f}s..."
                        print(comando_geral_display)
                        timestamp_mao_aberta_decolagem_menu_detectada = 0
                    else:
                        tempo_para_confirmar = DURACAO_MAO_ABERTA_PARA_DECOLAGEM_MENU_S - (current_time - timestamp_mao_aberta_decolagem_menu_detectada)
                        comando_geral_display = f"MENU: Mantenha mao aberta p/ decolar ({tempo_para_confirmar:.1f}s)"
                else: # Gesto mudou.
                    print("INFO: Mao nao esta (mais) aberta para decolagem do menu. Resetando.")
                    timestamp_mao_aberta_decolagem_menu_detectada = 0
                action_taken_this_menu_cycle = True
            elif current_gesture_tuple == (1,1,1,1,1) and not tello.is_flying:
                timestamp_mao_aberta_decolagem_menu_detectada = current_time
                comando_geral_display = f"MENU: Detectada mao aberta. Mantenha por {DURACAO_MAO_ABERTA_PARA_DECOLAGEM_MENU_S:.1f}s para decolar."
                print(f"INFO: Mao aberta detectada p/ decolagem do menu. Mantenha por {DURACAO_MAO_ABERTA_PARA_DECOLAGEM_MENU_S:.1f}s.")
                action_taken_this_menu_cycle = True
        
        # Lógica de Pouso (se voando) ou Seleção de Modo.
        if not action_taken_this_menu_cycle:
            if drone_connected and tello.is_flying:
                if current_gesture_tuple == (1, 1, 1, 1, 1): # Pousar com confirmação.
                    if timestamp_mao_aberta_pousar_menu_detectada == 0:
                        timestamp_mao_aberta_pousar_menu_detectada = current_time
                        comando_geral_display = f"POUSAR? Mantenha ({DURACAO_MAO_ABERTA_PARA_POUSAR_S:.1f}s)"
                    elif current_time - timestamp_mao_aberta_pousar_menu_detectada >= DURACAO_MAO_ABERTA_PARA_POUSAR_S:
                        comando_geral_display = "POUSANDO (Menu Confirmado)..."
                        try:
                            print("INFO: Comando Pousar Confirmado (Menu)."); tello.land()
                            drone_operational_status = "Drone Pousado."
                            mao_aberta_confirmada_para_decolagem_menu = False
                            timestamp_mao_aberta_decolagem_menu_detectada = 0
                        except Exception as e:
                            print(f"Erro ao pousar (Menu): {e}"); drone_operational_status = "Erro pouso."
                        comando_geral_display = drone_operational_status
                        timestamp_mao_aberta_pousar_menu_detectada = 0
                    else:
                        tempo_restante_pouso = DURACAO_MAO_ABERTA_PARA_POUSAR_S - (current_time - timestamp_mao_aberta_pousar_menu_detectada)
                        comando_geral_display = f"POUSAR? Mantenha ({tempo_restante_pouso:.1f}s)"
                    action_taken_this_menu_cycle = True
                else: # Cancela tentativa de pouso.
                    if timestamp_mao_aberta_pousar_menu_detectada != 0:
                        print("INFO: Pouso por mão aberta (Menu) cancelado.")
                    timestamp_mao_aberta_pousar_menu_detectada = 0

                if not action_taken_this_menu_cycle and current_gesture_tuple: # Mudar de modo.
                    if current_gesture_tuple == (0, 1, 0, 0, 1): # Ind+Min -> Controle Gestos.
                        estado_atual_programa = EstadoPrograma.CONTROLE_GESTAO
                        controle_gestos_armado = False; timestamp_mao_fechada_detectada = 0
                        print("INFO: Entrando no Modo de Controle por Gestos (Desarmado)")
                        action_taken_this_menu_cycle = True
                    elif current_gesture_tuple == (0, 1, 1, 0, 0): # Ind+Med -> Modo Comandos.
                        estado_atual_programa = EstadoPrograma.MODO_COMANDO
                        print("INFO: Entrando no Modo de Comandos Especiais")
                        action_taken_this_menu_cycle = True
            
            elif drone_connected and not tello.is_flying and decolagem_executada:
                if not timestamp_mao_aberta_decolagem_menu_detectada and not mao_aberta_confirmada_para_decolagem_menu:
                    comando_geral_display = "Pousado. (MAO ABERTA)=Decolar | 'q'=Sair"
            elif not decolagem_executada and drone_connected:
                estado_atual_programa = EstadoPrograma.INICIALIZANDO
            else:
                comando_geral_display = drone_operational_status

        # Display padrão do MENU se nenhuma ação específica de menu estiver em andamento.
        if not action_taken_this_menu_cycle and not (
            comando_geral_display.startswith("MENU: Decolar em") or \
            comando_geral_display.startswith("MENU: Mantenha mao aberta") or \
            comando_geral_display.startswith("POUSAR? Mantenha")):
            if drone_connected and tello.is_flying:
                comando_geral_display = "Escolha: (IND+MIN)=Controle | (IND+MED)=Comandos"
            elif drone_connected and not tello.is_flying and decolagem_executada:
                comando_geral_display = "Pousado. (MAO ABERTA)=Decolar | 'q'=Sair"
            elif not decolagem_executada and drone_connected:
                estado_atual_programa = EstadoPrograma.INICIALIZANDO
            else:
                comando_geral_display = drone_operational_status

    elif estado_atual_programa == EstadoPrograma.CONTROLE_GESTAO:
        info_modo_display = "MODO: CONTROLE POR GESTOS"
        timestamp_mao_aberta_pousar_menu_detectada = 0
        timestamp_mao_aberta_pousar_comando_detectada = 0
        
        if current_gesture_tuple != (1,1,0,0,1) and not foto_confirmada_para_disparo_controle: 
            if timestamp_gesto_foto_detectado_controle != 0:
                print("INFO: Gesto de foto (Controle) cancelado.")
            timestamp_gesto_foto_detectado_controle = 0

        rc_left_right, rc_fwd_bwd, rc_up_down, rc_yaw = 0, 0, 0, 0

        if not controle_gestos_armado: # Armar controle por gestos.
            comando_geral_display = f"Mostre MAO FECHADA ({DURACAO_MAO_FECHADA_PARA_ARMAR_S:.1f}s) para armar"
            if current_gesture_tuple == (0,0,0,0,0):
                if timestamp_mao_fechada_detectada == 0: timestamp_mao_fechada_detectada = current_time
                elif current_time - timestamp_mao_fechada_detectada >= DURACAO_MAO_FECHADA_PARA_ARMAR_S:
                    controle_gestos_armado = True; comando_geral_display = "Controle ARMADO! Use gestos."
                    print("INFO: Controle por Gestos ARMADO.")
                    timestamp_gesto_foto_detectado_controle = 0 
                    foto_confirmada_para_disparo_controle = False
            else: timestamp_mao_fechada_detectada = 0
        else: # Controle ARMADO.
            action_taken_this_cycle_gestures = False

            if current_gesture_tuple == (1, 1, 1, 1, 1) and tello.is_flying: 
                if timestamp_mao_aberta_pousar_controle_detectada == 0:
                    timestamp_mao_aberta_pousar_controle_detectada = current_time
                    if drone_connected and tello.is_flying:
                        try:
                            tello.send_rc_control(0,0,0,0)
                        except Exception: 
                            pass
                    comando_geral_display = f"POUSAR/MENU? Mantenha ({DURACAO_MAO_ABERTA_PARA_POUSAR_S:.1f}s)"
                elif current_time - timestamp_mao_aberta_pousar_controle_detectada >= DURACAO_MAO_ABERTA_PARA_POUSAR_S:
                    comando_geral_display = "POUSANDO E VOLTANDO AO MENU..."
                    if drone_connected and tello.is_flying:
                        try:
                            tello.send_rc_control(0,0,0,0); time.sleep(0.1); tello.land()
                            drone_operational_status = "Drone Pousado."
                            print("Pousado (Controle Confirmado). Voltando ao menu.")
                        except Exception as e:
                            print(f"Erro ao pousar (Controle): {e}"); drone_operational_status = "Erro pouso."
                    else: drone_operational_status = "Drone ja pousado."
                    estado_atual_programa = EstadoPrograma.MENU; comando_geral_display = drone_operational_status
                    controle_gestos_armado = False; timestamp_mao_aberta_pousar_controle_detectada = 0
                    mao_aberta_confirmada_para_decolagem_menu = False; timestamp_mao_aberta_decolagem_menu_detectada = 0
                    timestamp_gesto_foto_detectado_controle = 0; foto_confirmada_para_disparo_controle = False
                else:
                    tempo_restante_pouso = DURACAO_MAO_ABERTA_PARA_POUSAR_S - (current_time - timestamp_mao_aberta_pousar_controle_detectada)
                    comando_geral_display = f"POUSAR/MENU? Mantenha ({tempo_restante_pouso:.1f}s)"
                action_taken_this_cycle_gestures = True
            else: 
                if timestamp_mao_aberta_pousar_controle_detectada != 0:
                    print("INFO: Pouso por mão aberta (Controle) cancelado.")
                timestamp_mao_aberta_pousar_controle_detectada = 0

            if not action_taken_this_cycle_gestures: # Tirar Foto.
                if current_gesture_tuple == (1,1,0,0,1) and tello.is_flying: 
                    if foto_confirmada_para_disparo_controle:
                        pass 
                    elif timestamp_gesto_foto_detectado_controle == 0: 
                        timestamp_gesto_foto_detectado_controle = current_time
                        comando_geral_display = f"FOTO? Segure ({DURACAO_GESTO_FOTO_S:.1f}s)"
                        if drone_connected and tello.is_flying:
                            try:
                                tello.send_rc_control(0,0,0,0)
                            except Exception: 
                                pass 
                        action_taken_this_cycle_gestures = True 
                    elif current_time - timestamp_gesto_foto_detectado_controle >= DURACAO_GESTO_FOTO_S: 
                        foto_confirmada_para_disparo_controle = True
                        comando_geral_display = "FOTO CONFIRMADA!" 
                        timestamp_gesto_foto_detectado_controle = 0 
                        if drone_connected and tello.is_flying:
                            try:
                                tello.send_rc_control(0,0,0,0)
                            except Exception: 
                                pass 
                        action_taken_this_cycle_gestures = True
                    else: 
                        tempo_restante_foto = DURACAO_GESTO_FOTO_S - (current_time - timestamp_gesto_foto_detectado_controle)
                        comando_geral_display = f"FOTO? Segure ({tempo_restante_foto:.1f}s)"
                        if drone_connected and tello.is_flying:
                            try:
                                tello.send_rc_control(0,0,0,0)
                            except Exception: 
                                pass 
                        action_taken_this_cycle_gestures = True
            
            if foto_confirmada_para_disparo_controle and not action_taken_this_cycle_gestures: # Ação de salvar foto.
                if frame_drone_local is not None:
                    base_file_name = f"tello_foto_controle_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                    file_name_completo = os.path.abspath(base_file_name)
                    print(f"INFO: Tentando salvar foto em: {file_name_completo}")
                    try:
                        frame_rgb_para_salvar = cv2.cvtColor(frame_drone_local, cv2.COLOR_BGR2RGB)
                        success_save = cv2.imwrite(file_name_completo, frame_rgb_para_salvar)
                        
                        if success_save:
                            print(f"INFO: Foto salva com sucesso como {file_name_completo}")
                            feedback_overlay_text = "FOTO SALVA!"
                            timestamp_feedback_overlay_end = current_time + DURACAO_FOTO_OVERLAY_S
                            comando_geral_display = "FOTO SALVA!"
                        else:
                            print(f"ERRO: cv2.imwrite FALHOU ao salvar {file_name_completo}")
                            feedback_overlay_text = "ERRO AO SALVAR (imwrite)"
                            timestamp_feedback_overlay_end = current_time + DURACAO_FOTO_OVERLAY_S
                            comando_geral_display = "ERRO SALVAR FOTO"
                    except Exception as e:
                        print(f"EXCEÇÃO ao salvar foto {file_name_completo}: {e}")
                        feedback_overlay_text = "EXCECAO AO SALVAR FOTO"
                        timestamp_feedback_overlay_end = current_time + DURACAO_FOTO_OVERLAY_S
                        comando_geral_display = "ERRO EXCECAO FOTO"
                else:
                    feedback_overlay_text = "FOTO: SEM IMAGEM"
                    timestamp_feedback_overlay_end = current_time + DURACAO_FOTO_OVERLAY_S
                    comando_geral_display = "FOTO: SEM IMAGEM"
                foto_confirmada_para_disparo_controle = False
                action_taken_this_cycle_gestures = True
            
            if not action_taken_this_cycle_gestures: # Comandos RC de movimento.
                gesture_name = "Pairando"
                if current_gesture_tuple == (0, 1, 1, 1, 1): rc_fwd_bwd = RC_SPEED; gesture_name = "FRENTE"      
                elif current_gesture_tuple == (1, 1, 0, 0, 0): rc_fwd_bwd = -RC_SPEED; gesture_name = "TRAS"     
                elif current_gesture_tuple == (0, 1, 0, 0, 0): rc_up_down = RC_SPEED; gesture_name = "CIMA"       
                elif current_gesture_tuple == (0, 1, 1, 0, 0): rc_up_down = -RC_SPEED; gesture_name = "BAIXO" 
                elif current_gesture_tuple == (1, 0, 0, 0, 0): rc_left_right = -RC_SPEED; gesture_name = "ESQUERDA"
                elif current_gesture_tuple == (0, 0, 0, 0, 1): rc_left_right = RC_SPEED; gesture_name = "DIREITA" 
                elif current_gesture_tuple == (1, 0, 0, 0, 1): rc_yaw = RC_SPEED; gesture_name = "ROT.DIR"      
                elif current_gesture_tuple == (1, 1, 1, 0, 0): rc_yaw = -RC_SPEED; gesture_name = "ROT.ESQ"
                
                if gesture_name != "Pairando":
                    comando_geral_display = f"Controle ARMADO: {gesture_name}"
                else: 
                    if timestamp_gesto_foto_detectado_controle == 0: 
                        comando_geral_display = "Controle ARMADO: Pairando..."
                
                if drone_connected and tello.is_flying:
                    try: tello.send_rc_control(rc_left_right, rc_fwd_bwd, rc_up_down, rc_yaw)
                    except Exception as e: print(f"Erro ao enviar RC control: {e}"); comando_geral_display = "Erro RC"

        if not (drone_connected and tello.is_flying) and estado_atual_programa == EstadoPrograma.CONTROLE_GESTAO :
            estado_atual_programa = EstadoPrograma.MENU; comando_geral_display = drone_operational_status
            controle_gestos_armado = False
            timestamp_mao_aberta_pousar_controle_detectada = 0 
            timestamp_gesto_foto_detectado_controle = 0; foto_confirmada_para_disparo_controle = False
            mao_aberta_confirmada_para_decolagem_menu = False
            timestamp_mao_aberta_decolagem_menu_detectada = 0

    elif estado_atual_programa == EstadoPrograma.MODO_COMANDO:
        info_modo_display = "MODO: COMANDOS ESPECIAIS"
        timestamp_mao_aberta_pousar_menu_detectada = 0
        timestamp_mao_aberta_pousar_controle_detectada = 0
        timestamp_gesto_foto_detectado_controle = 0 
        foto_confirmada_para_disparo_controle = False
        rc_left_right, rc_fwd_bwd, rc_up_down, rc_yaw = 0, 0, 0, 0

        if not (drone_connected and tello.is_flying):
            comando_geral_display = drone_operational_status
            estado_atual_programa = EstadoPrograma.MENU
            follow_me_active = False; scan_ativo_global = False; follow_me_is_searching = False
            follow_me_target_acquired_this_session = False
            timestamp_mao_aberta_pousar_comando_detectada = 0
            mao_aberta_confirmada_para_decolagem_menu = False
            timestamp_mao_aberta_decolagem_menu_detectada = 0
        else:
            action_taken_this_cycle_cmd = False

            if current_gesture_tuple == (1, 1, 1, 1, 1) and tello.is_flying: 
                if timestamp_mao_aberta_pousar_comando_detectada == 0:
                    timestamp_mao_aberta_pousar_comando_detectada = current_time
                    if follow_me_active or scan_ativo_global: 
                        if drone_connected and tello.is_flying:
                            try:
                                tello.send_rc_control(0,0,0,0)
                            except Exception:
                                pass
                    comando_geral_display = f"POUSAR/MENU? Mantenha ({DURACAO_MAO_ABERTA_PARA_POUSAR_S:.1f}s)"
                elif current_time - timestamp_mao_aberta_pousar_comando_detectada >= DURACAO_MAO_ABERTA_PARA_POUSAR_S:
                    if scan_ativo_global:
                        scan_ativo_global = False; scan_fase_atual = "ocioso"; print("INFO: Scan interrompido para pouso confirmado.")
                    if follow_me_active:
                        follow_me_active = False; follow_me_target_acquired_this_session = False; follow_me_is_searching = False; print("INFO: Modo 'Me Siga' interrompido para pouso confirmado.")
                        try:
                            tello.send_rc_control(0,0,0,0) 
                        except Exception as e:
                            print(f"Erro ao parar RC para pouso: {e}")
                    
                    comando_geral_display = "POUSANDO E VOLTANDO AO MENU..."
                    if drone_connected and tello.is_flying:
                        try:
                            tello.land(); drone_operational_status = "Drone Pousado."; print("Pousado (Modo Comando Confirmado). Voltando ao menu.")
                        except Exception as e:
                            print(f"Erro ao pousar (Modo Comando): {e}"); drone_operational_status = "Erro pouso."
                    else: drone_operational_status = "Drone ja pousado."
                    
                    estado_atual_programa = EstadoPrograma.MENU
                    comando_geral_display = drone_operational_status
                    timestamp_mao_aberta_pousar_comando_detectada = 0
                    mao_aberta_confirmada_para_decolagem_menu = False 
                    timestamp_mao_aberta_decolagem_menu_detectada = 0
                else:
                    tempo_restante_pouso = DURACAO_MAO_ABERTA_PARA_POUSAR_S - (current_time - timestamp_mao_aberta_pousar_comando_detectada)
                    comando_geral_display = f"POUSAR/MENU? Mantenha ({tempo_restante_pouso:.1f}s)"
                action_taken_this_cycle_cmd = True
            else: 
                if timestamp_mao_aberta_pousar_comando_detectada != 0: 
                    print("INFO: Pouso por mão aberta (Modo Comando) cancelado.")
                timestamp_mao_aberta_pousar_comando_detectada = 0

            if not action_taken_this_cycle_cmd and current_gesture_tuple:
                if current_gesture_tuple == (1, 0, 0, 0, 1) and not scan_ativo_global and not follow_me_active:
                    if face_cascade is None:
                        comando_geral_display = "ME SIGA: Classificador nao carregado!"
                    else:
                        follow_me_active = True; scan_ativo_global = False
                        timestamp_last_face_seen_follow = 0; follow_me_target_acquired_this_session = False
                        follow_me_is_searching = True; follow_me_search_phase = "hover"
                        timestamp_follow_search_phase_start = current_time; follow_me_search_rotation_count = 0
                        comando_geral_display = "ME SIGA: Iniciando busca por rosto..."
                        print("INFO: Modo 'Me Siga' ativado - Iniciando busca.")
                    action_taken_this_cycle_cmd = True
                elif current_gesture_tuple == (0, 1, 1, 1, 0) and not scan_ativo_global and not follow_me_active:
                    if face_cascade is None:
                        comando_geral_display = "SCAN: Classificador nao carregado!"
                    else:
                        scan_ativo_global = True; scan_fase_atual = "iniciando"; scan_angulo_corrente = 0
                        scan_total_rostos_acumulado = 0; scan_rostos_nesta_etapa = 0
                        follow_me_active = False; follow_me_target_acquired_this_session = False; follow_me_is_searching = False
                        comando_geral_display = "SCAN ROSTOS: Iniciando..."; feedback_overlay_text = "" 
                        print("INFO: Modo 'Scan Rostos' ativado.")
                    action_taken_this_cycle_cmd = True
            
            if not timestamp_mao_aberta_pousar_comando_detectada != 0: # Se não estiver confirmando pouso.
                if follow_me_active: # Lógica do "Me Siga".
                    action_taken_this_cycle_cmd = True 
                    face_detected_this_cycle = False

                    if frame_drone_local is not None and face_cascade is not None:
                        gray_drone = cv2.cvtColor(frame_drone_local, cv2.COLOR_BGR2GRAY)
                        h_drone, w_drone = gray_drone.shape[:2]
                        dynamic_center_threshold_x = CENTER_THRESHOLD_X_RATIO * w_drone
                        dynamic_center_threshold_y = CENTER_THRESHOLD_Y_RATIO * h_drone
                        desired_face_width_pixels = DESIRED_FACE_WIDTH_SCREEN_RATIO * w_drone
                        min_face_size = (int(w_drone*0.08), int(h_drone*0.08))
                        detected_faces = face_cascade.detectMultiScale(gray_drone, scaleFactor=1.1, minNeighbors=5, minSize=min_face_size)
                        
                        if len(detected_faces) > 0:
                            face_detected_this_cycle = True
                            detected_faces = sorted(detected_faces, key=lambda f: f[2]*f[3], reverse=True)
                            fx, fy, fw, fh = detected_faces[0]
                            face_center_x = fx + fw // 2
                            face_center_y = fy + fh // 2
                            timestamp_last_face_seen_follow = current_time
                            
                            if follow_me_is_searching: 
                                print("INFO: Me Siga - Rosto encontrado durante a busca!")
                                feedback_overlay_text = "ROSTO ENCONTRADO!\nINICIANDO RASTREAMENTO..."
                                timestamp_feedback_overlay_end = current_time + 3.0 
                                try:
                                    tello.send_rc_control(0,0,0,0)
                                except Exception:
                                    pass
                            
                            follow_me_is_searching = False 
                            follow_me_target_acquired_this_session = True
                            
                            if face_center_x < (w_drone // 2 - dynamic_center_threshold_x): rc_yaw = -FOLLOW_YAW_SPEED
                            elif face_center_x > (w_drone // 2 + dynamic_center_threshold_x): rc_yaw = FOLLOW_YAW_SPEED
                            else: rc_yaw = 0
                            if face_center_y < (h_drone // 2 - dynamic_center_threshold_y): rc_up_down = FOLLOW_SPEED_UD
                            elif face_center_y > (h_drone // 2 + dynamic_center_threshold_y): rc_up_down = -FOLLOW_SPEED_UD
                            else: rc_up_down = 0
                            if fw < desired_face_width_pixels * 0.80: rc_fwd_bwd = FOLLOW_SPEED_FB
                            elif fw > desired_face_width_pixels * 1.20: rc_fwd_bwd = -FOLLOW_SPEED_FB
                            else:
                                if rc_yaw == 0 and rc_up_down == 0: rc_fwd_bwd = 0
                            
                            if not (feedback_overlay_text and current_time < timestamp_feedback_overlay_end): 
                                comando_geral_display = f"ME SIGA: Rastreando (Y:{rc_yaw}, UD:{rc_up_down}, FB:{rc_fwd_bwd})"
                        
                    if not face_detected_this_cycle and follow_me_active: 
                        if follow_me_is_searching: # Lógica de busca ativa.
                            if follow_me_search_phase == "hover":
                                comando_geral_display = f"ME SIGA: Buscando (Pairando {follow_me_search_rotation_count}/{FOLLOW_SEARCH_MAX_ROTATIONS})"
                                try:
                                    tello.send_rc_control(0,0,0,0)
                                except Exception:
                                    pass
                                if current_time - timestamp_follow_search_phase_start > FOLLOW_SEARCH_HOVER_DURATION_S:
                                    if follow_me_search_rotation_count >= FOLLOW_SEARCH_MAX_ROTATIONS:
                                        print("INFO: Me Siga - Limite de busca atingido. Nenhum rosto encontrado. Desativando.")
                                        comando_geral_display = "ME SIGA: Busca encerrada. Rosto nao encontrado."
                                        feedback_overlay_text = "BUSCA ENCERRADA\nROSTO NAO ENCONTRADO"
                                        timestamp_feedback_overlay_end = current_time + 4.0
                                        follow_me_active = False
                                        follow_me_is_searching = False
                                    else:
                                        follow_me_search_phase = "rotate"
                                        timestamp_follow_search_phase_start = current_time
                                        print(f"INFO: Me Siga - Buscando, iniciando rotação {follow_me_search_rotation_count + 1}")
                            
                            elif follow_me_search_phase == "rotate":
                                comando_geral_display = f"ME SIGA: Buscando (Rotacionando {follow_me_search_rotation_count + 1}/{FOLLOW_SEARCH_MAX_ROTATIONS})"
                                try:
                                    tello.rotate_clockwise(FOLLOW_SEARCH_ROTATE_ANGLE_DEG)
                                    time.sleep(FOLLOW_SEARCH_STABILIZE_TIME_S) 
                                except Exception as e_rot:
                                    print(f"Erro durante rotação de busca: {e_rot}")
                                    try:
                                        tello.send_rc_control(0,0,0,0)
                                    except Exception:
                                        pass
                                
                                follow_me_search_rotation_count += 1
                                follow_me_search_phase = "hover" 
                                timestamp_follow_search_phase_start = current_time
                        
                        else: # Alvo adquirido e perdido.
                            if follow_me_target_acquired_this_session: 
                                if current_time - timestamp_last_face_seen_follow > FACE_LOST_HOVER_TIMEOUT_S:
                                    comando_geral_display = "ME SIGA: Alvo perdido. Desativando."
                                    print("INFO: Me Siga - Alvo perdido por muito tempo. Desativando.")
                                    follow_me_active = False
                                    follow_me_target_acquired_this_session = False
                                else:
                                    tempo_restante_perdido = FACE_LOST_HOVER_TIMEOUT_S - (current_time - timestamp_last_face_seen_follow)
                                    comando_geral_display = f"ME SIGA: Alvo perdido! ({tempo_restante_perdido:.1f}s)"
                                rc_left_right, rc_fwd_bwd, rc_up_down, rc_yaw = 0,0,0,0 

                    if follow_me_active and not (follow_me_is_searching and follow_me_search_phase == "rotate"): 
                        try:
                            tello.send_rc_control(rc_left_right, rc_fwd_bwd, rc_up_down, rc_yaw)
                        except Exception as e:
                            print(f"Erro ao enviar RC no 'Me Siga': {e}")
                            comando_geral_display = "ME SIGA: Erro RC"
                    elif not follow_me_active: 
                        try:
                            tello.send_rc_control(0,0,0,0)
                        except Exception:
                            pass
                
                elif frame_drone_local is None and follow_me_active :
                    comando_geral_display = "ME SIGA: Sem frame do drone."
                    follow_me_active = False; follow_me_target_acquired_this_session = False; follow_me_is_searching = False
                    try: tello.send_rc_control(0,0,0,0)
                    except Exception: 
                        pass
                elif face_cascade is None and follow_me_active:
                    comando_geral_display = "ME SIGA: Classificador nao carregado!"
                    follow_me_active = False; follow_me_target_acquired_this_session = False; follow_me_is_searching = False
                
                # Lógica do Scan de Rostos.
                if scan_ativo_global and drone_connected and face_cascade is not None:
                    action_taken_this_cycle_cmd = True 
                    if scan_fase_atual == "iniciando":
                        if not tello.is_flying: comando_geral_display = "SCAN: Drone nao voando!"; scan_ativo_global = False; scan_fase_atual = "ocioso"
                        else: comando_geral_display = f"SCAN: Detectando em {scan_angulo_corrente}deg..."; print(comando_geral_display); scan_fase_atual = "detectando"
                    elif scan_fase_atual == "detectando":
                        scan_rostos_nesta_etapa = 0
                        if frame_drone_local is not None:
                            gray_drone = cv2.cvtColor(frame_drone_local, cv2.COLOR_BGR2GRAY)
                            detected_faces_current_step = face_cascade.detectMultiScale(gray_drone, scaleFactor=1.08, minNeighbors=7, minSize=(40, 40))
                            scan_rostos_nesta_etapa = len(detected_faces_current_step); scan_total_rostos_acumulado += scan_rostos_nesta_etapa
                        else: print("SCAN: Frame do drone NULO.")
                        comando_geral_display = f"SCAN ({scan_angulo_corrente}deg): {scan_rostos_nesta_etapa} faces. Total: {scan_total_rostos_acumulado}"
                        if scan_angulo_corrente >= (360 - ANGULO_ROTACAO_GRAUS):
                            scan_fase_atual = "girando_para_finalizar"
                        else:
                            scan_fase_atual = "preparando_giro"
                    elif scan_fase_atual == "preparando_giro":
                        comando_geral_display = f"SCAN: Girando para {scan_angulo_corrente + ANGULO_ROTACAO_GRAUS}deg... (Total: {scan_total_rostos_acumulado})"; print(comando_geral_display); scan_fase_atual = "girando"
                    elif scan_fase_atual == "girando":
                        if tello.is_flying:
                            try:
                                print(f"SCAN: Girando {ANGULO_ROTACAO_GRAUS} graus...");
                                tello.rotate_clockwise(ANGULO_ROTACAO_GRAUS);
                                time.sleep(2.5)
                                scan_angulo_corrente += ANGULO_ROTACAO_GRAUS
                                comando_geral_display = f"SCAN: Detectando em {scan_angulo_corrente}deg... (Total: {scan_total_rostos_acumulado})"; print(comando_geral_display); scan_fase_atual = "detectando"
                            except Exception as e: print(f"Erro giro: {e}"); comando_geral_display = "SCAN: Erro giro!"; scan_fase_atual = "erro"
                        else: comando_geral_display = "SCAN: Drone nao voando!"; scan_fase_atual = "erro"
                    elif scan_fase_atual == "girando_para_finalizar":
                        proximo_angulo_final = (scan_angulo_corrente + ANGULO_ROTACAO_GRAUS) % 360
                        comando_geral_display = f"SCAN: Finalizando volta para {proximo_angulo_final}deg... (Total: {scan_total_rostos_acumulado})"
                        print(comando_geral_display)
                        if tello.is_flying:
                            try:
                                print(f"SCAN: Girando últimos {ANGULO_ROTACAO_GRAUS} graus para completar a volta...")
                                tello.rotate_clockwise(ANGULO_ROTACAO_GRAUS)
                                time.sleep(3.0) # Aumentado
                                scan_angulo_corrente = proximo_angulo_final
                                scan_fase_atual = "concluido"
                            except Exception as e:
                                print(f"Erro no giro final: {e}")
                                scan_fase_atual = "erro"
                        else:
                            comando_geral_display = "SCAN: Drone nao voando durante giro final!"
                            scan_fase_atual = "erro"
                    elif scan_fase_atual == "concluido":
                        msg_final_scan = f"SCAN FINALIZADO!\n\n{scan_total_rostos_acumulado} ROSTOS DETECTADOS" 
                        comando_geral_display = f"SCAN FINALIZADO: {scan_total_rostos_acumulado} ROSTOS" 
                        print(f"INFO: {msg_final_scan.replace('\n', ' - ')}")
                        
                        if drone_connected and tello.is_flying: # Comando explícito para pairar
                            try:
                                print("INFO: Scan concluído. Comandando pairar para estabilizar.")
                                tello.send_rc_control(0,0,0,0)
                                time.sleep(0.2) 
                            except Exception as e_hover_end_scan:
                                print(f"Erro ao comandar pairar pós-scan: {e_hover_end_scan}")

                        feedback_overlay_text = msg_final_scan
                        timestamp_feedback_overlay_end = current_time + DURACAO_PADRAO_OVERLAY_S
                        scan_ativo_global = False; scan_fase_atual = "ocioso"
                        if drone_connected and tello.is_flying: drone_operational_status = "Em voo."
                        elif drone_connected: drone_operational_status = "Drone Pousado."
                    elif scan_fase_atual == "erro":
                        print(f"SCAN: Erro - {comando_geral_display}"); scan_ativo_global = False; scan_fase_atual = "ocioso"
                        if drone_connected and tello.is_flying: drone_operational_status = "Em voo (Scan erro)."
                        elif drone_connected: drone_operational_status = "Pousado (Scan erro)."

            # Display padrão do MODO_COMANDO se nenhuma ação específica estiver em andamento.
            if not action_taken_this_cycle_cmd and \
            not (comando_geral_display.startswith("POUSANDO") or \
                    comando_geral_display.startswith("SCAN:") or \
                    comando_geral_display.startswith("ME SIGA:") or \
                    comando_geral_display.startswith("POUSAR/MENU? Mantenha")):
                if drone_connected and tello.is_flying:
                    comando_geral_display = "Modo Comando: (P+Mn)=Siga|(I+M+A)=Scan"

    # Verificação Periódica da Bateria.
    if drone_connected and (current_time - last_battery_check_time > battery_check_interval):
        try: battery_level = tello.get_battery(); last_battery_check_time = current_time
        except Exception as e: print(f"Erro ao buscar bateria: {e}")

    # Lógica de Keep-Alive Global (evita timeout do drone).
    send_global_keep_alive = True
    if estado_atual_programa == EstadoPrograma.CONTROLE_GESTAO and controle_gestos_armado:
        is_confirming_action = (timestamp_mao_aberta_pousar_controle_detectada != 0 or \
                                timestamp_gesto_foto_detectado_controle != 0 or \
                                foto_confirmada_para_disparo_controle)
        if not (rc_left_right == 0 and rc_fwd_bwd == 0 and rc_up_down == 0 and rc_yaw == 0 and not is_confirming_action):
            send_global_keep_alive = False
        elif is_confirming_action:
            send_global_keep_alive = False # Desativa se RC ativo ou confirmando gesto.

    if estado_atual_programa == EstadoPrograma.MODO_COMANDO:
        is_confirming_landing_cmd = timestamp_mao_aberta_pousar_comando_detectada != 0
        if is_confirming_landing_cmd or follow_me_active or scan_ativo_global:
            send_global_keep_alive = False # Desativa se sub-modo ativo ou confirmando pouso.


    if drone_connected and tello.is_flying and send_global_keep_alive and \
    (current_time - last_keep_alive_time > KEEP_ALIVE_INTERVAL):
        try:
            tello.send_rc_control(0, 0, 0, 0)
            last_keep_alive_time = current_time
        except Exception as e:
            print(f"Erro global keep-alive: {e}")

    # Desenho das Informações na Tela.
    target_image_for_text = image_notebook_bgr
    cv2.putText(target_image_for_text, info_modo_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(target_image_for_text, hand_info_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if drone_connected: battery_text = f"Bateria Tello: {battery_level}%"
    else: battery_text = "Bateria Tello: N/A"
    battery_color = (0,255,0) if battery_level > 20 else (0,0,255)
    cv2.putText(target_image_for_text, battery_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, battery_color, 2)
    cv2.putText(target_image_for_text, comando_geral_display, (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    start_y_legend = 170
    if estado_atual_programa == EstadoPrograma.MENU:
        menu_options_texts = [
            "IND+MINIMO   -> Controle Gestos (voando)",
            "2 DEDOS (I+M) -> Modo Comandos (voando)",
            "MAO ABERTA   -> Pousar (se voando, segure)", 
            "MAO ABERTA   -> Decolar (se pousado)"
        ]
        for i, line in enumerate(menu_options_texts):
            cv2.putText(target_image_for_text, line, (10, start_y_legend + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 128, 0), 2)
    elif estado_atual_programa == EstadoPrograma.CONTROLE_GESTAO:
        controle_gestao_legend_texts = [
            "Ind(Unico): Subir", "Ind+Med: Descer",
            "Pol(Unico): Esquerda", "Minimo(Unico): Direita", 
            "Pol+Min: Rot.Dir", "Pol+Ind+Med: Rot.Esq",
            "4 dedos: Frente", 
            "Pol+Ind(L): Tras",
            "Mao Aberta: Pousar/Menu (segure)",
            "Pol+Ind+Min (segure): Tirar Foto" 
        ]
        for i, line in enumerate(controle_gestao_legend_texts):
            y_pos = start_y_legend + i * 20 
            if y_pos < target_image_for_text.shape[0] -10 :
                cv2.putText(target_image_for_text, line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2) 
    elif estado_atual_programa == EstadoPrograma.MODO_COMANDO:
        comando_mode_legend_texts = [
            "Pol+Min: Me Siga",
            "Ind+Med+Anel: Scan Rostos",
            "Mao Aberta: Pousar/Menu (segure)"
        ]
        for i, line in enumerate(comando_mode_legend_texts):
            y_pos = start_y_legend + i * 25
            if y_pos < target_image_for_text.shape[0] -10 :
                cv2.putText(target_image_for_text, line, (10, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 2)

    if scan_ativo_global and scan_fase_atual not in ["ocioso", "iniciando", "concluido", "erro"]:
        info_scan_tela = f"Angulo: {scan_angulo_corrente}deg | Etapa: {scan_rostos_nesta_etapa} | Total: {scan_total_rostos_acumulado}"
        y_pos_scan_info = target_image_for_text.shape[0] - 15
        cv2.putText(target_image_for_text, info_scan_tela, (10, y_pos_scan_info), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 150, 0), 2)

    if feedback_overlay_text and current_time < timestamp_feedback_overlay_end: # Overlay Genérico
        (h_img_note, w_img_note) = target_image_for_text.shape[:2]
        lines = feedback_overlay_text.split('\n')
        font_overlay = cv2.FONT_HERSHEY_TRIPLEX
        
        overlay_color = (255,255,255) # Branco
        font_scale_overlay = 1.0
        font_thickness_overlay = 2

        if "SCAN FINALIZADO" in feedback_overlay_text: 
            font_scale_overlay = 1.3 
            overlay_color = (0, 255, 255) # Amarelo
        elif "FOTO SALVA" in feedback_overlay_text :
            font_scale_overlay = 1.1
            overlay_color = (50, 255, 50) # Verde
        elif "ERRO" in feedback_overlay_text or "ROSTO NAO ENCONTRADO" in feedback_overlay_text: 
            font_scale_overlay = 1.1
            overlay_color = (0,0,255) # Vermelho
        elif "ROSTO ENCONTRADO" in feedback_overlay_text:
            font_scale_overlay = 1.1
            overlay_color = (50,200,50) # Verde escuro


        line_heights_actual = []
        total_text_height_calc = 0
        for i_line, line_text in enumerate(lines):
            (lw, lh), baseline = cv2.getTextSize(line_text, font_overlay, font_scale_overlay, font_thickness_overlay)
            actual_lh = lh + baseline 
            line_heights_actual.append(actual_lh)
            total_text_height_calc += actual_lh
            if i_line < len(lines) - 1:
                 total_text_height_calc += int(actual_lh * 0.3) 

        padding_overlay = 40 
        max_text_w_overlay = 0
        if lines: 
            max_text_w_overlay = max(cv2.getTextSize(line, font_overlay, font_scale_overlay, font_thickness_overlay)[0][0] for line in lines) if lines else 0
        
        rect_w_overlay = max_text_w_overlay + 2 * padding_overlay
        rect_h_overlay = total_text_height_calc + 2 * padding_overlay
        
        rect_x_overlay = max(0, (w_img_note - rect_w_overlay) // 2)
        rect_y_overlay = max(0, (h_img_note - rect_h_overlay) // 2)

        rect_w_overlay = min(rect_w_overlay, w_img_note - rect_x_overlay)
        rect_h_overlay = min(rect_h_overlay, h_img_note - rect_y_overlay)

        if rect_w_overlay > 0 and rect_h_overlay > 0:
            overlay_region_original = target_image_for_text[rect_y_overlay:rect_y_overlay+rect_h_overlay, rect_x_overlay:rect_x_overlay+rect_w_overlay].copy()
            cv2.rectangle(overlay_region_original, (0,0), (rect_w_overlay, rect_h_overlay), (20, 20, 20), -1) 
            alpha = 0.80
            target_image_for_text[rect_y_overlay:rect_y_overlay+rect_h_overlay, rect_x_overlay:rect_x_overlay+rect_w_overlay] = cv2.addWeighted(
                overlay_region_original, alpha, 
                target_image_for_text[rect_y_overlay:rect_y_overlay+rect_h_overlay, rect_x_overlay:rect_x_overlay+rect_w_overlay], 1 - alpha, 0)
            
            current_y_text_start = rect_y_overlay + padding_overlay
            for i_line, line in enumerate(lines):
                (lw, lh), baseline = cv2.getTextSize(line, font_overlay, font_scale_overlay, font_thickness_overlay)
                actual_lh = lh + baseline # Ajuste para baseline e espaçamento
                text_x_o = rect_x_overlay + (rect_w_overlay - lw) // 2
                text_y_o = current_y_text_start + lh 
                
                cv2.putText(target_image_for_text, line, (text_x_o, text_y_o), 
                            font_overlay, font_scale_overlay, 
                            overlay_color, 
                            font_thickness_overlay, cv2.LINE_AA)
                current_y_text_start += actual_lh + int(actual_lh * 0.3) 
                
    elif feedback_overlay_text and current_time >= timestamp_feedback_overlay_end: 
        feedback_overlay_text = "" 

    cv2.imshow('Camera Notebook - Gestos e Comandos', target_image_for_text)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

# --- Finalização do Programa ---
if drone_connected:
    print("Finalizando...");
    if tello.is_flying:
        try:
            print("Comandando pairar antes de pousar...");
            tello.send_rc_control(0,0,0,0); time.sleep(0.5)
            print("Pousando o drone...");
            tello.land(); time.sleep(3)
        except Exception as e: print(f"Erro ao pousar na finalizacao: {e}")
    try: tello.streamoff()
    except Exception as e: print(f"Erro streamoff na finalizacao: {e}")
    tello.end()
cap_notebook.release()
if hands: hands.close()
cv2.destroyAllWindows()
print("Programa finalizado.")
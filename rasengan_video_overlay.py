import cv2
import mediapipe as mp
import numpy as np
import math
import os
import pygame

# 1. Khởi tạo Pygame Audio (2 kênh riêng biệt)
pygame.mixer.init()
pygame.mixer.set_num_channels(2)
chan_rasengan = pygame.mixer.Channel(0)
chan_chidori = pygame.mixer.Channel(1)

snd_rasengan = pygame.mixer.Sound("rasengan.mp3") if os.path.exists("rasengan.mp3") else None
snd_chidori = pygame.mixer.Sound("chidori.mp3") if os.path.exists("chidori.mp3") else None

# 2. Khởi tạo Video & MediaPipe Hands
cap_webcam = cv2.VideoCapture(0)
cap_rasengan = cv2.VideoCapture("rasengan.mp4")
cap_chidori = cv2.VideoCapture("chidori.mp4")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65
)

# Trạng thái và vị trí nhớ để chống tráo tay
active_rasengan = False
active_chidori = False
prev_rasengan_pos = None  # Lưu tọa độ (x, y) của tay giữ Rasengan

def is_hand_open(landmarks):
    """Kiểm tra xòe bàn tay."""
    wrist = landmarks[0]
    tips = [8, 12, 16, 20]
    mips = [6, 10, 14, 18]
    open_count = 0
    for tip, mip in zip(tips, mips):
        d_tip = math.hypot(landmarks[tip].x - wrist.x, landmarks[tip].y - wrist.y)
        d_mip = math.hypot(landmarks[mip].x - wrist.x, landmarks[mip].y - wrist.y)
        if d_tip > d_mip:
            open_count += 1
    return open_count >= 3

def crop_center_square(img):
    """Cắt vuông tâm video 16:9 để bảo toàn hình cầu."""
    h, w = img.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    return img[start_y:start_y + min_dim, start_x:start_x + min_dim]

def read_loop_frame(cap):
    """Đọc frame video lặp vô tận."""
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    return ret, frame

def blend_effect_screen(frame, overlay, center_x, center_y, size, threshold_val=20):
    """
    Hòa trộn Screen Blending: Nét căng vân xoáy chiêu thức, không bị cháy trắng bệt.
    """
    h_f, w_f, _ = frame.shape
    half = size // 2

    x1, y1 = center_x - half, center_y - half
    x2, y2 = center_x + half, center_y + half

    clip_x1, clip_y1 = max(0, x1), max(0, y1)
    clip_x2, clip_y2 = min(w_f, x2), min(h_f, y2)

    if clip_x1 >= clip_x2 or clip_y1 >= clip_y2:
        return

    # Resize chất lượng cao bằng INTER_CUBIC để giữ độ sắc nét
    square_overlay = crop_center_square(overlay)
    resized_overlay = cv2.resize(square_overlay, (size, size), interpolation=cv2.INTER_CUBIC)

    # Lọc bỏ nền tối
    gray = cv2.cvtColor(resized_overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)

    ov_x1 = clip_x1 - x1
    ov_y1 = clip_y1 - y1
    ov_x2 = ov_x1 + (clip_x2 - clip_x1)
    ov_y2 = ov_y1 + (clip_y2 - clip_y1)

    roi = frame[clip_y1:clip_y2, clip_x1:clip_x2]
    overlay_crop = resized_overlay[ov_y1:ov_y2, ov_x1:ov_x2]
    mask_crop = mask[ov_y1:ov_y2, ov_x1:ov_x2]

    overlay_clean = cv2.bitwise_and(overlay_crop, overlay_crop, mask=mask_crop)

    # Công thức Screen Blend (Chống bão hòa trắng, hiện rõ từng luồng Chakra)
    roi_norm = roi.astype(np.float32) / 255.0
    ov_norm = overlay_clean.astype(np.float32) / 255.0
    blended_norm = 1.0 - (1.0 - roi_norm) * (1.0 - ov_norm)

    frame[clip_y1:clip_y2, clip_x1:clip_x2] = (blended_norm * 255.0).astype(np.uint8)

while cap_webcam.isOpened():
    ret_w, frame = cap_webcam.read()
    if not ret_w:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Thu thập tất cả các bàn tay đang xòe kèm vị trí tâm lòng bàn tay
    open_hands_data = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            if is_hand_open(lm):
                # Vị trí tâm lòng bàn tay (tỷ lệ 35% cổ tay - 65% khớp ngón giữa)
                cx = int((0.35 * lm[0].x + 0.65 * lm[9].x) * w)
                cy = int((0.35 * lm[0].y + 0.65 * lm[9].y) * h)
                span = math.hypot((lm[9].x - lm[0].x) * w, (lm[9].y - lm[0].y) * h)
                open_hands_data.append({"center": (cx, cy), "span": span, "lm": lm})

    rasengan_hand = None
    chidori_hand = None

    # --- KHÓA MỤC TIÊU (TRACKING): BẢO VỆ RASENGAN KHÔNG BỊ TRÁO ---
    if len(open_hands_data) == 1:
        # Chỉ có 1 tay -> Luôn là Rasengan
        rasengan_hand = open_hands_data[0]
        prev_rasengan_pos = rasengan_hand["center"]

    elif len(open_hands_data) >= 2:
        if prev_rasengan_pos is not None:
            # Tìm tay nào gần vị trí Rasengan cũ nhất -> Tiếp tục gán Rasengan cho tay đó
            dist_0 = math.hypot(open_hands_data[0]["center"][0] - prev_rasengan_pos[0],
                                open_hands_data[0]["center"][1] - prev_rasengan_pos[1])
            dist_1 = math.hypot(open_hands_data[1]["center"][0] - prev_rasengan_pos[0],
                                open_hands_data[1]["center"][1] - prev_rasengan_pos[1])

            if dist_0 < dist_1:
                rasengan_hand = open_hands_data[0]
                chidori_hand = open_hands_data[1]
            else:
                rasengan_hand = open_hands_data[1]
                chidori_hand = open_hands_data[0]

            prev_rasengan_pos = rasengan_hand["center"]
        else:
            # Nếu cả 2 tay cùng xuất hiện cùng lúc ở frame đầu tiên
            rasengan_hand = open_hands_data[0]
            chidori_hand = open_hands_data[1]
            prev_rasengan_pos = rasengan_hand["center"]

    else:
        prev_rasengan_pos = None

    # --- 1. RENDER RASENGAN ---
    if rasengan_hand:
        if not active_rasengan:
            cap_rasengan.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if snd_rasengan:
                chan_rasengan.play(snd_rasengan, loops=-1)
            active_rasengan = True

        ret_r, v_rasengan = read_loop_frame(cap_rasengan)
        if ret_r:
            cx, cy = rasengan_hand["center"]
            effect_size = max(180, min(int(rasengan_hand["span"] * 2.7), 650))
            blend_effect_screen(frame, v_rasengan, cx, cy, effect_size, threshold_val=22)
    else:
        if active_rasengan:
            chan_rasengan.stop()
            active_rasengan = False
            cap_rasengan.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- 2. RENDER CHIDORI ---
    if chidori_hand:
        if not active_chidori:
            cap_chidori.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if snd_chidori:
                chan_chidori.play(snd_chidori, loops=-1)
            active_chidori = True

        ret_c, v_chidori = read_loop_frame(cap_chidori)
        if ret_c:
            cx, cy = chidori_hand["center"]
            effect_size = max(200, min(int(chidori_hand["span"] * 3.0), 700))
            blend_effect_screen(frame, v_chidori, cx, cy, effect_size, threshold_val=18)
    else:
        if active_chidori:
            chan_chidori.stop()
            active_chidori = False
            cap_chidori.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # UI trạng thái chiêu thức
    cv2.putText(frame, f"Rasengan: {'ON' if active_rasengan else 'OFF'}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Chidori: {'ON' if active_chidori else 'OFF'}", (220, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    cv2.imshow("Rasengan vs Chidori Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

chan_rasengan.stop()
chan_chidori.stop()
cap_webcam.release()
cap_rasengan.release()
cap_chidori.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
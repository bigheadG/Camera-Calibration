#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_calib_gui_1280x720_fix6_roll_estimate_target.py

改進重點（針對你拍照後常出現 [FAIL] 的問題）：
1) 修正：原本 self.frame_raw 沒有在 timer 更新 → 按拍照時常是 None 或抓到不對的 frame。
2) 角點偵測更強韌：
   - 先用 findChessboardCornersSB（若 OpenCV 支援）
   - 失敗再用 findChessboardCorners + 多種前處理（equalize / blur / invert）+ FILTER_QUADS
3) Auto pattern：你 UI 輸入 (cols, rows) 後，會自動嘗試：
   - (cols, rows)、(rows, cols)、(cols-1, rows-1)、(rows-1, cols-1)
   這能解決「你說 10x7 但其實是 10x7 方格 → 9x6 內角點」的常見誤會。
4) Live corners 預覽（可關）：畫面左上角會顯示 LiveCorners=OK/FAIL，幫你在按下拍照前就知道是否抓得到角點。
5) Roll 角度即時估算：若抓到棋盤角點，畫面會顯示 Roll≈±xx°；在 roll 步驟時會提示 TargetRoll。

依賴：
    pip install opencv-python PyQt5 numpy
"""

import sys
import time
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets


# ---------------------------
# helpers
# ---------------------------

# ---------------------------
# distance guide (near/mid/far)
# ---------------------------
# 以「棋盤在畫面中的寬度佔比」來定義近/中/遠（比用公分更可靠）
# 也用來決定黃色 target box 的大小，方便你目測距離是否符合要求。
DIST_GUIDE = {
    "近": {"box_scale": 0.48, "wmin": 0.70, "wmax": 0.92},
    "中": {"box_scale": 0.32, "wmin": 0.40, "wmax": 0.65},
    "遠": {"box_scale": 0.22, "wmin": 0.20, "wmax": 0.38},
}

def cvimg_to_qimage(img_bgr: np.ndarray) -> QtGui.QImage:
    """BGR uint8 -> QImage (RGB888)"""
    if img_bgr is None:
        return QtGui.QImage()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    bytes_per_line = 3 * w
    return QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def draw_grid_overlay(img_bgr: np.ndarray, color=(80, 80, 80), thickness=1) -> np.ndarray:
    """畫 3x3 grid + 螢幕中心十字（中心十字固定黃色、thickness=2）"""
    if img_bgr is None:
        return img_bgr
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    # 3x3 grid
    for k in [1, 2]:
        x = int(w * k / 3)
        y = int(h * k / 3)
        cv2.line(out, (x, 0), (x, h - 1), color, thickness, cv2.LINE_AA)
        cv2.line(out, (0, y), (w - 1, y), color, thickness, cv2.LINE_AA)

    # center cross (yellow, thickness=2)
    cx, cy = w // 2, h // 2
    cross_len = 30
    cv2.line(out, (cx - cross_len, cy), (cx + cross_len, cy), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(out, (cx, cy - cross_len), (cx, cy + cross_len), (0, 255, 255), 2, cv2.LINE_AA)
    return out


def draw_target_hint(img_bgr: np.ndarray, target: str, dist_level: str = '中') -> np.ndarray:
    """在畫面上標示目標區域（左上/右上/左下/右下/中心），並在目標框中心畫準星十字"""
    if img_bgr is None:
        return img_bgr
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    pad = 20
    # 稍微縮小目標框，讓你更容易把棋盤放大拍清楚
    scale = DIST_GUIDE.get(dist_level, DIST_GUIDE['中'])['box_scale']
    bw, bh = int(w * scale), int(h * scale)

    if target == "左上":
        x1, y1 = pad, pad
    elif target == "右上":
        x1, y1 = w - pad - bw, pad
    elif target == "左下":
        x1, y1 = pad, h - pad - bh
    elif target == "右下":
        x1, y1 = w - pad - bw, h - pad - bh
    else:  # 中心
        x1, y1 = (w - bw) // 2, (h - bh) // 2

    x2, y2 = x1 + bw, y1 + bh

    # target box
    box_color = (0, 255, 255)
    cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
    cv2.putText(out, f"Target: {target}  Dist: {dist_level}", (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA)

    # crosshair at target center (aim)
    tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
    cross_len = max(12, int(min(bw, bh) * 0.14))
    cv2.line(out, (tx - cross_len, ty), (tx + cross_len, ty), box_color, 2, cv2.LINE_AA)
    cv2.line(out, (tx, ty - cross_len), (tx, ty + cross_len), box_color, 2, cv2.LINE_AA)

    return out




def estimate_roll_deg(corners: np.ndarray, pat: tuple) -> Optional[float]:
    """估算棋盤在影像平面內的 roll 角度（度）。
    正值/負值依影像座標系，主要用來做「0/15/30/45°」這類引導即可。
    """
    try:
        if corners is None or pat is None:
            return None
        cols, rows = int(pat[0]), int(pat[1])
        if cols <= 1 or rows <= 1:
            return None
        if len(corners) < cols:
            return None
        # OpenCV 的 corners 順序通常是 row-major；取第一列的左右端點
        tl = corners[0, 0]              # (x,y)
        tr = corners[cols - 1, 0]       # (x,y)
        dx = float(tr[0] - tl[0])
        dy = float(tr[1] - tl[1])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None
        ang = math.degrees(math.atan2(dy, dx))
        # normalize to [-90, 90] for readability
        while ang > 90:
            ang -= 180
        while ang < -90:
            ang += 180
        return float(ang)
    except Exception:
        return None

def laplacian_var(gray: np.ndarray) -> float:
    """粗略衡量清晰度（越大越清楚）"""
    if gray is None or gray.size == 0:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def build_pattern_candidates(cols: int, rows: int, auto_try: bool = True):
    """
    回傳 pattern candidates（OpenCV 期待的是「內角點數」）
    auto_try=True 會多嘗試：
      - 交換 cols/rows（避免你輸入反了）
      - -1/-1（避免你輸入的是「方格數」而非「內角點數」）
    """
    c, r = int(cols), int(rows)
    base = [(c, r)]
    if auto_try:
        if (r, c) != (c, r):
            base.append((r, c))
        if c > 3 and r > 3:
            base.append((c - 1, r - 1))
            if (r - 1, c - 1) != (c - 1, r - 1):
                base.append((r - 1, c - 1))

    # 去重、過濾不合理
    seen = set()
    out = []
    for pr in base:
        if pr[0] < 3 or pr[1] < 3:
            continue
        if pr in seen:
            continue
        seen.add(pr)
        out.append(pr)
    return out


def _detect_chessboard(gray: np.ndarray, pattern_size, thorough: bool = True):
    """
    回傳 (found, corners, method)
    thorough=True 會跑較重但更可靠的組合（用於拍照/校正）
    thorough=False 速度優先（用於 live preview）
    """
    if gray is None:
        return False, None, "none"

    pattern_size = (int(pattern_size[0]), int(pattern_size[1]))

    # --- 1) SB（通常最穩，但可能比較慢） ---
    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flag_sets = []
        if thorough:
            sb_flag_sets = [
                cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
                cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
                cv2.CALIB_CB_NORMALIZE_IMAGE,
            ]
        else:
            sb_flag_sets = [
                cv2.CALIB_CB_NORMALIZE_IMAGE,
            ]
        for flags in sb_flag_sets:
            try:
                found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
                if found and corners is not None:
                    return True, corners, "SB"
            except Exception:
                pass

    # --- 2) 傳統 findChessboardCorners（多種前處理/旗標） ---
    variants = [("orig", gray)]
    if thorough:
        try:
            variants.append(("eq", cv2.equalizeHist(gray)))
        except Exception:
            pass
        try:
            variants.append(("blur", cv2.GaussianBlur(gray, (5, 5), 0)))
        except Exception:
            pass
        variants.append(("inv", 255 - gray))

    flag_sets = []
    if thorough:
        flag_sets = [
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FILTER_QUADS,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
            cv2.CALIB_CB_NORMALIZE_IMAGE,
        ]
    else:
        flag_sets = [
            cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        ]

    for vname, img in variants:
        for flags in flag_sets:
            try:
                found, corners = cv2.findChessboardCorners(img, pattern_size, flags=flags)
                if found and corners is not None:
                    return True, corners, f"FC:{vname}"
            except Exception:
                continue

    return False, None, "none"


# ---------------------------
# calibration worker
# ---------------------------
class CalibWorker(QtCore.QThread):
    finished_signal = QtCore.pyqtSignal(dict)
    progress_signal = QtCore.pyqtSignal(str)

    def __init__(self, img_paths, pattern_cols, pattern_rows, square_size_m, out_json_path,
                 auto_pattern=True, parent=None):
        super().__init__(parent)
        self.img_paths = list(img_paths)
        self.pattern_cols = int(pattern_cols)
        self.pattern_rows = int(pattern_rows)
        self.square_size_m = float(square_size_m)
        self.out_json_path = str(out_json_path)
        self.auto_pattern = bool(auto_pattern)

    def run(self):
        try:
            res = self._do_calib()
            self.finished_signal.emit(res)
        except Exception as e:
            self.finished_signal.emit({"ok": False, "error": str(e)})

    def _choose_best_pattern(self, images, candidates):
        """掃一輪，選成功數最多的 pattern"""
        best = None
        best_cnt = -1
        per_pat = {pat: 0 for pat in candidates}

        for i, fn in enumerate(images):
            img = cv2.imread(fn)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for pat in candidates:
                found, _, _ = _detect_chessboard(gray, pat, thorough=True)
                if found:
                    per_pat[pat] += 1

        # choose best (tie → earlier)
        for pat in candidates:
            cnt = per_pat.get(pat, 0)
            if cnt > best_cnt:
                best_cnt = cnt
                best = pat
        return best, best_cnt, per_pat

    def _do_calib(self):
        images = self.img_paths
        if not images:
            raise RuntimeError("沒有可用影像。請先拍幾張。")

        candidates = build_pattern_candidates(self.pattern_cols, self.pattern_rows, auto_try=self.auto_pattern)
        if not candidates:
            raise RuntimeError("Pattern candidates 無效，請檢查 cols/rows。")

        # 自動挑最適 pattern（避免 10x7 方格 vs 內角點 9x6）
        self.progress_signal.emit("Selecting best pattern ...")
        best_pat, best_cnt, per_pat = self._choose_best_pattern(images, candidates)
        if best_pat is None or best_cnt <= 0:
            raise RuntimeError("所有 pattern 都找不到棋盤角點。請確認棋盤尺寸、畫面清晰、曝光與距離。")

        pattern_size = best_pat
        self.progress_signal.emit(f"Pattern selected: {pattern_size[0]}x{pattern_size[1]}  (hits={best_cnt})")

        # 3D object points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size_m

        objpoints = []
        imgpoints = []
        img_size = None

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        good = 0
        for i, fn in enumerate(images):
            self.progress_signal.emit(f"Corner detect {i+1}/{len(images)}: {Path(fn).name}")
            img = cv2.imread(fn)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if img_size is None:
                img_size = (gray.shape[1], gray.shape[0])

            found, corners, _ = _detect_chessboard(gray, pattern_size, thorough=True)
            if not found or corners is None:
                continue

            # refine (用原始 gray)
            try:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            except Exception:
                corners2 = corners

            objpoints.append(objp)
            imgpoints.append(corners2)
            good += 1

        if good < 10:
            raise RuntimeError(f"成功找到角點的圖片太少：{good}（建議至少 15 張；最低 10 張）")

        self.progress_signal.emit("Running calibrateCamera() ...")
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        # reprojection RMSE
        total_err = 0.0
        total_pts = 0
        for i in range(len(objpoints)):
            proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
            err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2)
            n = len(proj)
            total_err += err * err
            total_pts += n
        rmse = float(np.sqrt(total_err / max(1, total_pts)))

        out = {
            "ok": True,
            "image_size": [int(img_size[0]), int(img_size[1])],
            "pattern_size_used": [int(pattern_size[0]), int(pattern_size[1])],
            "pattern_candidates": [[int(a), int(b)] for (a, b) in candidates],
            "pattern_hit_count": {f"{a}x{b}": int(per_pat.get((a, b), 0)) for (a, b) in candidates},
            "K": K.tolist(),
            "distCoeffs": dist.ravel().tolist(),
            "rms": float(ret),
            "reproj_rmse": rmse,
            "used_images": int(good),
            "total_images": int(len(images)),
            "rvec": [0.0, 0.0, 0.0],
            "tvec": [0.0, 0.0, 0.0],
        }

        Path(self.out_json_path).write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out


# ---------------------------
# GUI
# ---------------------------
class CameraCalibGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Calibration Capture (1280x720) + Calibrate K/dist (fix4)")

        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)

        self.frame_bgr = None
        self.frame_raw = None  # raw camera frame (no overlays)  <-- fix4: 會在 _on_timer 更新
        self.last_saved_path = None

        self.plan = self._build_default_plan()
        self.plan_idx = 0
        self.auto_advance = True

        self.out_dir = Path("./calib_imgs")
        ensure_dir(self.out_dir)

        self._build_ui()
        self._open_camera()

    def closeEvent(self, e: QtGui.QCloseEvent):
        self._close_camera()
        super().closeEvent(e)

    def _build_ui(self):
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        root = QtWidgets.QVBoxLayout(cw)

        top = QtWidgets.QHBoxLayout()
        root.addLayout(top)

        top.addWidget(QtWidgets.QLabel("Device index:"))
        self.spin_dev = QtWidgets.QSpinBox()
        self.spin_dev.setRange(0, 10)
        self.spin_dev.setValue(0)
        top.addWidget(self.spin_dev)

        btn_reopen = QtWidgets.QPushButton("Re-open Camera")
        btn_reopen.clicked.connect(self._open_camera)
        top.addWidget(btn_reopen)

        top.addSpacing(20)
        top.addWidget(QtWidgets.QLabel("Save dir:"))
        self.edit_dir = QtWidgets.QLineEdit(str(self.out_dir.resolve()))
        top.addWidget(self.edit_dir, 2)
        btn_dir = QtWidgets.QPushButton("Browse...")
        btn_dir.clicked.connect(self.on_browse_dir)
        top.addWidget(btn_dir)

        top.addSpacing(20)
        self.chk_auto = QtWidgets.QCheckBox("Auto-advance on OK")
        self.chk_auto.setChecked(True)
        self.chk_auto.stateChanged.connect(lambda s: setattr(self, "auto_advance", s == QtCore.Qt.Checked))
        top.addWidget(self.chk_auto)

        self.chk_live = QtWidgets.QCheckBox("Live corners preview")
        self.chk_live.setChecked(True)
        top.addWidget(self.chk_live)

        top.addStretch(1)

        mid = QtWidgets.QHBoxLayout()
        root.addLayout(mid, 1)

        self.lbl_video = QtWidgets.QLabel()
        self.lbl_video.setMinimumSize(960, 540)
        self.lbl_video.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_video.setStyleSheet("background: #111; border: 1px solid #444;")
        mid.addWidget(self.lbl_video, 3)

        panel = QtWidgets.QVBoxLayout()
        mid.addLayout(panel, 2)

        self.lbl_hint = QtWidgets.QLabel("Hint:")
        self.lbl_hint.setWordWrap(True)
        self.lbl_hint.setStyleSheet("font-size: 14px;")
        panel.addWidget(self.lbl_hint)

        btns = QtWidgets.QHBoxLayout()
        panel.addLayout(btns)

        self.btn_capture = QtWidgets.QPushButton("📸 Take Picture")
        self.btn_capture.setStyleSheet("font-size: 16px; padding: 8px;")
        self.btn_capture.clicked.connect(self.on_take_picture)
        btns.addWidget(self.btn_capture, 2)

        self.btn_undo = QtWidgets.QPushButton("Undo last")
        self.btn_undo.clicked.connect(self.on_undo_last)
        btns.addWidget(self.btn_undo, 1)

        panel.addWidget(QtWidgets.QLabel("Captured images (corner detection result):"))
        self.list_imgs = QtWidgets.QListWidget()
        self.list_imgs.setMinimumHeight(200)
        self.list_imgs.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        # Right-click delete (robust across platforms):
        # Enable context menu on BOTH the QListWidget and its viewport, because some desktop
        # environments deliver the event to different widgets.
        self.list_imgs.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list_imgs.customContextMenuRequested.connect(lambda p: self.on_list_imgs_context_menu(p, from_viewport=False))
        self.list_imgs.viewport().setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list_imgs.viewport().customContextMenuRequested.connect(lambda p: self.on_list_imgs_context_menu(p, from_viewport=True))

        # Optional: keyboard Delete key to delete selected item
        act_del = QtWidgets.QAction(self)
        act_del.setShortcut(QtGui.QKeySequence.Delete)
        act_del.triggered.connect(self.on_delete_selected_item)
        self.addAction(act_del)

        panel.addWidget(self.list_imgs, 1)

        box = QtWidgets.QGroupBox("Calibrate Settings")
        panel.addWidget(box)
        g = QtWidgets.QGridLayout(box)

        g.addWidget(QtWidgets.QLabel("Pattern (cols, rows):"), 0, 0)
        self.spin_cols = QtWidgets.QSpinBox()
        self.spin_cols.setRange(3, 30)
        self.spin_cols.setValue(10)
        self.spin_rows = QtWidgets.QSpinBox()
        self.spin_rows.setRange(3, 30)
        self.spin_rows.setValue(7)

        row_pat = QtWidgets.QHBoxLayout()
        wpat = QtWidgets.QWidget()
        wpat.setLayout(row_pat)
        row_pat.addWidget(self.spin_cols)
        row_pat.addWidget(QtWidgets.QLabel("x"))
        row_pat.addWidget(self.spin_rows)
        g.addWidget(wpat, 0, 1)

        self.chk_auto_pattern = QtWidgets.QCheckBox("Auto pattern (try swap / -1)")
        self.chk_auto_pattern.setChecked(True)
        g.addWidget(self.chk_auto_pattern, 1, 1)

        g.addWidget(QtWidgets.QLabel("Square size (m):"), 2, 0)
        self.dsp_square = QtWidgets.QDoubleSpinBox()
        self.dsp_square.setDecimals(6)
        self.dsp_square.setRange(0.001, 0.2)
        self.dsp_square.setSingleStep(0.001)
        self.dsp_square.setValue(0.025)
        g.addWidget(self.dsp_square, 2, 1)

        g.addWidget(QtWidgets.QLabel("Output JSON:"), 3, 0)
        self.edit_json = QtWidgets.QLineEdit(str(Path("./fusion_camera.json").resolve()))
        g.addWidget(self.edit_json, 3, 1)

        row_c = QtWidgets.QHBoxLayout()
        panel.addLayout(row_c)

        self.btn_calib = QtWidgets.QPushButton("✅ Calibrate (K/dist)")
        self.btn_calib.setStyleSheet("font-size: 15px; padding: 6px;")
        self.btn_calib.clicked.connect(self.on_calibrate)
        row_c.addWidget(self.btn_calib, 2)

        self.btn_validate = QtWidgets.QPushButton("🔎 Validate (RMSE)")
        self.btn_validate.clicked.connect(self.on_validate)
        row_c.addWidget(self.btn_validate, 1)

        self.txt_result = QtWidgets.QPlainTextEdit()
        self.txt_result.setReadOnly(True)
        self.txt_result.setMaximumHeight(190)
        panel.addWidget(self.txt_result)

        self.status = self.statusBar()
        self._update_hint()

        self.resize(1500, 880)

    # camera
    def _open_camera(self):
        self._close_camera()
        dev = int(self.spin_dev.value())

        # Linux 上用 CAP_V4L2 會比較穩
        try:
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        except Exception:
            cap = cv2.VideoCapture(dev)

        if not cap.isOpened():
            self.status.showMessage(f"❌ Cannot open camera index {dev}")
            return

        # reduce latency
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # set 1280x720
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

        self.cap = cap
        self.timer.start(15)
        self.status.showMessage(f"✅ Camera opened: {w}x{h}  FPS={fps:.1f}  (dev={dev})")

    def _close_camera(self):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.frame_raw = None
        self.frame_bgr = None

    def _on_timer(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return

        # fix4: keep raw frame (no overlays)
        self.frame_raw = frame.copy()

        # overlay frame for display
        frame2 = draw_grid_overlay(frame)
        step = self.plan[self.plan_idx] if self.plan else {"pos": "中心", "dist": "中", "angle": "-"}
        target = step.get("pos", "中心")
        dist_level = step.get("dist", "中")
        frame2 = draw_target_hint(frame2, target, dist_level)

        txt = f"Step {self.plan_idx+1}/{len(self.plan)}  pos={step['pos']}  dist={step['dist']}  angle={step['angle']}"
        cv2.putText(frame2, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2, cv2.LINE_AA)

        # live corner preview (speed-first)
        if self.chk_live.isChecked():
            gray = cv2.cvtColor(self.frame_raw, cv2.COLOR_BGR2GRAY)
            sharp = laplacian_var(gray)
            cols = int(self.spin_cols.value())
            rows = int(self.spin_rows.value())
            candidates = build_pattern_candidates(cols, rows, auto_try=self.chk_auto_pattern.isChecked())

            live_found = False
            live_pat = None
            live_corners = None

            # live：只跑輕量版本
            for pat in candidates:
                found, corners, _ = _detect_chessboard(gray, pat, thorough=False)
                if found and corners is not None:
                    live_found = True
                    live_pat = pat
                    live_corners = corners
                    break

            # draw corners if found
            if live_found and live_corners is not None:
                try:
                    cv2.drawChessboardCorners(frame2, live_pat, live_corners, True)
                except Exception:
                    pass

            # status text

            # distance check (based on chessboard bbox width ratio)
            dist_ok = None
            dist_ratio = None
            dist_msg = ""
            if live_found and live_corners is not None and len(live_corners) > 0:
                xs = live_corners[:, 0, 0]
                ys = live_corners[:, 0, 1]
                bw_pix = float(xs.max() - xs.min())
                dist_ratio = bw_pix / float(gray.shape[1])
                g = DIST_GUIDE.get(dist_level, DIST_GUIDE["中"])
                dist_ok = (g["wmin"] <= dist_ratio <= g["wmax"])
                if dist_ok:
                    dist_msg = f"  DistOK ({dist_ratio*100:.0f}%)"
                else:
                    # too small -> move closer; too large -> move farther
                    if dist_ratio < g["wmin"]:
                        dist_msg = f"  TOO FAR ({dist_ratio*100:.0f}%)"
                    else:
                        dist_msg = f"  TOO CLOSE ({dist_ratio*100:.0f}%)"

            
            st = f"LiveCorners={'OK' if live_found else 'FAIL'}"
            if live_pat is not None:
                st += f" ({live_pat[0]}x{live_pat[1]})"
            st += f"  Sharp={sharp:.0f}" + dist_msg

            # roll estimate (image-plane)
            roll_deg = None
            if live_found and live_corners is not None and live_pat is not None:
                roll_deg = estimate_roll_deg(live_corners, live_pat)
                if roll_deg is not None:
                    st += f"  Roll≈{roll_deg:+.0f}°"

                    # if current plan step asks for roll, add target hint
                    try:
                        cur_step = self.plan[self.plan_idx] if self.plan else None
                        ang_name = (cur_step.get("angle", "") if isinstance(cur_step, dict) else "")
                        if "roll" in ang_name:
                            target_abs = [15.0, 30.0, 45.0][int(self.plan_idx) % 3]  # cycle 15/30/45
                            target = (-target_abs if "左" in ang_name else target_abs)
                            ok = (abs(roll_deg - target) <= 7.5)  # within ~±7.5°
                            st += f"  TargetRoll≈{target:+.0f}°" + (" (OK)" if ok else " (ADJUST)")
                    except Exception:
                        pass

            if live_found and (dist_ok is None or dist_ok):
                color = (0, 220, 0)
            elif live_found and (dist_ok is False):
                color = (0, 180, 255)  # orange
            else:
                color = (0, 0, 255)
            cv2.putText(frame2, st, (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        self.frame_bgr = frame2
        qimg = cvimg_to_qimage(frame2)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.lbl_video.setPixmap(pix.scaled(self.lbl_video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    # plan
    def _build_default_plan(self):
        positions = ["中心", "左上", "右上", "左下", "右下"]
        dists = ["近", "中", "遠"]
        angles = ["正面", "左傾(roll)", "右傾(roll)", "上仰(pitch)", "下俯(pitch)", "左轉(yaw)", "右轉(yaw)"]
        plan = []
        ai = 0
        for dist in dists:
            for pos in positions:
                plan.append({"pos": pos, "dist": dist, "angle": angles[ai % len(angles)]})
                ai += 1
        return plan

    def _update_hint(self):
        if not self.plan:
            self.lbl_hint.setText("Hint: (no plan)")
            return
        s = self.plan[self.plan_idx]
        msg = (
            f"請把棋盤放在：{s['pos']}\n"
            f"距離：{s['dist']}（近/中/遠）\n"
            f"建議：近(棋盤寬度約 70~92% 畫面寬)、中(40~65%)、遠(20~38%)\n"
            f"角度：{s['angle']}\n""（提示：畫面會顯示 Roll≈±xx°，roll 步驟請盡量拍到 15°/30°/45° 其中幾種）\n\n"
            "拍照要點：\n"
            "• 棋盤整張都在畫面內、不要糊（Sharp 值越大越清楚）\n"
            "• 棋盤占畫面越大越容易成功（太遠/太小會 FAIL）\n"
            "• 每張的角度/位置要明顯不同\n"
            "• 建議至少拍 15~30 張（有效角點 ≥ 15 張）\n"
        )
        self.lbl_hint.setText(msg)

    def _next_plan(self):
        if not self.plan:
            return
        self.plan_idx += 1
        if self.plan_idx >= len(self.plan):
            self.plan_idx = 0
        self._update_hint()

    # capture
    def on_browse_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select folder", str(self.out_dir.resolve()))
        if d:
            self.out_dir = Path(d)
            ensure_dir(self.out_dir)
            self.edit_dir.setText(str(self.out_dir.resolve()))
            self.status.showMessage(f"Save dir set: {self.out_dir}")

    def _corner_check(self, img_path: Path, candidates):
        img = cv2.imread(str(img_path))
        if img is None:
            return False, None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        for pat in candidates:
            found, _, _ = _detect_chessboard(gray, pat, thorough=True)
            if found:
                return True, pat
        return False, None

    def on_take_picture(self):
        if self.frame_raw is None:
            self.status.showMessage("No frame yet.")
            return

        p = Path(self.edit_dir.text().strip() or str(self.out_dir))
        ensure_dir(p)
        self.out_dir = p

        s = self.plan[self.plan_idx] if self.plan else {"pos": "center", "dist": "x", "angle": "x"}
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = f"calib_{ts}_pos-{s['pos']}_dist-{s['dist']}_ang-{s['angle']}.jpg"
        out_path = self.out_dir / fn

        # save raw with high jpeg quality (less artifacts -> easier for corners)
        ok = cv2.imwrite(str(out_path), self.frame_raw, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            self.status.showMessage("❌ Save failed.")
            return

        self.last_saved_path = out_path

        cols = int(self.spin_cols.value())
        rows = int(self.spin_rows.value())
        candidates = build_pattern_candidates(cols, rows, auto_try=self.chk_auto_pattern.isChecked())

        found, used_pat = self._corner_check(out_path, candidates)

        # NOTE: 不改檔名、不在檔名後面加 [FAIL]，只在 list 顯示
        tag = "OK" if found else "FAIL"
        pat_txt = f"{used_pat[0]}x{used_pat[1]}" if used_pat else "-"
        item = QtWidgets.QListWidgetItem(f"{out_path.name}  [{tag}]  pat={pat_txt}")
        item.setData(QtCore.Qt.UserRole, str(out_path))
        item.setForeground(QtGui.QColor("#00ff66") if found else QtGui.QColor("#ff6666"))
        self.list_imgs.addItem(item)
        self.list_imgs.scrollToBottom()

        self.status.showMessage(f"✅ Saved: {out_path.name}   corners={tag}   pat={pat_txt}")

        if self.auto_advance and found:
            self._next_plan()
        elif self.auto_advance and (not found):
            self.status.showMessage("⚠️ 角點 FAIL：請讓棋盤更大、更清楚（避免糊/反光），或確認你的 10x7 是否其實是「方格數」。")

    def on_undo_last(self):
        if self.last_saved_path is None:
            self.status.showMessage("No last image to undo.")
            return
        try:
            p = Path(self.last_saved_path)
            if p.exists():
                p.unlink()
            for i in range(self.list_imgs.count() - 1, -1, -1):
                it = self.list_imgs.item(i)
                if it and it.data(QtCore.Qt.UserRole) == str(p):
                    self.list_imgs.takeItem(i)
                    break
            self.status.showMessage(f"Undo removed: {p.name}")
            self.last_saved_path = None
        except Exception as e:
            self.status.showMessage(f"Undo failed: {e}")

    # list context menu (right-click delete)
    def on_list_imgs_context_menu(self, pos: QtCore.QPoint, from_viewport: bool = False):
        # Right click on a captured image item to delete the file (with confirmation).
        # NOTE:
        # - If the signal comes from QListWidget.customContextMenuRequested, `pos` is in QListWidget coords.
        # - If the signal comes from viewport().customContextMenuRequested, `pos` is in viewport coords.
        # QListWidget.itemAt() expects viewport coords, so we always convert.
        if from_viewport:
            vp_pos = pos
        else:
            vp_pos = self.list_imgs.viewport().mapFrom(self.list_imgs, pos)

        item = self.list_imgs.itemAt(vp_pos)
        if item is None:
            return

        self._delete_list_item_with_confirm(item)

    def on_delete_selected_item(self):
        # Press Delete key to delete currently selected item (with confirmation).
        item = self.list_imgs.currentItem()
        if item is None:
            return
        self._delete_list_item_with_confirm(item)

    def _delete_list_item_with_confirm(self, item: QtWidgets.QListWidgetItem):
        # Internal: confirm + delete file on disk + remove list item.
        p_str = item.data(QtCore.Qt.UserRole)
        if not p_str:
            return
        p = Path(p_str)

        mb = QtWidgets.QMessageBox(self)
        mb.setIcon(QtWidgets.QMessageBox.Question)
        mb.setWindowTitle("Delete")
        mb.setText(f"Delete this file?\n\n{p.name}\n\nThis will remove it from disk.")
        btn_del = mb.addButton("Delete", QtWidgets.QMessageBox.DestructiveRole)
        btn_cancel = mb.addButton("Cancel", QtWidgets.QMessageBox.RejectRole)
        mb.setDefaultButton(btn_cancel)
        mb.exec_()
        if mb.clickedButton() != btn_del:
            return

        # delete from disk
        try:
            if p.exists():
                p.unlink()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Delete", f"Failed to delete file:\n{p}\n\n{e}")
            return

        # remove from list
        row = self.list_imgs.row(item)
        if row >= 0:
            self.list_imgs.takeItem(row)

        # sync last_saved_path
        try:
            if self.last_saved_path is not None and str(self.last_saved_path) == str(p):
                self.last_saved_path = None
        except Exception:
            pass

        self.status.showMessage(f"🗑️ Deleted: {p.name}")

    # calibrate
    def _collect_images(self):
        paths = []
        for i in range(self.list_imgs.count()):
            it = self.list_imgs.item(i)
            p = it.data(QtCore.Qt.UserRole)
            if p:
                paths.append(p)
        return paths

    def on_calibrate(self):
        paths = self._collect_images()
        if not paths:
            QtWidgets.QMessageBox.warning(self, "Calibrate", "沒有已拍攝圖片。")
            return

        out_json = self.edit_json.text().strip() or "fusion_camera.json"
        cols = int(self.spin_cols.value())
        rows = int(self.spin_rows.value())
        sq = float(self.dsp_square.value())

        self.btn_calib.setEnabled(False)
        self.btn_validate.setEnabled(False)

        self.worker = CalibWorker(
            paths, cols, rows, sq, out_json,
            auto_pattern=self.chk_auto_pattern.isChecked(),
            parent=self
        )
        self.worker.progress_signal.connect(lambda s: self.status.showMessage(s))
        self.worker.finished_signal.connect(self.on_calib_done)
        self.worker.start()
        self.status.showMessage("Calibrating...")

    def on_calib_done(self, res: dict):
        self.btn_calib.setEnabled(True)
        self.btn_validate.setEnabled(True)

        if not res.get("ok", False):
            self.txt_result.setPlainText(f"❌ Calibrate failed:\n{res.get('error','unknown error')}")
            self.status.showMessage("❌ Calibrate failed.")
            return

        K = np.array(res["K"], dtype=float)
        dist = np.array(res["distCoeffs"], dtype=float).ravel()

        msg = []
        msg.append("✅ Calibration OK")
        msg.append(f"Image size: {res['image_size']}")
        if "pattern_size_used" in res:
            msg.append(f"Pattern used: {res['pattern_size_used']}  (auto-detected)")
        msg.append(f"Used images: {res['used_images']} / {res['total_images']}")
        msg.append(f"RMS (cv2): {res['rms']:.6f}")
        msg.append(f"Reproj RMSE: {res['reproj_rmse']:.4f}")
        msg.append("\nK =")
        msg.append(str(K))
        msg.append("\ndistCoeffs =")
        msg.append(str(dist))
        msg.append(f"\nSaved JSON: {self.edit_json.text().strip()}")
        self.txt_result.setPlainText("\n".join(msg))
        self.status.showMessage("✅ Calibrate done.")

    def on_validate(self):
        text = self.txt_result.toPlainText()
        if "Reproj RMSE:" not in text:
            QtWidgets.QMessageBox.information(self, "Validate", "請先 Calibrate 一次。")
            return

        rmse = None
        used = None
        for line in text.splitlines():
            if line.startswith("Reproj RMSE:"):
                try:
                    rmse = float(line.split(":")[1].strip())
                except Exception:
                    pass
            if line.startswith("Used images:"):
                try:
                    used = int(line.split(":")[1].split("/")[0].strip())
                except Exception:
                    pass

        ok = (rmse is not None and rmse <= 1.0) and (used is not None and used >= 15)
        hint = [f"RMSE={rmse:.4f}, used_images={used}"]
        hint.append("✅ 夠用" if ok else "⚠️ 建議再補拍/提高畫質")
        QtWidgets.QMessageBox.information(self, "Validate", "\n".join(hint))


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = CameraCalibGUI()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

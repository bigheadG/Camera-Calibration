#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
camera_calib_gui_1280x720.py

用途：
- 以 1280x720 直播顯示 webcam
- 一鍵拍照存檔（自動命名、可回復上一張）
- 內建「拍攝引導步驟」（不同位置/距離/角度）讓你更快收集可用校正資料
- 立即檢測棋盤角點（成功/失敗），避免拍一堆不能用的圖
- 一鍵 OpenCV calibrateCamera() → 輸出 K / distCoeffs
- 產出 fusion_camera.json（只含內參/畸變；外參 rvec/tvec 留 0 讓你後續在 Fusion GUI 微調）

依賴：
    pip install opencv-python PyQt5 numpy
"""

import sys
import time
import json
from pathlib import Path

import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets


def cvimg_to_qimage(img_bgr: np.ndarray) -> QtGui.QImage:
    """BGR uint8 -> QImage (RGB888)"""
    if img_bgr is None:
        return QtGui.QImage()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    bytes_per_line = 3 * w
    return QtGui.QImage(img_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()


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



def draw_target_hint(img_bgr: np.ndarray, target: str) -> np.ndarray:
    """在畫面上標示目標區域（左上/右上/左下/右下/中心），並在目標框中心畫準星十字"""
    if img_bgr is None:
        return img_bgr
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    pad = 20
    bw, bh = int(w * 0.28), int(h * 0.28)

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
    box_color = (0, 200, 255)
    cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)
    cv2.putText(out, f"Target: {target}", (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2, cv2.LINE_AA)

    # crosshair at target center (aim)
    tx, ty = (x1 + x2) // 2, (y1 + y2) // 2
    cross_len = max(12, int(min(bw, bh) * 0.12))
    cv2.line(out, (tx - cross_len, ty), (tx + cross_len, ty), box_color, 2, cv2.LINE_AA)
    cv2.line(out, (tx, ty - cross_len), (tx, ty + cross_len), box_color, 2, cv2.LINE_AA)

    return out



def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


class CalibWorker(QtCore.QThread):
    finished_signal = QtCore.pyqtSignal(dict)
    progress_signal = QtCore.pyqtSignal(str)

    def __init__(self, img_paths, pattern_cols, pattern_rows, square_size_m, out_json_path, parent=None):
        super().__init__(parent)
        self.img_paths = list(img_paths)
        self.pattern_cols = int(pattern_cols)
        self.pattern_rows = int(pattern_rows)
        self.square_size_m = float(square_size_m)
        self.out_json_path = str(out_json_path)

    def run(self):
        try:
            res = self._do_calib()
            self.finished_signal.emit(res)
        except Exception as e:
            self.finished_signal.emit({"ok": False, "error": str(e)})

    def _do_calib(self):
        images = self.img_paths
        if not images:
            raise RuntimeError("沒有可用影像。請先拍幾張。")

        pattern_size = (self.pattern_cols, self.pattern_rows)  # (cols, rows)

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

            found, corners = cv2.findChessboardCorners(
                gray, pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if not found:
                continue

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
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


class CameraCalibGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera Calibration Capture (1280x720) + Calibrate K/dist")

        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)

        self.frame_bgr = None
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
        panel.addWidget(self.list_imgs, 1)

        box = QtWidgets.QGroupBox("Calibrate Settings")
        panel.addWidget(box)
        g = QtWidgets.QGridLayout(box)

        g.addWidget(QtWidgets.QLabel("Pattern (cols, rows):"), 0, 0)
        self.spin_cols = QtWidgets.QSpinBox()
        self.spin_cols.setRange(3, 20)
        self.spin_cols.setValue(10)
        self.spin_rows = QtWidgets.QSpinBox()
        self.spin_rows.setRange(3, 20)
        self.spin_rows.setValue(7)
        row_pat = QtWidgets.QHBoxLayout()
        wpat = QtWidgets.QWidget()
        wpat.setLayout(row_pat)
        row_pat.addWidget(self.spin_cols)
        row_pat.addWidget(QtWidgets.QLabel("x"))
        row_pat.addWidget(self.spin_rows)
        g.addWidget(wpat, 0, 1)

        g.addWidget(QtWidgets.QLabel("Square size (m):"), 1, 0)
        self.dsp_square = QtWidgets.QDoubleSpinBox()
        self.dsp_square.setDecimals(6)
        self.dsp_square.setRange(0.001, 0.2)
        self.dsp_square.setSingleStep(0.001)
        self.dsp_square.setValue(0.025)
        g.addWidget(self.dsp_square, 1, 1)

        g.addWidget(QtWidgets.QLabel("Output JSON:"), 2, 0)
        self.edit_json = QtWidgets.QLineEdit(str(Path("./fusion_camera.json").resolve()))
        g.addWidget(self.edit_json, 2, 1)

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
        self.txt_result.setMaximumHeight(160)
        panel.addWidget(self.txt_result)

        self.status = self.statusBar()
        self._update_hint()

        self.resize(1500, 860)

    # camera
    def _open_camera(self):
        self._close_camera()
        dev = int(self.spin_dev.value())
        cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            self.status.showMessage(f"❌ Cannot open camera index {dev}")
            return

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

    def _on_timer(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return

        frame2 = draw_grid_overlay(frame)
        target = self.plan[self.plan_idx]["pos"] if self.plan else "中心"
        frame2 = draw_target_hint(frame2, target)

        step = self.plan[self.plan_idx] if self.plan else {"pos": "中心", "dist": "-", "angle": "-"}
        txt = f"Step {self.plan_idx+1}/{len(self.plan)}  pos={step['pos']}  dist={step['dist']}  angle={step['angle']}"
        cv2.putText(frame2, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 0), 2, cv2.LINE_AA)

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
            f"角度：{s['angle']}\n\n"
            "拍照要點：\n"
            "• 棋盤整張都在畫面內、不要糊\n"
            "• 每張的角度/位置要明顯不同\n"
            "• 至少拍 15~30 張\n"
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

    def on_take_picture(self):
        if self.frame_bgr is None:
            self.status.showMessage("No frame yet.")
            return

        p = Path(self.edit_dir.text().strip() or str(self.out_dir))
        ensure_dir(p)
        self.out_dir = p

        s = self.plan[self.plan_idx] if self.plan else {"pos": "center", "dist": "x", "angle": "x"}
        ts = time.strftime("%Y%m%d_%H%M%S")
        fn = f"calib_{ts}_pos-{s['pos']}_dist-{s['dist']}_ang-{s['angle']}.jpg"
        out_path = self.out_dir / fn

        ok = cv2.imwrite(str(out_path), self.frame_bgr)
        if not ok:
            self.status.showMessage("❌ Save failed.")
            return

        self.last_saved_path = out_path
        found = self._corner_check(out_path)

        item = QtWidgets.QListWidgetItem(f"{out_path.name}  [{'OK' if found else 'FAIL'}]")
        item.setData(QtCore.Qt.UserRole, str(out_path))
        item.setForeground(QtGui.QColor("#00ff66") if found else QtGui.QColor("#ff6666"))
        self.list_imgs.addItem(item)
        self.list_imgs.scrollToBottom()

        self.status.showMessage(f"✅ Saved: {out_path.name}   corners={'OK' if found else 'FAIL'}")

        # 只有棋盤角點偵測 OK 才自動前進到下一個目標
        if self.auto_advance and found:
            self._next_plan()
        elif self.auto_advance and (not found):
            self.status.showMessage("⚠️ 棋盤角點偵測 FAIL：請保持同一目標位置，調整棋盤角度/距離後再拍。")

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

    def _corner_check(self, img_path: Path) -> bool:
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pattern = (int(self.spin_cols.value()), int(self.spin_rows.value()))
        found, _ = cv2.findChessboardCorners(
            gray, pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        return bool(found)

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

        self.worker = CalibWorker(paths, cols, rows, sq, out_json, parent=self)
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

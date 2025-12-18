# Camera Calibration GUI (OpenCV `calibrateCamera`) — User Guide

這個工具提供一個 **PyQt5 + OpenCV** 的 GUI，協助你用棋盤格拍攝校正照片，並一鍵輸出 **Camera Intrinsics**：
- `K`（camera matrix）
- `distCoeffs`（畸變係數）

另外包含：
- **解析度下拉選單**（切換後自動重新開相機）
- **LiveCorners 預覽**（按拍照前就知道角點能不能抓到）
- **距離判斷 Near/Mid/Far（近/中/遠）**：用「棋盤在畫面寬度佔比」判斷 Too Far/Too Close
- **Roll 即時估算**：顯示 `Roll≈±xx°`，並在 roll 步驟提示 `TargetRoll≈15/30/45°`
- 右側清單 **右鍵刪除檔案**（Delete/Cancel）＋ `Delete` 鍵刪除

---

## 1. Requirements

- Python 3.8+
- 建議使用 virtualenv/venv
- 套件：
  ```bash
  pip install opencv-python PyQt5 numpy
  ```

> Linux：程式會優先用 `cv2.CAP_V4L2` 開相機，通常較穩；其他平台會自動 fallback。

---

## 2. Run

假設你的主程式檔名是：

- `camera_calib_gui_fix8_resolution_select_v2.py`

執行：

```bash
python3 camera_calib_gui_fix8_resolution_select_v2.py
```

---

## 3. Before You Start (棋盤格準備)

### 3.1 內角點（Pattern cols/rows）怎麼填？
OpenCV 需要的是 **內角點數**（inner corners），不是方格數（squares）。

- 如果你的棋盤是 **10 x 7 個方格**  
  → 內角點是 **9 x 6**
- 如果你的棋盤是 **9 x 6 個方格**  
  → 內角點是 **8 x 5**

本工具有 **Auto pattern**（會自動嘗試 swap / -1），可降低填錯的風險。

### 3.2 Square size（單格尺寸）
請量棋盤「單一格邊長」的實際尺寸，並填入 **公尺（m）**：
- 例如 25mm → `0.025`
- 例如 30mm → `0.030`

---

## 4. UI Overview

### 4.1 Camera / Capture
- **Device index**：相機編號（0,1,2...）
- **Resolution**：解析度選擇  
  - 建議：同一組校正資料集 **不要混用解析度**
  - 程式會顯示「實際取得的解析度」，因為相機可能不支援你選的某些 preset
- **Save dir**：照片儲存資料夾（預設 `./calib_imgs`）
- **Auto-advance on OK**：拍照後角點 OK 會自動跳到下一個拍攝步驟
- **Live corners preview**：即時角點偵測（可關閉以省 CPU）

### 4.2 Calibrate Settings
- **Pattern (cols, rows)**：內角點數
- **Auto pattern**：自動嘗試 `(cols,rows)/(rows,cols)/(cols-1,rows-1)...`
- **Square size (m)**：單格邊長（公尺）
- **Output JSON**：輸出 `K/distCoeffs` 的 JSON 檔案（預設 `fusion_camera.json`）

### 4.3 Captured images list
- 拍照後會新增一筆：`[OK]` 或 `[FAIL]`
- **右鍵點該筆** → Delete/Cancel（會刪硬碟檔案）
- 選取後按 **Delete 鍵** 也可刪除
- `Undo last`：刪除最後一張

---

## 5. How to Capture (拍照流程建議)

### 5.1 三個距離：近 / 中 / 遠（用「棋盤佔畫面寬度」判斷）
本工具用「棋盤在畫面中的寬度佔比」做判斷（比公分更可靠）：

- **近（Near）**：70% ~ 92%
- **中（Mid）**：40% ~ 65%
- **遠（Far）**：20% ~ 38%

LiveCorners 開啟時，你會看到：
- `DistOK (xx%)`：距離合格
- `TOO FAR (xx%)`：棋盤太小 → 請靠近
- `TOO CLOSE (xx%)`：棋盤太大 → 請拉遠

### 5.2 角度建議（相機固定也 OK）
你的相機固定沒問題：你只要 **移動/旋轉棋盤** 來產生視角變化。

建議範圍：
- **pitch（上下仰俯）**：約 ±10° ~ ±25°（最多約 ±30°）
- **yaw（左右轉）**：約 ±10° ~ ±25°
- **roll（平面旋轉）**：0° / 15° / 30° / 45°（越多樣越好）

LiveCorners 有顯示 `Roll≈+xx°`，在 roll 步驟會提示：
- `TargetRoll≈+15° (OK/ADJUST)` 等

### 5.3 建議拍多少張？
- 最低：**有效角點 ≥ 10 張**
- 建議：**15 ~ 30 張**（越多越穩，且角度/位置要多樣）

---

## 6. Calibrate (輸出 K / distCoeffs)

1. 拍完後按：**✅ Calibrate (K/dist)**
2. 右側結果會顯示：
   - `K`（3x3）
   - `distCoeffs`（通常為 5 或更多係數，依 OpenCV 版本/模型）
   - `RMS (cv2)` / `Reproj RMSE`
3. 同時會輸出 JSON 到你指定的檔案（預設 `fusion_camera.json`）

> 注意：`K/distCoeffs` **與解析度綁定**。  
> 若你之後改用另一個解析度，建議重新拍一套並重新校正。

---

## 7. Validate (快速判斷是否需要補拍)

按 **🔎 Validate (RMSE)**，工具會用簡單規則提示：
- RMSE ≤ 1.0 且 used_images ≥ 15 → 多數情況「夠用」
- 否則建議：再補拍、提高畫質、增加角度與位置多樣性

---

## 8. Troubleshooting

### 8.1 為什麼拍照後常常 `[FAIL]`？
最常見原因：
- 棋盤太小（太遠）→ 靠近、讓棋盤佔畫面更大
- 模糊（手震/對焦不準）→ 提高光線、避免移動、等待對焦完成
- 反光/曝光太強 → 換角度、降低反光、避免亮面覆膜
- 棋盤被裁切 → 棋盤四邊要完整入鏡

### 8.2 Pattern 10x7 填了還是抓不到？
你很可能拿的是 **10x7 方格**，OpenCV 需要的是 **9x6 內角點**。  
請開啟 **Auto pattern** 或直接改填 9x6。

### 8.3 我換了解析度，校正結果怪怪的？
請不要混用不同解析度的照片。  
本工具會在校正時自動略過「尺寸不一致」的照片，但最佳做法是：
- 每個解析度用 **獨立資料夾** 拍一套

---

## 9. Output JSON Example

輸出的 JSON 會包含（示意）：

```json
{
  "ok": true,
  "image_size": [1280, 720],
  "pattern_size_used": [9, 6],
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distCoeffs": [k1, k2, p1, p2, k3],
  "rms": 0.32,
  "reproj_rmse": 0.58,
  "used_images": 18,
  "total_images": 22
}
```

---

## 10. Tips (讓 K/dist 更穩的關鍵)

- 棋盤要 **大、清楚、無反光**
- 位置要分散（中心、四角）
- 角度要多樣（pitch/yaw/roll 都要有）
- 不要只拍一種距離或只拍正面
- 同一資料集不要混解析度

---

## License
依你的專案需求自行加入（MIT/Apache-2.0/Proprietary…）。

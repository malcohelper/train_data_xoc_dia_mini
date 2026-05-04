# XocDia – Hướng dẫn cài đặt & sử dụng

App: `XocDia.app` (macOS, Apple Silicon, 12.3+).
Cách phân phối: nhận `XocDia.zip` từ admin → giải nén → kéo `XocDia.app`
vào `/Applications`.

## 1. Yêu cầu hệ thống

| Yêu cầu | Chi tiết |
|---------|----------|
| Hệ điều hành | macOS 12.3 (Monterey) hoặc mới hơn |
| Chip | Apple Silicon (M1 / M2 / M3 / M4). Intel Mac **không chạy được**. |
| RAM | 8 GB trở lên |
| Ổ đĩa trống | 2 GB (bao gồm ~300 MB model PaddleOCR tải lần đầu) |
| Internet | Cần khi chạy lần đầu (download PaddleOCR models). Sau đó offline OK. |
| Trình duyệt | Safari (đã đăng nhập tài khoản chơi xóc đĩa) |

## 2. Cài đặt lần đầu

### Bước 1 — giải nén

Double-click `XocDia.zip` → Finder tạo `XocDia.app`. Kéo file đó vào
`/Applications` (thư mục Applications của Mac).

### Bước 2 — mở app lần đầu (vượt cảnh báo Gatekeeper)

App **chưa được Apple ký số** nên macOS sẽ chặn lần đầu. Cách bypass:

1. Vào `/Applications`, tìm `XocDia.app`.
2. **Chuột phải** (hoặc Control-click) → chọn **Open**.
3. Hộp thoại sẽ hỏi *"Apple could not verify XocDia is free of
   malware..."* → bấm **Open Anyway**.
4. Lần sau double-click bình thường, Mac không hỏi lại nữa.

### Bước 3 — cấp quyền Screen Recording (BẮT BUỘC)

App cần quyền *Screen Recording* để đọc nội dung Safari.

1. Mở **System Settings** (logo Apple → System Settings).
2. Vào **Privacy & Security** → **Screen Recording**.
3. Tìm **XocDia** trong danh sách, bật toggle (xanh).
   - Nếu chưa thấy XocDia: chạy app 1 lần, macOS sẽ tự thêm vào list,
     sau đó quay lại bật toggle.
4. **Quit XocDia hoàn toàn** (Cmd-Q) rồi mở lại để quyền có hiệu lực.

### Bước 4 — chạy lần đầu

1. Mở Safari, đăng nhập game, vào trang xóc đĩa mini.
2. Mở `XocDia.app` (double-click).
3. Vài giây sau xuất hiện hộp thoại **"Pick Window or Drag ROI"** —
   bấm **Pick Window**.
4. Hộp thoại tiếp theo liệt kê các cửa sổ đang mở — chọn dòng có chữ
   **"Safari - XÓC ĐĨA MINI"** → bấm **OK**.
5. Cửa sổ preview "XocDia Preview" hiện ra với hình game được bao
   bằng các bbox màu — tức là nhận diện đang hoạt động.
6. App tự động ghi log mỗi round vào
   `~/Documents/XocDia/rounds/<timestamp>.json`.

## 3. Sử dụng hằng ngày

* Mở Safari → trang game → đăng nhập (giữ tab này active).
* Mở `XocDia.app` → Pick Window → chọn Safari.
* Để app chạy nền, cứ chơi bình thường. Mỗi round kết thúc, app ghi
  một file JSON vào `~/Documents/XocDia/rounds/`.

### Phím tắt khi cửa sổ "XocDia Preview" đang focus

| Phím | Tác dụng |
|------|----------|
| `r` | Pick lại window khác (trường hợp đổi tab/refresh Safari) |
| `c` | Tự co vùng nhận diện về bbox UI nhỏ nhất |
| `d` | Bật/tắt overlay diagnostics |
| `s` | Lưu frame hiện tại thành PNG (debug) |
| `q` | Thoát app |

## 4. Xem kết quả các round

Mỗi round tạo 1 file JSON, ví dụ
`~/Documents/XocDia/rounds/20260430_011604.json`:

```json
{
  "round_id": "20260430_011604",
  "dice_result": "2w_2r",
  "bets": {
    "chan":   { "total_bet": "8.88M",  "count": "326", "percent": "49%" },
    "le":     { "total_bet": "16.39M", "count": "166", "percent": "51%" },
    "4_red":  { "total_bet": "4.21M",  "count": "205", "percent": "-"   },
    "4_white":{ "total_bet": "3.52M",  "count": "208", "percent": "15%" },
    "3r_1w":  { "total_bet": "3.13M",  "count": "120", "percent": "33%" },
    "3w_1r":  { "total_bet": "3.35M",  "count": "12",  "percent": "-"   }
  }
}
```

(Đang chuẩn bị tooling để hiển thị các round dưới dạng dashboard ngay
trong app — tạm thời mở file JSON bằng bất kỳ editor nào.)

## 5. Sự cố thường gặp

### App mở rồi tự thoát ngay

Mở **Console.app** (Spotlight: `Console`) → tab **Log Reports** → tìm
file `~/Library/Logs/XocDia/<timestamp>.log` mới nhất. Đọc dòng cuối
sẽ thấy lỗi cụ thể. Gửi cho admin.

### Preview hiện ra nhưng `dets=0` mãi (không nhận diện được)

Nguyên nhân hay gặp:

1. **Chưa cấp Screen Recording permission** → quay lại Bước 3 mục
   "Cài đặt lần đầu", bật toggle, rồi **Cmd-Q quit và mở lại app**.
2. **Pick nhầm tab Safari khác** → bấm `r` trong preview window, chọn
   lại tab đúng (có chữ "XÓC ĐĨA MINI" trong tiêu đề).
3. **Safari mới refresh, chưa load xong game** → đợi game hiển thị
   bàn cờ rồi bấm `r` để pick lại.

### "App is damaged and can't be opened" khi mở

Đây là khi macOS quarantine flag bị set (thường do download qua
trình duyệt cũ). Mở Terminal, chạy:

```bash
xattr -dr com.apple.quarantine /Applications/XocDia.app
```

Sau đó mở lại app như bình thường.

### App chạy chậm / lag preview

Đây là CPU-bound (YOLO + PaddleOCR). Trên M1 cơ bản app chạy ~1 detect/giây
là bình thường. Đóng các app nặng khác (Chrome, Slack, Zoom) để giải
phóng RAM.

### Round JSON không được tạo dù app đang chạy

Round chỉ ghi khi state-machine **hoàn tất 1 round** (timer chạy từ
~46s xuống 0 → dice xuất hiện). Nếu chỉ chạy 30s rồi đổi tab, không có
round nào được lưu cả. Đợi ít nhất 1 phút để round đầu tiên hoàn tất.

## 6. Gỡ cài đặt

```bash
# Xoá app
rm -rf /Applications/XocDia.app

# Xoá data app đã ghi (giữ rounds nếu muốn dùng tiếp)
rm -rf ~/Documents/XocDia
rm -rf ~/Library/Logs/XocDia

# (tùy chọn) xoá PaddleOCR models đã download
rm -rf ~/.paddlex
```

Quyền Screen Recording vẫn còn trong System Settings — gỡ thủ công nếu
cần (System Settings → Privacy & Security → Screen Recording → tìm
XocDia → trừ icon "−").

## 7. Mô phỏng Backtest trên thư mục `rounds/` (Node.js)

Dùng khi bạn đã có **nhiều file JSON** (ví dụ copy từ
`~/Documents/XocDia/rounds/` hoặc dùng thư mục `rounds/` trong repo dev).
Script **không cần `npm install`**, chỉ cần **Node.js 18+** (có sẵn `node`).

### Cách chạy (từ thư mục gốc của project)

```bash
# Chạy mặc định: đọc ./rounds, burn-in 40, ghi analytics/backtest-report.json
node analytics/run-backtest-rounds.js
```

Các tùy chọn thường dùng:

```bash
# Chỉ định file báo cáo đầu ra
node analytics/run-backtest-rounds.js --out ./analytics/backtest-report.json

# Đổi số phiên “nạp đà” (mặc định 40; từ phiên 41 trở đi mới tính %)
node analytics/run-backtest-rounds.js --burn-in 60

# Trỏ sang thư mục JSON khác (vd. bản copy từ máy chơi)
node analytics/run-backtest-rounds.js --rounds /đường/dẫn/tới/rounds

# Thử nghiệm: tăng β (đè bẹp thuật toán yếu mạnh hơn)
node analytics/run-backtest-rounds.js --beta 4

# Trả H về dạng có trộn cả tỉ lệ đúng “vị” (không chỉ Chẵn/Lẻ)
node analytics/run-backtest-rounds.js --hit-blend-exact 0.55
```

### Script làm gì (tóm tắt)

1. **Đọc** mọi `*.json` trong thư mục, **sắp xếp theo tên file** (định dạng
   `YYYYMMDD_HHmmss` → thứ tự đúng thời gian).
2. **Chuyển** mỗi file sang `RoundItem` (trường `red`, `type`, `time`,
   `percent`, `bets`, …) giống chuẩn `prediction-engine.js`.
3. **Tạm thời** chỉnh engine cho lần chạy này: mặc định **`HIT_BLEND_EXACT = 0`**
   (điểm H trong ensemble chỉ dựa **Chẵn/Lẻ**, không nhấn “đúng vị”), và
   **`BETA = 2`**. Sau khi xong, giá trị trong `prediction-engine.js` **được khôi phục** —
   không làm bẩn cấu hình khi bạn mở analytics trên trình duyệt.
4. Gọi **walk-forward** `runBacktest` + **`runBaselines`** (random / lặp lại cửa trước).
5. In kết quả ra **terminal** và ghi **JSON** (động / tĩnh, `byAlgo`, file lỗi nếu có).

### Lưu ý quan trọng

- **Không phải lời khuyên cờ bạc.** Backtest chỉ đo mô hình heuristic trên
  dữ liệu đã ghi; quá khứ **không** đảm bảo tương lai.
- **Chất lượng dữ liệu:** file thiếu `dice_result` hoặc sai khóa sẽ vào mục
  `skipped` trong báo cáo — nên kiểm tra nếu số “bỏ qua” lớn.
- **Cùng một bộ rounds:** so sánh Động vs Tĩnh và các TT mới có ý nghĩa khi
  mọi lần chạy dùng **cùng thư mục và cùng thứ tự** file.
- **Thời gian chạy:** vài trăm đến vài nghìn file có thể mất vài giây đến vài
  chục giây tùy máy (mỗi bước gọi lại ensemble).
- **Tham số `--beta` / `--hit-blend-exact`:** chỉ áp trong **phiên chạy script**;
  nếu muốn đổi mặc định cho UI, phải sửa **DYNAMIC_ENSEMBLE** trong
  `analytics/prediction-engine.js` riêng.

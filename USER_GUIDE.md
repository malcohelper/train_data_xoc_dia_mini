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

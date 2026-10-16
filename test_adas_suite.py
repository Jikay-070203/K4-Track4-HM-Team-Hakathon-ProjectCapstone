"""
VinFast ADAS - Pedestrian Collision Warning (PCW)
10 Comprehensive Automated Test Cases
Track 4 Hackathon Capstone (Evidence & Verification Suite)
"""

import unittest
import numpy as np
import time

class Kalman1DDistance:
    """1D Kalman Filter to track distance and estimate relative velocity"""
    def __init__(self, init_d=3.0, process_noise=0.1, measurement_noise=0.25):
        self.d = init_d
        self.v = 0.0
        self.last_t = time.time()
        self.P = np.array([[1.0, 0.0], [0.0, 2.0]], dtype=float)
        self.Q = np.array([[process_noise, 0.0], [0.0, process_noise * 2]], dtype=float)
        self.R = float(measurement_noise)

    def update(self, measured_d):
        now = time.time()
        dt = max(1e-3, now - self.last_t)
        self.last_t = now

        pred_d = self.d + self.v * dt
        pred_v = self.v
        
        P00 = self.P[0,0] + dt * (self.P[1,0] + self.P[0,1]) + dt*dt * self.P[1,1] + self.Q[0,0]
        P01 = self.P[0,1] + dt * self.P[1,1] + self.Q[0,1]
        P10 = self.P[1,0] + dt * self.P[1,1] + self.Q[1,0]
        P11 = self.P[1,1] + self.Q[1,1]

        y = measured_d - pred_d
        S = P00 + self.R
        K0 = P00 / S
        K1 = P10 / S

        self.d = pred_d + K0 * y
        self.v = pred_v + K1 * y

        self.P[0,0] = (1 - K0) * P00
        self.P[0,1] = (1 - K0) * P01
        self.P[1,0] = -K1 * P00 + P10
        self.P[1,1] = -K1 * P01 + P11

        return self.d, self.v

class ADASGeometryHelper:
    """Lightweight test harness for ADAS Geometry and Threat Logic"""
    def __init__(self, focal_length=680.0, real_height=1.70):
        self.focal_length = focal_length
        self.real_height = real_height

    def estimate_distance(self, bbox_height_px):
        if bbox_height_px <= 5:
            return 99.0
        return (self.focal_length * self.real_height) / bbox_height_px

    def check_in_corridor(self, foot_x, foot_y, w=1280, h=720):
        top_y = int(h * 0.48)
        top_w = int(w * 0.26)
        bot_y = h
        bot_w = int(w * 0.70)
        cx = w // 2
        top_left = cx - top_w // 2
        top_right = cx + top_w // 2
        bot_left = cx - bot_w // 2
        bot_right = cx + bot_w // 2

        if foot_y < top_y or foot_y > bot_y:
            return False
        ratio = (foot_y - top_y) / (bot_y - top_y)
        cur_left = top_left + ratio * (bot_left - top_left)
        cur_right = top_right + ratio * (bot_right - top_right)
        return cur_left <= foot_x <= cur_right

    def classify_threat(self, d, ttc, in_corridor):
        if in_corridor and (d <= 2.2 or ttc <= 1.8):
            return "CRITICAL"
        elif in_corridor and (d <= 4.0 or ttc <= 3.0):
            return "WARNING"
        else:
            return "SAFE"


class TestVinFastADASSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geo = ADASGeometryHelper(focal_length=680.0, real_height=1.70)

    # -------------------------------------------------------------
    # NHÓM 1: TRỤ CỘT HÌNH HỌC (GEOMETRY PILLAR)
    # -------------------------------------------------------------
    def test_TC01_pinhole_accuracy(self):
        """TC01: Kiểm tra độ chính xác mô hình Pinhole tại các cự ly chuẩn 1m, 2m, 3m"""
        # Công thức: h = (f * H) / d -> với f=680, H=1.70
        # Tại d=1.0m -> h=1156px; tại d=2.0m -> h=578px; tại d=3.0m -> h=385px
        gt_distances = [1.0, 2.0, 3.0]
        for d_true in gt_distances:
            h_px = (self.geo.focal_length * self.geo.real_height) / d_true
            d_est = self.geo.estimate_distance(h_px)
            err = abs(d_est - d_true)
            self.assertAlmostEqual(d_est, d_true, delta=0.05, 
                                   msg=f"Sai số khoảng cách tại {d_true}m vượt quá 5cm (Thực tế sai: {err:.4f}m)")

    def test_TC02_pinhole_edge_cases(self):
        """TC02: Kiểm tra xử lý biên chống chia cho 0 khi box rỗng hoặc siêu nhỏ"""
        # Box h <= 5px phải trả về giá trị sentinel 99.0m, không văng ngoại lệ
        self.assertEqual(self.geo.estimate_distance(0), 99.0)
        self.assertEqual(self.geo.estimate_distance(-10), 99.0)
        self.assertEqual(self.geo.estimate_distance(4), 99.0)

    # -------------------------------------------------------------
    # NHÓM 2: TRỤ CỘT BẤT ĐỊNH & THỜI GIAN (KALMAN & TEMPORAL)
    # -------------------------------------------------------------
    def test_TC03_kalman_noise_reduction(self):
        """TC03: Kiểm tra bộ lọc Kalman giảm phương sai nhiễu đo đạc (ít nhất 10%)"""
        np.random.seed(42)
        true_dist = 3.0
        # Tạo chuỗi đo bị nhiễu Gauss +- 0.35m
        noisy_measurements = true_dist + np.random.normal(0, 0.35, size=50)
        
        kf = Kalman1DDistance(init_d=true_dist, process_noise=0.05, measurement_noise=0.35)
        filtered_results = []
        for z in noisy_measurements:
            d_smooth, _ = kf.update(z)
            filtered_results.append(d_smooth)
            time.sleep(0.002)

        raw_var = np.var(noisy_measurements)
        filtered_var = np.var(filtered_results)
        reduction_pct = (1.0 - filtered_var / raw_var) * 100.0

        self.assertGreater(reduction_pct, 10.0, 
                           f"Kalman không giảm đủ 10% nhiễu. Thực tế giảm: {reduction_pct:.2f}%")

    def test_TC04_kalman_velocity_tracking(self):
        """TC04: Kiểm tra khả năng ước lượng vận tốc tiếp cận (v_rel âm khi tiến lại gần)"""
        kf = Kalman1DDistance(init_d=5.0)
        # Giả lập người đi từ 5m về 2m trong 10 bước
        distances = np.linspace(5.0, 2.0, 10)
        final_v = 0.0
        for d in distances:
            _, final_v = kf.update(d)
            time.sleep(0.01)

        self.assertLess(final_v, 0.0, 
                        f"Vận tốc tiếp cận phải mang giá trị âm khi người tiến lại gần. Thực tế: {final_v:.2f}")

    # -------------------------------------------------------------
    # NHÓM 3: HÀNH LANG LÁI XE HÌNH THANG (HAZARD CORRIDOR)
    # -------------------------------------------------------------
    def test_TC05_corridor_in_path(self):
        """TC05: Người đứng ngay giữa làn đường xe chạy phải được nhận diện trong hành lang"""
        w, h = 1280, 720
        ped_center_x = w // 2 # 640
        ped_foot_y = int(h * 0.85) # 612
        is_in = self.geo.check_in_corridor(ped_center_x, ped_foot_y, w, h)
        self.assertTrue(is_in, "Người đứng ở giữa làn đường xe chạy bắt buộc phải nằm trong Corridor!")

    def test_TC06_corridor_off_path(self):
        """TC06: Người đứng ngoài vỉa hè hoặc mép trái/phải phải được loại trừ khỏi hành lang"""
        w, h = 1280, 720
        # Người đứng sát mép trái (x = 50px)
        is_in_left = self.geo.check_in_corridor(50, int(h * 0.85), w, h)
        # Người đứng sát mép phải (x = 1230px)
        is_in_right = self.geo.check_in_corridor(1230, int(h * 0.85), w, h)
        # Người ở trên đường chân trời (y < 48% h)
        is_above = self.geo.check_in_corridor(w // 2, int(h * 0.3), w, h)

        self.assertFalse(is_in_left, "Người ở mép vỉa hè bên trái không được coi là trong làn va chạm!")
        self.assertFalse(is_in_right, "Người ở mép vỉa hè bên phải không được coi là trong làn va chạm!")
        self.assertFalse(is_above, "Người ở ngoài phạm vi chân trời không được tính vào hành lang!")

    # -------------------------------------------------------------
    # NHÓM 4: PHÂN TẦNG CẢNH BÁO AN TOÀN ADAS (THREAT ASSESSMENT)
    # -------------------------------------------------------------
    def test_TC07_threat_level_safe(self):
        """TC07: Trạng thái AN TOÀN (SAFE) khi người ở xa hoặc ngoài hành lang"""
        # Ngoài hành lang -> luôn SAFE
        self.assertEqual(self.geo.classify_threat(d=2.0, ttc=1.0, in_corridor=False), "SAFE")
        # Trong hành lang nhưng d > 4m và TTC > 3.0s -> SAFE
        self.assertEqual(self.geo.classify_threat(d=5.5, ttc=3.8, in_corridor=True), "SAFE")

    def test_TC08_threat_level_warning(self):
        """TC08: Trạng thái CẢNH BÁO SỚM (WARNING) khi người trong làn ở cự ly 2.2 - 4.0m"""
        threat = self.geo.classify_threat(d=3.2, ttc=2.4, in_corridor=True)
        self.assertEqual(threat, "WARNING", "Ở cự ly 3.2m và TTC 2.4s, hệ thống phải kích hoạt WARNING!")

    def test_TC09_threat_level_critical(self):
        """TC09: Trạng thái PHANH KHẨN CẤP (CRITICAL) khi người lọt vào cự ly <= 2.2m hoặc TTC <= 1.8s"""
        threat_dist = self.geo.classify_threat(d=1.8, ttc=2.5, in_corridor=True)
        threat_ttc = self.geo.classify_threat(d=3.0, ttc=1.2, in_corridor=True)
        self.assertEqual(threat_dist, "CRITICAL", "Cự ly <= 2.2m trong làn bắt buộc phải kích hoạt CRITICAL!")
        self.assertEqual(threat_ttc, "CRITICAL", "TTC <= 1.8s trong làn bắt buộc phải kích hoạt CRITICAL!")

    # -------------------------------------------------------------
    # NHÓM 5: MÔ PHỎNG CA LỖI BẮT BUỘC (FAILURE MODE ANALYSIS)
    # -------------------------------------------------------------
    def test_TC10_failure_mode_occlusion(self):
        """TC10: Kiểm chứng ca lỗi che khuất nửa thân dưới (False Safe Bias)"""
        # Giả lập người thực tế đứng ở d = 2.0m -> chiều cao chuẩn h = 578px
        d_actual = 2.0
        h_full = (self.geo.focal_length * self.geo.real_height) / d_actual

        # Khi bị che nửa thân dưới (ví dụ đứng sau xe khác), box h giảm 50%
        h_occluded = h_full * 0.50
        d_measured = self.geo.estimate_distance(h_occluded)

        # Tính toán độ lệch: d_measured sẽ gấp đôi d_actual (False Safe)
        ratio = d_measured / d_actual
        self.assertAlmostEqual(ratio, 2.0, delta=0.05, 
                               msg=f"Mô hình Pinhole khi che khuất 50% chiều cao phải tính sai gấp đôi khoảng cách! Thực tế tỷ số: {ratio:.2f}")


if __name__ == "__main__":
    print("=" * 70)
    print("   VINFAST ADAS - PEDESTRIAN COLLISION WARNING (PCW) TEST SUITE   ")
    print("     10 Comprehensive Verification Test Cases (Track 4 Capstone)  ")
    print("=" * 70)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVinFastADASSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f" >>> [KẾT QUẢ]: TẤT CẢ 10/10 TEST CASES ĐỀU ĐẠT CHUẨN (PASSED)! <<<")
        print("  1. Hình học Pinhole chuẩn xác ở các cự ly 1m - 3m")
        print("  2. Bảo vệ chống lỗi chia 0 (Division by Zero Protected)")
        print("  3. Bộ lọc Kalman giảm nhiễu đo đạc và bám vận tốc thành công")
        print("  4. Hành lang lái xe Hazard Corridor phân biệt đúng làn đường")
        print("  5. Phân tầng cảnh báo SAFE / WARNING / CRITICAL chuẩn xác")
        print("  6. Tái hiện ca lỗi che khuất (Occlusion Failure Case) thành công")
    else:
        print(f" >>> [KẾT QUẢ]: CÓ {len(result.failures)} TEST FAILED, {len(result.errors)} LỖI! <<<")
    print("=" * 70 + "\n")

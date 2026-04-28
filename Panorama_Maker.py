import cv2
import numpy as np
import math

# 1. 이미지를 원통형 좌표계로 변형하는 함수 (핵심 추가 기능)
def cylindrical_warp(img, focal_length=None):
    h, w = img.shape[:2]
    
    # 초점 거리(f)가 주어지지 않으면 대략적인 값(이미지 너비)으로 추정합니다.
    # (실제 완벽한 투영을 위해서는 카메라의 내부 파라미터가 필요하지만, 
    # 일반적인 스마트폰 사진에서는 f = width 정도면 적절히 작동합니다.)
    if focal_length is None:
        f = w 
    else:
        f = focal_length

    # 원통형 좌표계를 평면으로 펼치기 위한 좌표 매핑 (Backward Warping)
    y_i, x_i = np.indices((h, w))
    X = (x_i - w / 2.0)
    Y = (y_i - h / 2.0)

    # 3D 원통 상의 좌표 계산
    theta = X / f
    X_cyl = np.sin(theta)
    Z_cyl = np.cos(theta)
    Y_cyl = Y / f

    # 다시 2D 소스 이미지 좌표로 투영
    x_src = (f * X_cyl / Z_cyl) + w / 2.0
    y_src = (f * Y_cyl / Z_cyl) + h / 2.0

    # OpenCV remap을 사용하여 픽셀 값을 빠르게 매핑
    map_x = x_src.astype(np.float32)
    map_y = y_src.astype(np.float32)
    
    warped_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    return warped_img

def stitch_multiple_images(image_paths):
    # 이미지 로드
    raw_images = [cv2.imread(path) for path in image_paths]
    if any(img is None for img in raw_images):
        print("경로에 이미지가 없습니다. 파일 경로를 다시 확인해주세요.")
        return None

    # ★ 가산점 포인트: 모든 이미지를 매칭 전에 원통형으로 변환합니다.
    print("원통형 투영(Cylindrical Warping) 진행 중...")
    images = [cylindrical_warp(img) for img in raw_images]

    sift = cv2.SIFT_create()

    base_h, base_w = images[0].shape[:2]
    canvas_h = base_h * 2 
    canvas_w = base_w * len(images)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    start_y = canvas_h // 4
    canvas[start_y:start_y+base_h, 0:base_w] = images[0]

    kp_prev, des_prev = sift.detectAndCompute(images[0], None)

    # 초기 누적 Homography 행렬 (Y축 이동 포함)
    H_acc = np.array([
        [1, 0, 0],
        [0, 1, start_y],
        [0, 0, 1]
    ], dtype=np.float64)

    for i in range(1, len(images)):
        print(f"이미지 {i+1} 정합 중...")
        img_curr = images[i]
        kp_curr, des_curr = sift.detectAndCompute(img_curr, None)

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_curr, des_prev, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10:
            src_pts = np.float32([kp_curr[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_prev[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            H_curr, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # 누적 행렬곱 계산
            H_acc = np.matmul(H_acc, H_curr)

            # 캔버스에 현재 이미지 워핑
            warped_img = cv2.warpPerspective(img_curr, H_acc, (canvas_w, canvas_h))

            # 검은색 배경을 제외하고 캔버스에 덮어쓰기
            warp_mask = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY) > 0
            canvas[warp_mask] = warped_img[warp_mask]

            kp_prev, des_prev = kp_curr, des_curr
        else:
            print(f"매칭점이 부족하여 스티칭 중단 (이미지 {i+1})")
            break

    # (선택) 결과 이미지의 검은색 불필요한 여백을 대략적으로 잘라냅니다.
    # 결과를 더 깔끔하게 보기 위한 후처리입니다.
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_canvas, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        canvas = canvas[y:y+h, x:x+w]

    print("스티칭 완료!")
    return canvas

if __name__ == "__main__":
    # 요청하신 이미지 경로 반영
    image_list = [
        "image/image_01.jpg", 
        "image/image_02.jpg", 
        "image/image_03.jpg", 
        "image/image_04.jpg"
    ] 
    
    result = stitch_multiple_images(image_list)

    if result is not None:
        cv2.imwrite("final_cylindrical_panorama.jpg", result)
        cv2.namedWindow("Cylindrical Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Cylindrical Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
import cv2
import numpy as np

def stitch_multiple_images(image_paths):
    images = [cv2.imread(path) for path in image_paths]
    if len(images) < 3:
        print("과제 조건에 따라 최소 3장 이상의 이미지가 필요합니다.")
        return None

    sift = cv2.SIFT_create()

    # 1. 넉넉한 배경 캔버스 생성 (오른쪽으로 이어붙인다고 가정)
    base_h, base_w = images[0].shape[:2]
    canvas_h = base_h * 2  # 위아래로 어긋날 것을 대비해 세로 캔버스 크기를 2배로 늘림
    canvas_w = base_w * len(images)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # 첫 번째 이미지를 캔버스 좌측 중앙쯤에 배치
    start_y = canvas_h // 4
    canvas[start_y:start_y+base_h, 0:base_w] = images[0]

    # 2. 첫 번째 이미지를 기준(Base)으로 설정하고, 특징점 추출
    kp_prev, des_prev = sift.detectAndCompute(images[0], None)

    # 3. ★ 핵심: 누적 Homography 행렬 초기화 (단위 행렬) ★
    # 파이썬(Numpy)에서는 np.eye(3)을 통해 단위 행렬을 만듭니다.
    # 단, 우리가 첫 번째 이미지를 캔버스의 start_y 위치로 옮겼기 때문에, 
    # 그 Y축 이동(Translation) 값을 포함한 초기 행렬을 만들어줍니다.
    H_acc = np.array([
        [1, 0, 0],
        [0, 1, start_y],
        [0, 0, 1]
    ], dtype=np.float64)

    for i in range(1, len(images)):
        img_curr = images[i]
        kp_curr, des_curr = sift.detectAndCompute(img_curr, None)

        # FLANN 매칭
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des_curr, des_prev, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10:
            # 현재 이미지(i)를 이전 이미지(i-1)에 맞추기 위한 점들
            src_pts = np.float32([kp_curr[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_prev[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 이전 이미지와 현재 이미지 간의 Homography 계산 (H_curr)
            H_curr, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # ★ 핵심: 행렬 곱셈을 통한 누적 Homography 계산 ★
            # 이전까지의 누적 변환(H_acc)에 현재 변환(H_curr)을 곱해서 누적시킵니다.
            # 파이썬에서 행렬 곱셈은 np.matmul() 또는 '@' 기호를 사용합니다.
            H_acc = np.matmul(H_acc, H_curr)

            # 누적된 H 행렬을 사용하여 현재 이미지를 '첫 번째 이미지의 시점(캔버스)'에 맞게 Warping
            warped_img = cv2.warpPerspective(img_curr, H_acc, (canvas_w, canvas_h))

            # 캔버스에 덮어쓰기 (검은색 빈 공간은 덮어쓰지 않도록 마스크 처리)
            warp_mask = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY) > 0
            canvas[warp_mask] = warped_img[warp_mask]

            # 다음 루프를 위해 현재 이미지의 특징점을 '이전 특징점'으로 업데이트
            kp_prev, des_prev = kp_curr, des_curr
            print(f"이미지 {i+1} 스티칭 완료!")
        else:
            print(f"매칭점이 부족하여 스티칭 중단 (이미지 {i+1})")
            break

    return canvas

if __name__ == "__main__":
    # 사진 찍으신 순서대로 파일명을 넣어주세요 (왼쪽 -> 오른쪽 방향 권장)
    image_list = ["image/image_01.jpg", "image/image_02.jpg", "image/image_03.jpg", "image/image_04.jpg"]
    
    result = stitch_multiple_images(image_list)

    if result is not None:
        cv2.imwrite("final_panorama.jpg", result)
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
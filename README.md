# 📸 Panorama_Maker

### 📝 프로젝트 설명
OpenCV와 Python을 활용하여 여러 장의 이미지를 자동으로 정합(Stitching)하고 하나의 광각 파노라마 이미지를 생성하는 프로그램입니다. 

이 프로젝트는 OpenCV의 High-level API(`cv2.Stitcher`)를 사용하지 않고, 컴퓨터 비전의 핵심 원리인 **특징점 추출(Feature Extraction), 매칭(Matching), 원근 변환(Homography), 이미지 워핑(Warping)** 과정을 직접 파이썬 코드로 구현하는 것을 목적으로 합니다.

---

### ✨ 주요 기능 및 특징
1. **SIFT 특징점 검출 및 FLANN 매칭**
    * 이미지 간의 공통된 특징을 찾기 위해 SIFT(Scale-Invariant Feature Transform) 알고리즘을 사용합니다.
    * 대량의 특징점을 효율적으로 매칭하기 위해 FLANN(Fast Library for Approximate Nearest Neighbors) 기반 매처를 적용했습니다.
    * Lowe's Ratio Test를 통해 잘못된 매칭을 제거하고 신뢰도 높은 포인트만 선별합니다.

2. **누적 Homography 행렬 계산 (Matrix Multiplication)**
    * **구현 핵심:** 3장 이상의 이미지를 정합할 때 발생하는 오차 누적을 방지하기 위해, 첫 번째 이미지를 기준 좌표계(단위 행렬)로 설정합니다.  
    * 각 단계에서 계산된 $H$ 행렬을 이전의 누적 행렬과 행렬곱($H_{acc} = H_{acc} \times H_{curr}$)하여 모든 이미지를 하나의 통합된 캔버스 좌표계로 투영합니다.

3. **원통형 투영 (Cylindrical Projection) 적용**
    * **왜곡 방지:** 일반적인 평면 투영(Planar View)에서 화각이 넓어질수록 이미지 가장자리가 심하게 늘어나는 현상을 해결하기 위해 원통형 투영 기법을 적용했습니다.  
    * 이미지 매칭 전, 모든 이미지를 가상의 원통 좌표계로 워핑(Warping)하여 더 자연스럽고 넓은 시야각의 파노라마를 생성합니다.

4. **대용량 캔버스 및 마스크 합성**
    * 여러 장의 이미지가 합쳐질 때 잘리지 않도록 동적인 캔버스를 생성합니다.
    * 이미지 간 겹치는 부분에서 검은색 배경이 덮어씌워지지 않도록 마스크(Masking) 처리를 통해 자연스럽게 병합합니다.

---

### 📊 결과물
* **Input Images:**  
![image_01](https://github.com/park2722/Panorama-Maker/blob/main/image/image_01.jpg)  
![image_02](https://github.com/park2722/Panorama-Maker/blob/main/image/image_02.jpg)
![image_03](https://github.com/park2722/Panorama-Maker/blob/main/image/image_03.jpg)
![image_04](https://github.com/park2722/Panorama-Maker/blob/main/image/image_04.jpg)

* **Output Panorama:**  
![result_panorama](https://github.com/park2722/Panorama-Maker/blob/main/final_cylindrical_panorama.jpg)

from django.shortcuts import render
from django.http import HttpResponse
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import matplotlib.image as img
from PIL import Image
from Levenshtein import distance
import re
from PIL import Image, ImageDraw

def mapApi(request):
    return render(request, 'map/map_api.html')

def detail(request):
    return render(request, 'map/detail.html')

def homeinfo(request):
    return render(request, 'map/homeinfo.html')


def process_pdf(request):
    if request.method == 'POST':
        homeinfo_pdf = request.FILES.get('pdfFileInput')
        if homeinfo_pdf:
            with open('saved_pdf.pdf', 'wb') as f:
                f.write(homeinfo_pdf.read())

        # pdf to img
        pages = convert_from_path('saved_pdf.pdf')
        page_num = 0

        # pdf to image
        for i, page in enumerate(pages):
            page.save('image' + str(i + 1) + '.jpg', 'JPEG')
            page_num = page_num + 1

        image = []
        for i in range(page_num):
            encoded_img = cv2.imread('image1.jpg', cv2.IMREAD_COLOR)

            ## 이미지 전처리
            # Watermark delete
            _, sample = cv2.threshold(encoded_img, 150, 255, cv2.THRESH_BINARY)
            image.append(sample)

            # gray scale
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 이미지 사이즈 변경
            img_gray = cv2.resize(img_gray, (2480, 3508))
            
            # 이진화 변환
            _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # denoising
            # denoised = cv2.fastNlMeansDenoising(gray_enlarge, height*2, searchWindowSize=21, templateWindowSize=7)
            denoised = cv2.medianBlur(binary, 1)
            
            # 글자 강조
            kernel_size_row = 3
            kernel_size_col = 3
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(denoised, kernel, iterations=1)
            
            # Enlarge
            height, width = img_gray.shape
            enlarge = cv2.resize(denoised, (3*width, 3*height), interpolation=cv2.INTER_LINEAR)
            
            # sharpening
            kernel_sharpening = np.array([[-1, -1, -1],
                                           [-1, 9, -1],
                                           [-1, -1, -1]])
            sharpened = cv2.filter2D(denoised, -1, kernel_sharpening)
            
            # Inverting -> 순서 맨 끝으로 할지 말지
            sharpened[0:-1] = ~sharpened[0:-1]
            
            # OCR 실행
            text = pytesseract.image_to_string(sharpened, lang='kor')
            
            # 특정 단어와 유사한 단어 찾기
            target_word = "대지권비율"
            similar_word = find_similar_word(target_word, text)
            
            # 단어의 위치 찾기
            word_locations = find_word_location(similar_word, text)
            
            # 이미지 로드
            org_image = Image.open('image' + str(i + 2) + '.jpg')
            
            # 이미지에 사각형 그리기
            ############ 텍스트 위치 찾아서 사각형 그리기
            x1, x2 = word_locations
            margin = 1
            
            a = cv2.rectangle(org_image, (x - margin, y - margin), (x + w + margin, y + h + margin), (0, 0, 255), 2)

            # 처리된 결과를 웹 페이지에 전달
        return render(request, 'map/homeinfo.html', {'result': image})

    return render(request, 'map/homeinfo.html')

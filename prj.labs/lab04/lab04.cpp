#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include "../../utils/utils.cpp"


cv::Mat getBinary(cv::Mat& image) {
  cv::Mat imgCopy;

  cv::threshold(image, imgCopy, 170, 255, cv::THRESH_OTSU);

  return imgCopy;
}

cv::Mat getMask(cv::Mat& image) {
  cv::Mat imgCopy;

  int erosion_type = 0;
  int erosion_size = 1;
  cv::Mat element = getStructuringElement( erosion_type,
                                           cv::Size( 2 * erosion_size + 1, 2 * erosion_size + 1 ),
                                           cv::Point( -1, -1 ) );
  erode( image, imgCopy, element );
  erosion_size = 4;
  element = getStructuringElement( erosion_type,
                                   cv::Size( 2 * erosion_size + 1, 2 * erosion_size + 1 ),
                                   cv::Point( -1, -1 ) );
  dilate( imgCopy, imgCopy, element, cv::Point(-1, -1), 2);

  return imgCopy;
}

cv::Mat getLargestComponent(cv::Mat& img) {
  cv::Mat colored = getColoredComponents(img);
  cv::Mat components;
  cv::Mat stats;
  cv::Mat centroids;
  int nLabels = cv::connectedComponentsWithStats(img, components, stats, centroids);
  int maxSize = 0;
  int maxLabel = 0;
  for (int l = 1; l < nLabels; l++) {
    int size = stats.at<int>(l, cv::CC_STAT_AREA);
    if (size > maxSize) {
      maxLabel = l;
      maxSize = size;
    }
  }

  cv::Mat component(img.size(), CV_8UC1);
  for(int r = 0; r < component.rows; ++r){
    for(int c = 0; c < component.cols; ++c){
      int lbl = components.at<int>(r, c);
      if (lbl == maxLabel) {
        auto &pixel = component.at<uint8_t>(r, c);
        pixel = (uint8_t)255;
      }
    }
  }

  return component;
}

void colorizeMask(cv::Mat& src, cv::Scalar& color) {
  cv::Mat mask;
  cv::inRange(src, cv::Scalar(255, 255, 255), cv::Scalar(255, 255, 255), mask);
  src.setTo(color, mask);
}

cv::Mat getDiff(cv::Mat& standart, cv::Mat& mask, cv::Mat& grayscale) {
  cv::Mat dst(grayscale.size(), CV_8UC3);
  cv::cvtColor(grayscale, dst, cv::COLOR_GRAY2BGR);
  for(int r = 0; r < mask.rows; ++r){
    for(int c = 0; c < mask.cols; ++c){
      auto &stdPixel = standart.at<uint8_t>(r, c);
      auto &maskPixel = mask.at<uint8_t>(r, c);
      auto &dstPixel = dst.at<cv::Vec3b>(r, c);
      if (stdPixel == 0 && maskPixel == 255) {
        dstPixel = cv::Vec3b(0, 255, 0);
      }
      if (stdPixel == 255 && maskPixel == 0) {
        dstPixel = cv::Vec3b(0, 0, 255);
      }
      if (stdPixel == 255 && maskPixel == 255) {
        dstPixel = cv::Vec3b(255, 0, 0);
      }
    }
  }

  cv::Mat standardCopy(standart.size(), CV_8UC3);
  cv::cvtColor(standart, standardCopy, cv::COLOR_GRAY2BGR);

  cv::addWeighted( dst, 0.8, standardCopy, 0.4, 0.0, dst);
  return dst;
}

double getMaskRatio(cv::Mat& standart, cv::Mat& mask) {
  cv::Mat totalMask = standart + mask;
  int standartTotal = cv::countNonZero(standart + mask);
  double commonPixelsTotal = 0;
  for (int r = 0; r < standart.rows; ++r) {
    for (int c = 0; c < standart.cols; ++c) {
      auto stdPixel = standart.at<uint8_t>(r, c);
      auto mskPixel = mask.at<uint8_t>(r, c);
      if (stdPixel == mskPixel && mskPixel == cv::saturate_cast<uint8_t>(255)) {
        commonPixelsTotal += 1;
      }
    }
  }

  double total = commonPixelsTotal / standartTotal;
  return total;
}

int main() {
  const std::vector<const std::string> videosList({
    "../data/video/video_1.MOV",
    "../data/video/video_2.MOV",
    "../data/video/video_3.MOV",
    "../data/video/video_4.MOV",
    "../data/video/video_5.MOV",
    "../data/video/video_6.MOV",
    "../data/video/video_7.MOV"
  });
  std::vector<double> ratios;
  for (int v = 0; v < videosList.size(); v++) {
    std::string videoPath = videosList[v];
    auto video = cv::VideoCapture(videoPath);
    std::vector<cv::Mat> frames = getFrames(video);

    for (int f = 0; f < frames.size(); f++) {
      int videoIndex = v + 1;
      int frameIndex = f + 1;
      cv::Mat frame = frames[f];
      cv::imwrite("frame" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex) + ".png", frame);
      cv::Mat grayscale;

      cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);
      cv::imwrite("frame" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex) + "_gray.png", grayscale);

      cv::Mat binary = getBinary(grayscale);
      cv::imwrite("frame" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex) + "_binary.png", binary);

      cv::Mat mask = getMask(binary);
      cv::imwrite("frame" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex) + "_mask.png", mask);

      cv::Mat finalMask = getLargestComponent(mask);
      cv::imwrite("frame" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex) + "_mask_pre_final.png", finalMask);

      cv::Mat element = getStructuringElement( 0,
                                               cv::Size( 15, 15 ),
                                               cv::Point( -1, -1 ) );
      cv::morphologyEx(finalMask, finalMask, cv::MORPH_CLOSE, element);

      cv::imwrite("frame" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex) + "_mask_final.png", finalMask);
      std::string maskName = "mask" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex);
      cv::Mat standart = cv::imread("../data/standart.mask/" + maskName + "_ideal.jpg", CV_8UC1);

      cv::Mat diff = getDiff(standart, finalMask, grayscale);
      double ratio = getMaskRatio(standart, finalMask);
      ratios.push_back(ratio);

      cv::putText(diff, "K: " + toStringWithPrecision<double>(ratio, 3), cv::Point2f(30, 100), 0, 3, { 200, 200, 200 });
      cv::imwrite(maskName + "_diff" + ".png", diff);
    }
    video.release();
  }

  double sum = 0;
  for (double ratio : ratios) {
    sum += ratio;
  }
  double average = sum / (double)ratios.size();

  double sum2 = 0;
  for (double ratio : ratios) {
    sum2 += std::pow((ratio), 2);
  }
  double dispersion = sum2 - std::pow(sum / (double)(ratios.size()), 2);

  std::cout << "Average ratio: " << toStringWithPrecision(average, 3) << '\n';
  std::cout << "Average dispersion: " << toStringWithPrecision(dispersion, 3);

  return 0;
}
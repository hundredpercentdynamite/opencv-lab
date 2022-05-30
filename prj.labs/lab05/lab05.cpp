#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include "../../utils/utils.cpp"

cv::RNG rng(12345);

cv::Mat getBlur(cv::Mat& img) {
  cv::Mat result;
  cv::blur(img, result, cv::Size(3, 3));
  return result;
}


cv::Mat getBinary(cv::Mat& image) {
  cv::Mat imgCopy;

  cv::threshold(image, imgCopy, 170, 255, cv::THRESH_OTSU);

  return imgCopy;
}

cv::Mat morphologyPreparation(cv::Mat& image) {
  cv::Mat result;
  cv::Mat element = getStructuringElement( 0,
                                           cv::Size( 5, 5 ),
                                           cv::Point( -1, -1 ) );
  cv::morphologyEx(image, result, cv::MORPH_OPEN, element);
  cv::morphologyEx(result, result, cv::MORPH_CLOSE, element);

  return result;
}

cv::Mat getEdges(cv::Mat& gray) {
  cv::Mat result;
  cv::Mat blur;
  cv::blur(gray, blur, cv::Size(3, 3));
  cv::imshow("blur", blur);
  cv::threshold(blur, result, 170, 255, cv::THRESH_OTSU);
  cv::Mat element = getStructuringElement( 0,
                                           cv::Size( 5, 5 ),
                                           cv::Point( -1, -1 ) );
  cv::morphologyEx(result, result, cv::MORPH_OPEN, element);
  cv::morphologyEx(result, result, cv::MORPH_CLOSE, element);
  return result;
}

void getCountours(cv::Mat& edges, std::vector<std::vector<cv::Point>>& contours, std::vector<cv::Vec4i>& hierarchy) {
  cv::findContours( edges, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_L1 );
}

std::vector<std::vector<cv::Point>> getApproxContours(std::vector<std::vector<cv::Point>> contours) {
  std::vector<std::vector<cv::Point>> approxxs;
  for( size_t i = 0; i < contours.size(); i++ )
  {
    std::vector<cv::Point> currContour = contours[i];
    cv::Scalar color = cv::Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
    std::vector<cv::Point> approx;
    double epsilon = 0.1 * cv::arcLength(currContour, true);
    cv::approxPolyDP(currContour, approx, epsilon, true);
    approxxs.push_back(approx);
  }

  return approxxs;
}

std::vector<cv::Point> getLargestSquareContour(std::vector<std::vector<cv::Point>> contours) {
  std::vector<cv::Point> largestContour;
  double largestArea = 0;
  for( size_t i = 0; i < contours.size(); i++ )
  {
    cv::Scalar color = cv::Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
    std::vector<cv::Point> current = contours[i];
    if (cv::isContourConvex(current) && current.size() == 4) {
      double area = cv::contourArea(current);
      if (area > largestArea) {
        largestArea = area;
        largestContour = current;
      }
    }
  }

  return largestContour;
}

std::vector<cv::Point> getStandardContour(cv::FileStorage& json, std::string& filename) {
  std::vector<cv::Point> standardContour;
  for (int i = 0; i < 4; ++i) {
    int x = json[filename]["regions"][0]["shape_attributes"]["all_points_x"][i];
    int y = json[filename]["regions"][0]["shape_attributes"]["all_points_y"][i];
    cv::Point point(x, y);
    standardContour.push_back(point);
  }

  return standardContour;
}

void sortContour(std::vector<cv::Point>& contour) {
  cv::Point topLeft;
  cv::Point topRight;
  cv::Point bottomRight;
  cv::Point bottomLeft;
  std::vector<cv::Point> topPoints;
  std::vector<cv::Point> bottomPoints;
  std::sort(contour.begin(), contour.end(), [](cv::Point a, cv::Point b) {
    return a.y < b.y;
  });
  topPoints.push_back(contour[0]);
  topPoints.push_back(contour[1]);
  bottomPoints.push_back(contour[2]);
  bottomPoints.push_back(contour[3]);
  std::sort(topPoints.begin(), topPoints.end(), [](cv::Point a, cv::Point b) {
    return a.x < b.x;
  });
  std::sort(bottomPoints.begin(), bottomPoints.end(), [](cv::Point a, cv::Point b) {
    return a.x < b.x;
  });
  topLeft = topPoints[0];
  topRight = topPoints[1];
  bottomLeft = bottomPoints[0];
  bottomRight = bottomPoints[1];
  contour[0] = topLeft;
  contour[1] = topRight;
  contour[2] = bottomRight;
  contour[3] = bottomLeft;
}

double compareContours(std::vector<cv::Point> contour, std::vector<cv::Point> standard) {
  double standardArcLength = cv::arcLength(standard, true);
  double maxValue = 0;
  for (int i = 0; i < 4; ++i) {
    cv::Point contourPoint = contour[i];
    cv::Point standardContourPoint = standard[i];
    double value = (cv::norm(contourPoint - standardContourPoint)) / standardArcLength;
    if (value > maxValue) {
      maxValue = value;
    }
  }

  return maxValue;
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
  cv::FileStorage standardJson = cv::FileStorage("../data/contour_standard.json", cv::FileStorage::Mode::READ);
  for (int v = 0; v < videosList.size(); v++) {
    std::string videoPath = videosList[v];
    auto video = cv::VideoCapture(videoPath);
    std::vector<cv::Mat> frames = getFrames(video);

    for (int f = 0; f < frames.size(); f++) {
      int videoIndex = v + 1;
      int frameIndex = f + 1;
      cv::Mat frame = frames[f];
      std::string filename = "frame" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex) + ".png";
      cv::imwrite(filename, frame);
      cv::Mat grayscale;

      cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);
      cv::imwrite("frame" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex) + "_gray.png", grayscale);

      cv::Mat blurred = getBlur(grayscale);
      cv::imwrite("frame" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex) + "_blur.png", blurred);

      cv::Mat binary = getBinary(blurred);
      cv::imwrite("frame" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex) + "_binary.png", binary);

      cv::Mat edges = morphologyPreparation(binary);
      cv::imwrite("frame" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex) + "_edges.png", edges);

      std::vector<std::vector<cv::Point>> contours;
      std::vector<cv::Vec4i> hierarchy;
      getCountours(edges, contours, hierarchy);

      cv::Mat drawnContours = frame.clone();
      cv::Mat drawnLargestContour = frame.clone();

      std::vector<std::vector<cv::Point>> approxxs = getApproxContours(contours);
      for (int i = 0; i < approxxs.size(); i++) {
        cv::Scalar color = cv::Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
        cv::drawContours(drawnContours, approxxs, i, color, 2);
      }
      cv::imwrite("frame" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex) + "_all_contours.png", drawnContours);

      std::vector<cv::Point> largestApproxContour = getLargestSquareContour(approxxs);
      std::vector<std::vector<cv::Point>> lac {largestApproxContour};
      cv::drawContours(drawnLargestContour, lac, 0, cv::Scalar(100, 200, 100), 2);
      cv::imwrite("frame" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex) + "_largest_contour.png", drawnLargestContour);

      std::vector<cv::Point> standardContour = getStandardContour(standardJson, filename);
      sortContour(largestApproxContour);
      sortContour(standardContour);
      double quality = compareContours(largestApproxContour, standardContour);
      ratios.push_back(quality);
      cv::Mat frameCopy = frame.clone();
      std::vector<std::vector<cv::Point>> contoursResult = { largestApproxContour, standardContour };

      std::vector<cv::Scalar> colors { cv::Scalar(255, 100, 0), cv::Scalar(0, 255, 0) };
      for (int i = 0; i < contoursResult.size(); i++) {
        cv::drawContours(frameCopy, contoursResult, i, colors[i], 2);
      }
      cv::putText(frameCopy, "Error: " + toStringWithPrecision<double>(quality, 4), cv::Point2f(30, 100), 0, 3, { 255, 255, 255 } );

      cv::imwrite("frame" + std::to_string(videoIndex) + "_" + std::to_string(frameIndex) + "_diff.png", frameCopy);
    }
    video.release();
  }
  standardJson.release();
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

  std::cout << "Average error: " << toStringWithPrecision(average, 4) << '\n';
  std::cout << "Dispersion: " << toStringWithPrecision(dispersion, 4);

  return 0;
}
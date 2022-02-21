#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat getMosaic(cv::Mat& img) {
  int rows = img.rows;
  int cols = img.cols;

  cv::Mat mosaic(rows * 2, cols * 2, CV_8UC3);
  cv::Mat channels[3];
  cv::split(img, channels);
  cv::Mat zeros = cv::Mat::zeros(rows, cols, CV_8UC1);
  cv::Rect2d rc({0, 0, (double) cols, (double) rows});


  cv::Mat leftUpper = mosaic(rc);
  img.copyTo(leftUpper);

  rc.x = 256;
  cv::Mat rightUpper = mosaic(rc);
  std::vector<cv::Mat> blackRed({zeros, zeros, channels[2]});
  cv::merge(blackRed, rightUpper);

  rc.x = 0;
  rc.y = 256;
  cv::Mat leftLower = mosaic(rc);
  std::vector<cv::Mat> blackGreen({zeros, channels[1], zeros});
  cv::merge(blackGreen, leftLower);

  rc.x = 256;
  cv::Mat rightLower = mosaic(rc);
  std::vector<cv::Mat> blackBlue({channels[0], zeros, zeros});
  cv::merge(blackBlue, rightLower);

  return mosaic;
}

cv::Mat getHist(cv::Mat& img) {
  float range[] = { 0, 256 } ;
  const float* histRange = { range };
  cv::Mat channels[3];
  cv::split(img, channels);
  int histSize = 256;
  bool uniform = true;
  bool accumulate = false;

  cv::Mat b_hist, g_hist, r_hist;

  int hist_w = 512; int hist_h = 400;
  int bin_w = cvRound( (double) hist_w/histSize );

  calcHist( &channels[0], 1, 0, cv::Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &channels[1], 1, 0, cv::Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
  calcHist( &channels[2], 1, 0, cv::Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

  cv::Mat histImage( hist_h, hist_w, CV_8UC3, cv::Scalar( 255,255,255) );

  normalize(b_hist, b_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
  normalize(g_hist, g_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );
  normalize(r_hist, r_hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat() );

  for( int i = 1; i < histSize; i++ )
  {
    line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
          cv::Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
          cv::Scalar( 255, 0, 0), 2, 8, 0  );
    line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
          cv::Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
          cv::Scalar( 0, 255, 0), 2, 8, 0  );
    line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
          cv::Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
          cv::Scalar( 0, 0, 255), 2, 8, 0  );
  }

  return histImage;
}


int main() {
  // read png
  auto imgPath = cv::samples::findFile("../data/cross_0256x0256.png");
  cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);

  // write 25% jpeg
  std::vector<int> compression({cv::IMWRITE_JPEG_QUALITY, 25});
  cv::imwrite("cross_0256x0256_025.jpg", img, compression);

  // png-mosaic
  cv::Mat pngMosaic = getMosaic(img);
  cv::imwrite("cross_0256x0256_png_channels.png", pngMosaic);

  // jpg-mosaic
  auto jpgPath = cv::samples::findFile("cross_0256x0256_025.jpg");
  cv::Mat jpg = cv::imread(jpgPath, cv::IMREAD_COLOR);
  cv::Mat jpgMosaic = getMosaic(jpg);
  cv::imwrite("cross_0256x0256_jpg_channels.png", jpgMosaic);


  cv::Mat pngHist = getHist(img);
  cv::Mat jpgHist = getHist(jpg);

  cv::Mat resultHist(400, 1054, CV_8UC3);
  cv::Rect2d rc({ 0, 0, 512, 400 });
  pngHist.copyTo(resultHist(rc));
  rc.x = 532;
  jpgHist.copyTo(resultHist(rc));

  cv::imwrite("cross_0256x0256_hists.png", resultHist);
}

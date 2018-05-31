#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <iomanip>
#include <sstream>


using namespace cv;

void readme();
std::string ZeroPadNumber(int num);

/** @function main */
int main( int argc, char** argv )
{
  if( argc != 10 )
  { readme(); return -1; }

  //Images

  std::string pathKin, pathIDS, pathPola, pathNIR;
  std::string outKin, outIDS, outPola, outNIR;
  std::string mask;

  pathKin = argv[1];
  pathIDS = argv[2];
  pathPola = argv[3];
  pathNIR = argv[4];
  mask = argv[5];
  outKin = argv[6];
  outIDS = argv[7];
  outPola = argv[8];
  outNIR = argv[9];

  vector<float> corner1X, corner1Y, corner2X, corner2Y, corner3X, corner3Y, corner4X, corner4Y;
  vector<Point2f> corner1, corner2, corner3, corner4;

  vector<std::string> imagename;
  int n_image = 204;
  vector<Mat> vec; vec.resize(3);

  for(int i = 1 ; i < n_image + 1; i++){imagename.push_back("/image_"+ZeroPadNumber(i)+".tiff");}
  // Mat img_scene = imread( pathKin + imagename[203] , CV_LOAD_IMAGE_GRAYSCALE );
  // imshow("Hello",img_scene);
  // waitKey(0);
  // return 0;
  Mat img_scene;
  Mat img_mask = imread( mask, CV_LOAD_IMAGE_GRAYSCALE );
  Mat img_object_warped, img_object_diff;

  int indx = 5;

  vector<Mat> H;

  img_scene = imread( pathKin + imagename[indx] , CV_LOAD_IMAGE_GRAYSCALE );
  vec[0] = imread( pathIDS + imagename[indx] , CV_LOAD_IMAGE_GRAYSCALE );
  vec[1] = imread( pathPola + imagename[indx] , CV_LOAD_IMAGE_GRAYSCALE );
  vec[2] = imread( pathNIR + imagename[indx] , CV_LOAD_IMAGE_GRAYSCALE );

  if( !vec[0].data || !img_scene.data || !vec[1].data || !vec[2].data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }
  // int cnt = 0;
  for (std::vector<Mat>::iterator it = vec.begin() ; it != vec.end(); ++it){
      //-- Step 1: Detect the keypoints using SURF Detector
      int minHessian = 1000;

      SiftFeatureDetector detector( minHessian );

      std::vector<KeyPoint> keypoints_object, keypoints_scene;

      detector.detect( *it, keypoints_object );
      detector.detect( img_scene, keypoints_scene,img_mask );

      //-- Step 2: Calculate descriptors (feature vectors)
      SiftDescriptorExtractor extractor;

      Mat descriptors_object, descriptors_scene;

      extractor.compute( *it, keypoints_object, descriptors_object );
      extractor.compute( img_scene, keypoints_scene, descriptors_scene );

      //-- Step 3: Matching descriptor vectors using FLANN matcher
      FlannBasedMatcher matcher;
      std::vector< DMatch > matches;
      matcher.match( descriptors_object, descriptors_scene, matches );

      double max_dist = 0; double min_dist = 150;

      //-- Quick calculation of max and min distances between keypoints
      for( int i = 0; i < descriptors_object.rows; i++ )
      { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
      }

      //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
      std::vector< DMatch > good_matches;

      for( int i = 0; i < descriptors_object.rows; i++ )
      { if( matches[i].distance < 3*min_dist )
         { good_matches.push_back( matches[i]); }
      }

      Mat img_matches;
      drawMatches( *it, keypoints_object, img_scene, keypoints_scene,
                   good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                   vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

      //-- Localize the object
      std::vector<Point2f> obj;
      std::vector<Point2f> scene;

      for( int i = 0; i < good_matches.size(); i++ )
      {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
      }

      H.push_back(findHomography( obj, scene, CV_RANSAC, 4 ));

      //-- Get the corners from the image_1 ( the object to be "detected" )
      std::vector<Point2f> obj_corners(4);
      obj_corners[0] = cvPoint(0,0);
      obj_corners[1] = cvPoint( it->cols, 0 );
      obj_corners[2] = cvPoint( it->cols, it->rows );
      obj_corners[3] = cvPoint( 0, it->rows );
      std::vector<Point2f> scene_corners(4);

      perspectiveTransform( obj_corners, scene_corners, H.back());

      warpPerspective(*it,img_object_warped,H.back(),img_scene.size());

      absdiff(img_scene, img_object_warped, img_object_diff);

      corner1.push_back(scene_corners[0]);
      corner2.push_back(scene_corners[1]);
      corner3.push_back(scene_corners[2]);
      corner4.push_back(scene_corners[3]);


      corner1X.push_back(corner1.back().x);
      corner1Y.push_back(corner1.back().y);

      corner2X.push_back(corner2.back().x);
      corner2Y.push_back(corner2.back().y);

      corner3X.push_back(corner3.back().x);
      corner3Y.push_back(corner3.back().y);

      corner4X.push_back(corner4.back().x);
      corner4Y.push_back(corner4.back().y);


  }

  float maxX14 = max(*max_element(corner1X.begin(), corner1X.end()), *max_element(corner4X.begin(), corner4X.end()));
  float maxY12 = max(*max_element(corner1Y.begin(), corner1Y.end()), *max_element(corner2Y.begin(), corner2Y.end()));
  float minX23 = min(*min_element(corner2X.begin(), corner2X.end()), *min_element(corner3X.begin(), corner3X.end()));
  float minY34 = min(*min_element(corner3Y.begin(), corner3Y.end()), *min_element(corner4Y.begin(), corner4Y.end()));

  Point2f corner1fin(maxX14, maxY12);
  Point2f corner2fin(minX23, maxY12);
  Point2f corner3fin(minX23, minY34);
  Point2f corner4fin(maxX14, minY34);

  std::cout << corner1fin << std::endl;
  std::cout << corner2fin << std::endl;
  std::cout << corner3fin << std::endl;
  std::cout << corner4fin << std::endl;


  cv::Rect roi(corner1fin, corner3fin);
  cv::Mat tmp;
  std::string temporary;

  for(int n = 1; n < n_image; n++){
      //td::cout << std::to_string(n) << std::endl;
      // if(n == 30 || n == 49 || n == 50) continue;
      img_scene = imread( pathKin + imagename[n] , CV_LOAD_IMAGE_GRAYSCALE );
      vec[0] = imread( pathIDS + imagename[n] , CV_LOAD_IMAGE_GRAYSCALE );
      vec[1] = imread( pathPola + imagename[n] , CV_LOAD_IMAGE_GRAYSCALE );
      vec[2] = imread( pathNIR + imagename[n] , CV_LOAD_IMAGE_GRAYSCALE );

      img_scene(roi).copyTo(tmp);
      imwrite(outKin + imagename[n] , tmp);

      if( !vec[0].data || !img_scene.data || !vec[1].data || !vec[2].data )
      { std::cout<< " --(!) Error reading images " << std::endl; return -1; }
      int cnt = 0;
          for (std::vector<Mat>::iterator it = vec.begin() ; it != vec.end(); ++it){

              //-- Get the corners from the image_1 ( the object to be "detected" )
              std::vector<Point2f> obj_corners(4);
              obj_corners[0] = cvPoint(0,0);
              obj_corners[1] = cvPoint( it->cols, 0 );
              obj_corners[2] = cvPoint( it->cols, it->rows );
              obj_corners[3] = cvPoint( 0, it->rows );
              std::vector<Point2f> scene_corners(4);

              perspectiveTransform( obj_corners, scene_corners, H[cnt]);

              warpPerspective(*it,img_object_warped,H[cnt],img_scene.size());

              absdiff(img_scene, img_object_warped, img_object_diff);

              if(cnt == 0){
                  temporary = outIDS;
              }else if(cnt == 1){
                  temporary = outPola;
              }else if(cnt == 2){
                  temporary = outNIR;
              }

              img_object_warped(roi).copyTo(tmp);
              imwrite(temporary + imagename[n] , tmp);
              cnt++;

            }
    }

  }

  /** @function readme */
  void readme()
  { std::cout << " Incorrect amount of arguments" << std::endl; }


  std::string ZeroPadNumber(int num){
    std::ostringstream ss;
    ss << std::setw( 5 ) << std::setfill( '0' ) << num;
    return ss.str();
    }

//
// Created by lishuai on 23-6-20.
//

#ifndef IMAGEFUSION_IMAGEFUSION_H
#define IMAGEFUSION_IMAGEFUSION_H
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/version.hpp"
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include<opencv2/xfeatures2d/nonfree.hpp>//sift
#include <opencv2/imgproc/types_c.h>
//#include "opencv2/"


using namespace std;
using namespace cv;
using namespace xfeatures2d;


//左目相机标定参数------------------------
Mat cameraMatrixL = (Mat_<double>(3, 3) <<9.009037138922226e+02, 0, 1.550693327796179e+02,
        0, 9.013382314086579e+02, 1.917709156647431e+02,
        0, 0, 1);

Mat distCoeffL = (Mat_<double>(5, 1) << 0.021037929851499,0.949227651407186, 0, 0, 0);
//[k1,  k2,  p1,  p2,  k3]



//右目相机标定参数------------------------
Mat cameraMatrixR = (Mat_<double>(3, 3) << 8.947898819848670e+02, 0, 1.538706580678034e+02,
        0, 8.949306273118307e+02, 1.785279821510671e+02,
        0, 0, 1);

Mat distCoeffR = (Mat_<double>(5, 1) << 0.067768586022539, -0.095559145472375, 0, 0, 0);

Mat T = (Mat_<double>(3, 1) << -61.611390774404550,
        0.032196635244676,
        -1.097594798574489);                 //T平移向量

//Mat R;
Mat R = (Mat_<double>(3, 3) << 0.999814318148340, -6.393716306569230e-04, 0.019259294625899,
        6.317071005046178e-04,0.999999718846320, 4.040463104126284e-04,
        -0.019259547546826, -3.918050531796604e-04, 0.999814440942464);          //R矩阵，用于中间计算

//--左相机和热红外相机标定参数-stereoParams_left_thermal.mat------------------------------------------------------
Mat VIS_cameraMatrix_L = (Mat_<double>(3, 3) << 8.944967564974489e+02, 0, 1.519480461972283e+02,
        0, 8.949330105331678e+02, 1.942776397416282e+02,
        0, 0, 1);

Mat VIS_distCoeffs_L = (Mat_<double>(5, 1) << 0.022712952103076, 0.670867062708879, 0, 0, 0);

Mat TIR_cameraMatrix = (Mat_<double>(3, 3) << 8.948418595300168e+02 , 0, 1.911681780133693e+02,
        0, 8.951354061242225e+02, 1.530577208387779e+02,
        0, 0, 1);

Mat TIR_distCoeffs = (Mat_<double>(5, 1) << -0.029241786636051, 1.055064913836022, 0, 0, 0);

Mat VIS_TIR_T = (Mat_<double>(3, 1) << -31.581720666393240,
        54.975137423063025,
        0.888415891465847);

Mat VIS_TIR_R = (Mat_<double>(3, 3) << 0.999927296760173, -0.006782256643534 , -0.010117229384401,
        0.006590631049641, 0.999790607936295 , -0.019375032830189,
        0.010097760952781 , 0.019061808188127, 0.999767314274846);
Mat VIS_TIR_recv; //VIS_TIR_R的旋转向量


class ImageFusion
{
public:
    ImageFusion()
    {

    }
    //surf特征点检测
   void OpenSurf(Mat &ImageL,Mat &ImageR,vector<KeyPoint> &KeyPointL,vector<KeyPoint> &KeyPointR,\
                vector<DMatch> &good_matches,vector<Point2f> &good_keypoints_L,vector<Point2f> &good_keypoints_R)
    {

        //surf 特征点检测
        Ptr<SURF> surf = SURF::create();
        surf->detect(ImageL,KeyPointL);
        surf->detect(ImageR,KeyPointR);
        Mat KeyPointImageL,KeyPointImageR;

        KeyPointsFilter f;
        f.runByKeypointSize(KeyPointL, 0, 60);//特征点的半径设置的小于60像素
        f.runByKeypointSize(KeyPointR, 0, 60);
        //cout << "keyPointL_Number:" << keyPointL.size() << endl; cout << "keyPointR_Number:" << keyPointR.size() << endl;
        f.runByImageBorder(KeyPointL, ImageL.size(), 55);//过滤边界半径55以内的点
        f.runByImageBorder(KeyPointR, ImageR.size(), 55);//过滤边界的点
        f.removeDuplicated(KeyPointL);//删除重复的点
        f.removeDuplicated(KeyPointR);//删除重复的点

        drawKeypoints(ImageL, KeyPointL, KeyPointImageL, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        drawKeypoints(ImageR, KeyPointR, KeyPointImageR, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        //特征点匹配
        cv::Mat despL, despR;
        surf->compute(ImageL, KeyPointL, despL);
        surf->compute(ImageR, KeyPointR, despR);

        //定义匹配结果变量
        std::vector<cv::DMatch> matches;

        //如果采用 flannBased 方法 那么 desp通过orb的到的类型不同需要先转换类型
        if (despL.type() != CV_32F || despR.type() != CV_32F)
        {
            despL.convertTo(despL, CV_32F);
            despR.convertTo(despR, CV_32F);
        }
        cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
        matcher->match(despL, despR, matches);
        Mat image_match1;

        //极线几何约束--------------------------------------------------------------------------------
        cout << "原匹配点数为：" << matches.size() << endl;
        int count = 0;
        for (int i = 0; i < matches.size(); i++)
        {
            Point2f pt1, pt2;
            pt1 = KeyPointL[matches[i].queryIdx].pt;
            pt2 = KeyPointR[matches[i].trainIdx].pt;
            if (abs(pt1.y - pt2.y) < 1)
            {
                good_matches.push_back(matches[i]);
                good_keypoints_L.push_back(KeyPointL[matches[i].queryIdx].pt); //good_keypoints_L是Point2f点
                good_keypoints_R.push_back(KeyPointR[matches[i].trainIdx].pt);
                count++;
            }

        }
        cout << "极线几何约束筛选的匹配点对数为：" << good_matches.size() << endl;
        //在左右图像上画出特征点
        for (int i = 0; i < count; i++)
        {
        	circle(ImageL, cvPoint(good_keypoints_L[i].x, good_keypoints_L[i].y),
        		3, cvScalar(255, 0, 0), -1);
        	circle(ImageR, cvPoint(good_keypoints_R[i].x, good_keypoints_R[i].y),
        		3, cvScalar(255, 0, 0), -1);
        }
        Mat ImageOutput;
        cv::drawMatches(ImageL, KeyPointL, ImageR, KeyPointR, good_matches, ImageOutput);
        cv::namedWindow("极线几何约束精匹配后的左&右图片-原图");	cv::imshow("极线几何约束精匹配后的左&右图片-原图", ImageOutput);
    }
    //--将像素坐标转换至相机平面坐标---------------------------------------------------------------------
    Point2f pixel2cam(const Point2d& p, const Mat& K)
    {
        return Point2f
                (
                        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
                );
    }
    //三角法测距
    void triangulation(vector<KeyPoint> &KeyPointL,vector<KeyPoint> &KeyPointR,vector<DMatch>&matches,Mat & R, Mat& t,vector<Point3f>& points)
    {
        //相机第一个位置处的位姿
        Mat T1 = (Mat_<float>(3, 4) <<
                                    1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0);
        //相机第二个位置处的位姿
        Mat T2 = (Mat_<float>(3, 4) <<
                                    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
        );

        vector<Point2f> pts_1;
        vector<Point2f> pts_2;

        for (DMatch m : matches)
        {
            // 将像素坐标转换至相机平面坐标，为什么要这一步，上面推导中有讲
            pts_1.push_back(pixel2cam(KeyPointL[m.queryIdx].pt, cameraMatrixL));
            pts_2.push_back(pixel2cam(KeyPointR[m.trainIdx].pt, cameraMatrixR));
        }

        Mat pts_4d;
        //opencv提供的三角测量函数
        cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

        // 转换成非齐次坐标
        for (int i = 0; i < pts_4d.cols; i++)
        {
            Mat XYZ = pts_4d.col(i);
            XYZ /= XYZ.at<float>(3, 0);

            //世界坐标系中坐标
            Point3f world;
            world.x = XYZ.at<float>(0, 0);
            world.y = XYZ.at<float>(1, 0);
            world.z = XYZ.at<float>(2, 0);
            cout << "WorldPoint" << i << '(' << world.x << ',' << world.y << ',' << world.z << ')' << endl;
            points.push_back(world);
        }
    }

    //世界坐标投影到热红外坐标系
    vector<Point3f>World_Pixel_Project(vector< Point3f > &worldcoordinates)
    {
        //--公示法求世界坐标投影到图像像素坐标-------------------------------------------------------------
        Mat T3 = (Mat_<float>(4, 4) <<
                                    VIS_TIR_R.at<double>(0, 0), VIS_TIR_R.at<double>(0, 1), VIS_TIR_R.at<double>(0, 2), VIS_TIR_T.at<double>(0, 0),
                VIS_TIR_R.at<double>(1, 0), VIS_TIR_R.at<double>(1, 1), VIS_TIR_R.at<double>(1, 2), VIS_TIR_T.at<double>(1, 0),
                VIS_TIR_R.at<double>(2, 0), VIS_TIR_R.at<double>(2, 1), VIS_TIR_R.at<double>(2, 2), VIS_TIR_T.at<double>(2, 0),
                0, 0, 0, 1
        );

        //将热相机内参矩阵转换为齐次矩阵
        Mat TIR_Matrix_Homogeneous = (Mat_<float>(3, 4) <<
                                                        TIR_cameraMatrix.at<double>(0, 0), TIR_cameraMatrix.at<double>(0, 1), TIR_cameraMatrix.at<double>(0, 2), 0,
                TIR_cameraMatrix.at<double>(1, 0), TIR_cameraMatrix.at<double>(1, 1), TIR_cameraMatrix.at<double>(1, 2), 0,
                TIR_cameraMatrix.at<double>(2, 0), TIR_cameraMatrix.at<double>(2, 1), TIR_cameraMatrix.at<double>(2, 2), 0
        );

        vector< Point3f >imagePoints3f(worldcoordinates.size());//(worldcoordinates.size())
        for (size_t i = 0; i < imagePoints3f.size(); i++)
        {
            imagePoints3f[i] = Point3f((float)(0), (float)(0), (float)(1));
            //cout << imagePoints3f[i] << endl;
        }
        //将Point3f的坐标转换成Mat型
        Mat imagePoints3f_matrix(imagePoints3f);
        //将Mat类型的矩阵reshape成指定通道和大小的Mat
        Mat imagePoints3f_matrix2 = imagePoints3f_matrix.reshape(1, worldcoordinates.size());//reshape(channels,row)


        //提取出世界坐标的z分量
        vector<float>worldcoordinates_z;
        for (size_t i = 0; i < worldcoordinates.size(); i++)
        {
            //cout << worldcoordinates[i].z << endl;
            worldcoordinates_z.push_back(worldcoordinates[i].z);
        }


        //--将世界坐标转换为齐次坐标，它是N行1列的(Point3f->Mat->----------------------------------------------------
        Mat worldcoordinates_matrix(worldcoordinates);//将Point3f的坐标转换成Mat型


        Mat worldcoordinates_matrix2 = worldcoordinates_matrix.reshape(1, worldcoordinates.size());//将Mat型reshape成单通道，指定大小


        Mat w_c_Homogeneous(worldcoordinates.size(), 4, CV_32FC1);//定义一个新Mat用作齐次坐标

        convertPointsToHomogeneous(worldcoordinates_matrix2, w_c_Homogeneous); //convertPointsToHomogeneous
        Mat w_c_Homogeneous2 = w_c_Homogeneous.reshape(1, worldcoordinates.size());


        std::vector<Point3f> imagePointsmatrix_Point3f;
        Point3f tmpPoint3f;

        for (size_t i = 0; i < worldcoordinates.size(); i++)
        {
            Mat imagePoints3f_matrix3 = TIR_Matrix_Homogeneous * T3 * w_c_Homogeneous2.row(i).t() / worldcoordinates[i].z;//worldcoordinates_z[i]
            //cout << "World Points Coordinate in TIR" << i << imagePoints3f_matrix3 << endl;
            //cout << "TIR_Matrix_Homogeneous矩阵和T3矩阵相乘" << TIR_Matrix_Homogeneous * T3 * w_c_Homogeneous2.row(i).t() / worldcoordinates[i].z << endl;
            tmpPoint3f.x = imagePoints3f_matrix3.at<float>(0, 0)+35 ;// +4 3
            tmpPoint3f.y = imagePoints3f_matrix3.at<float>(1, 0) ;// +25  34
            //tmpPoint3f.z = imagePoints3f_matrix3.at<float>(2, 0);
            tmpPoint3f.z = 1;
            imagePointsmatrix_Point3f.push_back(tmpPoint3f);
        }

        return imagePointsmatrix_Point3f;
    }
    //--给感兴趣的若干对点连线----------------------------------------------------------------------------
    inline Mat DrawInlier(Mat& src1, Mat& src2, vector<Point2f>& kpt1, vector<Point2f>& kpt2, int& inlier_size, int type)
    {
        const int height = max(src1.rows, src2.rows);
        const int width = src1.cols + src2.cols;
        Mat output(height, width, CV_8UC3, Scalar(0, 0, 0));
        src1.copyTo(output(Rect(0, 0, src1.cols, src1.rows)));
        src2.copyTo(output(Rect(src1.cols, 0, src2.cols, src2.rows)));

        if (type == 1)
        {
            for (size_t i = 0; i < inlier_size; i++)//inlier_size
            {
                Point2f left = kpt1[i];
                Point2f right = (kpt2[i] + Point2f((float)src1.cols, 0.f));
                //line(output, left, right, Scalar(0, 255, 0),1,CV_AA);
                //circle(output, left, 3, cvScalar(255, 0, 0), -1);
                //circle(output, right, 3, cvScalar(0, 0, 255), -1);
            }
        }
        else if (type == 2)
        {
            for (size_t i = 0; i < inlier_size; i++)
            {
                Point2f left = kpt1[i];
                Point2f right = (kpt2[i] + Point2f((float)src1.cols, 0.f));
                line(output, left, right, Scalar(0, 255, 0), 1,  cv::LINE_AA);
            }

            for (size_t i = 0; i < inlier_size; i++)
            {
                Point2f left = kpt1[i];
                Point2f right = (kpt2[i] + Point2f((float)src1.cols, 0.f));
                //circle(output, left, 3, cvScalar(255, 0, 0), -1);
                //circle(output, right, 3, cvScalar(0, 0, 255), -1);
            }
        }
        return output;
    }

//SSIM检测相似性，遍历像素点
    double ssimDetect(cv::Mat& image_ref, cv::Mat& image_obj)
    {
        double C1 = 6.5025, C2 = 58.5225;
        //cv::Mat image_ref = cv::imread(imgOrg, CV_LOAD_IMAGE_GRAYSCALE);
        //cv::Mat image_obj = cv::imread(imgComp, CV_LOAD_IMAGE_GRAYSCALE);
        int width = image_ref.cols;
        int height = image_ref.rows;
        int width2 = image_obj.cols;
        int height2 = image_obj.rows;
        double mean_x = 0;//图像x均值
        double mean_y = 0;//图像y均值
        double sigma_x = 0;//图像x标准差
        double sigma_y = 0;//图像y标准差
        double sigma_xy = 0;//图像x和图像y协方差
        for (int v = 0; v < height; v++)
        {
            for (int u = 0; u < width; u++)
            {
                mean_x += image_ref.at<uchar>(v, u);
                mean_y += image_obj.at<uchar>(v, u);

            }
        }
        mean_x = mean_x / width / height;
        mean_y = mean_y / width / height;
        for (int v = 0; v < height; v++)
        {
            for (int u = 0; u < width; u++)
            {
                sigma_x += (image_ref.at<uchar>(v, u) - mean_x) * (image_ref.at<uchar>(v, u) - mean_x);
                sigma_y += (image_obj.at<uchar>(v, u) - mean_y) * (image_obj.at<uchar>(v, u) - mean_y);
                sigma_xy += abs((image_ref.at<uchar>(v, u) - mean_x) * (image_obj.at<uchar>(v, u) - mean_y));
            }
        }
        sigma_x = sigma_x / (width * height - 1);
        sigma_y = sigma_y / (width * height - 1);
        sigma_xy = sigma_xy / (width * height - 1);
        double fenzi = (2 * mean_x * mean_y + C1) * (2 * sigma_xy + C2);
        double fenmu = (mean_x * mean_x + mean_y * mean_y + C1) * (sigma_x + sigma_y + C2);
        double ssim = fenzi / fenmu;
        return ssim;
    }

//一般采用高斯函数计算图像的均值、方差以及协方差，而不是采用遍历像素点的方式，以换来更高的效率
    Scalar getMSSIM(Mat  inputimage1, Mat inputimage2)
    {
        Mat i1 = inputimage1;
        Mat i2 = inputimage2;
        const double C1 = 6.5025, C2 = 58.5225;
        int d = CV_32F;
        Mat I1, I2;
        i1.convertTo(I1, d);
        i2.convertTo(I2, d);
        Mat I2_2 = I2.mul(I2);
        Mat I1_2 = I1.mul(I1);
        Mat I1_I2 = I1.mul(I2);
        Mat mu1, mu2;
        GaussianBlur(I1, mu1, Size(11, 11), 1.5);
        GaussianBlur(I2, mu2, Size(11, 11), 1.5);
        Mat mu1_2 = mu1.mul(mu1);
        Mat mu2_2 = mu2.mul(mu2);
        Mat mu1_mu2 = mu1.mul(mu2);
        Mat sigma1_2, sigma2_2, sigma12;
        GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
        sigma1_2 -= mu1_2;
        GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
        sigma2_2 -= mu2_2;
        GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
        sigma12 -= mu1_mu2;
        Mat t1, t2, t3;
        t1 = 2 * mu1_mu2 + C1;
        t2 = 2 * sigma12 + C2;
        t3 = t1.mul(t2);                 // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
        t1 = mu1_2 + mu2_2 + C1;
        t2 = sigma1_2 + sigma2_2 + C2;
        t1 = t1.mul(t2);                 // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
        Mat ssim_map;
        divide(t3, t1, ssim_map);        // ssim_map =  t3./t1;
        Scalar mssim = mean(ssim_map);   // mssim = average of ssim map
        return mssim;
    }

    //求取单应性变换矩阵
    void FindHomographyRANSAC(Mat &ImageL,Mat &thermalImage,vector<Point2f> &good_keypoints_L,vector<Point2f> &imagePoints,vector<DMatch>& good_matches\
    )
    {

        std::vector<Point2f> good_good_keypoints_L, good_good_keypoints_TIR;  //挑选出left&thermal图像中好的特征点

        //--单应性热图像与左图像配准-----------------------------------------------------------
        vector<uchar>inliersMask2(good_keypoints_L.size());
        Mat H = cv::findHomography(imagePoints, good_keypoints_L, inliersMask2, FM_RANSAC, 2000);//.0

        cout << "单应性矩阵H=" << H << endl;

        cout << "原匹配点数为：" << good_matches.size() << endl;
        vector<DMatch>inliers;
        int count = 0;
        for (size_t i = 0; i < inliersMask2.size(); i++)
        {
            if (inliersMask2[i])
            {
                count += 1;
                inliers.push_back(good_matches[i]);
                good_good_keypoints_L.push_back(good_keypoints_L[i]); //good_keypoints_L是Point2f点
                good_good_keypoints_TIR.push_back(imagePoints[i]);
            }
        }

        //good_matches.swap(inliers);
        cout << "内点数为：" << inliers.size() << endl;

        //--给左图像和热图像中的点对连线------------------------------------------------------
        int g_m_size1 = good_matches.size();
//        Mat image_match2 = DrawInlier(ImageL, thermalImage, good_keypoints_L, imagePoints, g_m_size1, 1);
        Mat image_match2 = ImageL.clone();
        imshow("before F_Homo&RANSAC", image_match2);

        //--单应性热图像与左图像配准-----------------------------------------------------------
        Size size2 = ImageL.size();
        Mat dstimg2 = Mat::zeros(size2, CV_8UC3);
        Mat thermalImage1;
        thermalImage1= thermalImage.clone();
        //thermalImage.copyTo(thermalImage1);
        warpPerspective(thermalImage1, dstimg2, H, size2);
        cout << "单应性矩阵H=" << H << endl;
        imshow("left-thermal homography", dstimg2);
        imwrite("H_thermalImage.jpg", dstimg2);
        Mat Registration2_2 = ImageL * 0.4 + dstimg2 * 0.6;
        imshow("left-thermal fusion", Registration2_2);
        imwrite("Registration_result.jpg", Registration2_2);

        //计算单应变换后反投影点的坐标
        std::vector<cv::Point2f> imagePoints_pro;
        cv::perspectiveTransform(good_good_keypoints_TIR, imagePoints_pro, H);// imagePoints

        //计算单应变换误差
        double mse = 0.0;
        for (int i = 0; i < imagePoints_pro.size(); i++)
        {
            mse += (imagePoints_pro[i].x - good_good_keypoints_L[i].x) * (imagePoints_pro[i].y - good_good_keypoints_L[i].y);
        }
        mse = mse / imagePoints_pro.size();
        cout << "单应变换误差mse为:" << mse << endl;


        //---------------------------------------------------------------------------------------------------------------

        //--给单应性变换后的热图像和左图像中的点对连线-----------------------------------------
        int g_m_size2_2 = inliers.size();
//        Mat image_match3_2 = DrawInlier(ImageL, dstimg2, good_good_keypoints_L, imagePoints_pro, g_m_size2_2,1);
        Mat image_match3_2 = ImageL.clone();
        imshow("after F_Homo&RANSAC", image_match3_2);

        double ssim = 0.0;
        ssim = ssimDetect(ImageL, dstimg2);//dstimg1为单应变换后的左图像，thermalImage为热图像
        cout << "ssim:" << ssim << endl;

        Scalar SSIM1 = 0.0;
        SSIM1 = getMSSIM(ImageL, dstimg2);
        cout << "SSIM1:" << (SSIM1.val[2] + SSIM1.val[1] + SSIM1.val[0]) / 3 * 100 << endl;
    }
    void FindWarpTrans(vector<Point2f> &keypoints, vector<Point2f> &keypoints_out)
    {
        //定义平移尺度
        double tx = -15.0;   //水平平移尺度
        double ty = 55;    //垂直平移尺度

        for (const auto& point : keypoints) {
            float x = point.x;
            float y = point.y;
            float new_x = x + tx;
            float new_y = y + ty;
            keypoints_out.push_back(cv::Point2f(new_x, new_y));
        }


    }
private:


};

#endif //IMAGEFUSION_IMAGEFUSION_H

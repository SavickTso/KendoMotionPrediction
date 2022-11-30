#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <stdio.h>
#include <ur_rtde/rtde_control_interface.h>
#include <ur_rtde/rtde_io_interface.h>
#include <ur_rtde/rtde_receive_interface.h>
#include <thread>
#include <time.h>
#include "ximea.h"
#include "display_info.h"
#include <xiApiPlusOcv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <filesystem>
#include <cstdio>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include <opencv2/opencv.hpp>
#define OPENPOSE_FLAGS_DISABLE_PRODUCER
#define OPENPOSE_FLAGS_DISABLE_DISPLAY
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>
#include <fstream>

#define EXPECTED_IMAGES 10000
std::vector<double> KamaeState{ 0, -2.21, -0.87, -0.17, 0, 0.44 };
std::vector<double> MenState{ -0.24, -1.95, -0.18, 0.10, 1.43, 0.21 };
std::vector<double> KoteState{ -0.16, -2.23, -0.72, 0.98, 1.45, 0.16 };
std::vector<double> DouState{ -0.04, -2.27, -1.12, 0.58, 0.44, 1.13 };
const std::string saveRootDir = "E:/imgs0118left";
const std::string saveDir = saveRootDir + "/";
const std::string saveRootDirr = "E:/imgs0118right";
const std::string saveDirr = saveRootDirr + "/";

using namespace cv;
using namespace std;

std::vector<cv::Point2d> printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        std::vector<cv::Point2d> Jointlist(50);
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            op::opLog("\nKeypoints:", op::Priority::High);
            const auto& poseKeypoints = datumsPtr->at(0)->poseKeypoints;
            op::opLog("Person pose keypoints:", op::Priority::High);
            std::ofstream outFile;
            outFile.open("E:\Test.txt", ios::binary | ios::app | ios::in | ios::out);
            //for (auto person = 0; person < poseKeypoints.getSize(0); person++)
            for (auto person = 0; person < 2; person++)//here to add another person
            {
                op::opLog("Person " + std::to_string(person) + " (x, y, score):", op::Priority::High);
                // outFile<<"Person:"<<std::to_string(person)<<"\n";
                //cv::Point2d pointa(0., 0.), pointb(0., 0.);
                for (auto bodyPart = 0; bodyPart < 25; bodyPart++)//poseKeypoints.getSize(1)
                {
                    op::opLog("body part " + std::to_string(bodyPart), op::Priority::High);
                    //outFile<<"body part:"<<std::to_string(bodyPart)<<"\n";
                    //Jointlist.push_back(pointa);
                    std::string valueToPrint;
                    for (auto xyscore = 0; xyscore < poseKeypoints.getSize(2); xyscore++) {
                        valueToPrint += std::to_string(poseKeypoints[{person, bodyPart, xyscore}]) + " ";

                        switch (xyscore) {
                        case 0:
                            Jointlist[bodyPart + person * 25].x = poseKeypoints[{person, bodyPart, xyscore}];
                            break;
                        case 1:
                            Jointlist[bodyPart + person * 25].y = poseKeypoints[{person, bodyPart, xyscore}];
                            break;
                        }
                    }
                    op::opLog(valueToPrint, op::Priority::High);
                    outFile << valueToPrint << "  " << "\n";
                }
            }

            if (Jointlist[5].x > Jointlist[30].x && Jointlist[6].x > Jointlist[31].x && Jointlist[7].x > Jointlist[32].x) {

                auto start = Jointlist.begin();
                auto end = Jointlist.begin() + 24 + 1;
                // To store the sliced vector
                vector<cv::Point2d> resultfront(24 - 0 + 1);
                copy(start, end, resultfront.begin());

                auto starte = Jointlist.begin() + 25;
                auto ende = Jointlist.begin() + 49 + 1;
                // To store the sliced vector
                vector<cv::Point2d> resultend(49 - 25 + 1);
                copy(starte, ende, resultend.begin());
                resultend.insert(resultend.end(), resultfront.begin(), resultfront.end());
                std::cout << "resultend" << resultend << "resultsize   " << resultend.size() << std::endl;
                Jointlist = resultend;
            }
        }
        else
            op::opLog("Nullptr or empty datumsPtr found.", op::Priority::High);

        return Jointlist;
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}


void configureWrapper(op::Wrapper& opWrapper)
{
    try
    {
        // Configuring OpenPose
        op::checkBool(
            0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
            __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        // Applying user defined configuration - GFlags to program variables
        // outputSize
        const auto outputSize = op::flagsToPoint(op::String(FLAGS_output_resolution), "-1x-1");
        // netInputSize
        const auto netInputSize = op::flagsToPoint(op::String(FLAGS_net_resolution), "-1x368");
        // faceNetInputSize
        const auto faceNetInputSize = op::flagsToPoint(op::String(FLAGS_face_net_resolution), "368x368 (multiples of 16)");
        // handNetInputSize
        const auto handNetInputSize = op::flagsToPoint(op::String(FLAGS_hand_net_resolution), "368x368 (multiples of 16)");
        // poseMode
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        // poseModel
        const auto poseModel = op::flagsToPoseModel(op::String(FLAGS_model_pose));
        // JSON saving
        if (!FLAGS_write_keypoint.empty())
            op::opLog(
                "Flag `write_keypoint` is deprecated and will eventually be removed. Please, use `write_json`"
                " instead.", op::Priority::Max);
        // keypointScaleMode
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        // heatmaps to add
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg,
            FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        // >1 camera view?
        const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
        // Face and hand detectors
        const auto faceDetector = op::flagsToDetector(FLAGS_face_detector);
        const auto handDetector = op::flagsToDetector(FLAGS_hand_detector);
        // Enabling Google Logging
        const bool enableGoogleLogging = true;

        // Pose configuration (use WrapperStructPose{} for default and recommended configuration)
        const op::WrapperStructPose wrapperStructPose{
            poseMode, netInputSize, FLAGS_net_resolution_dynamic, outputSize, keypointScaleMode, FLAGS_num_gpu,
            FLAGS_num_gpu_start, FLAGS_scale_number, (float)FLAGS_scale_gap,
            op::flagsToRenderMode(FLAGS_render_pose, multipleView), poseModel, !FLAGS_disable_blending,
            (float)FLAGS_alpha_pose, (float)FLAGS_alpha_heatmap, FLAGS_part_to_show, op::String(FLAGS_model_folder),
            heatMapTypes, heatMapScaleMode, FLAGS_part_candidates, (float)FLAGS_render_threshold,
            FLAGS_number_people_max, FLAGS_maximize_positives, FLAGS_fps_max, op::String(FLAGS_prototxt_path),
            op::String(FLAGS_caffemodel_path), (float)FLAGS_upsampling_ratio, enableGoogleLogging };
        opWrapper.configure(wrapperStructPose);
        // Face configuration (use op::WrapperStructFace{} to disable it)
        const op::WrapperStructFace wrapperStructFace{
            FLAGS_face, faceDetector, faceNetInputSize,
            op::flagsToRenderMode(FLAGS_face_render, multipleView, FLAGS_render_pose),
            (float)FLAGS_face_alpha_pose, (float)FLAGS_face_alpha_heatmap, (float)FLAGS_face_render_threshold };
        opWrapper.configure(wrapperStructFace);
        // Hand configuration (use op::WrapperStructHand{} to disable it)
        const op::WrapperStructHand wrapperStructHand{
            FLAGS_hand, handDetector, handNetInputSize, FLAGS_hand_scale_number, (float)FLAGS_hand_scale_range,
            op::flagsToRenderMode(FLAGS_hand_render, multipleView, FLAGS_render_pose), (float)FLAGS_hand_alpha_pose,
            (float)FLAGS_hand_alpha_heatmap, (float)FLAGS_hand_render_threshold };
        opWrapper.configure(wrapperStructHand);
        // Extra functionality configuration (use op::WrapperStructExtra{} to disable it)
        const op::WrapperStructExtra wrapperStructExtra{
            FLAGS_3d, FLAGS_3d_min_views, FLAGS_identification, FLAGS_tracking, FLAGS_ik_threads };
        opWrapper.configure(wrapperStructExtra);
        // Output (comment or use default argument to disable any output)
        const op::WrapperStructOutput wrapperStructOutput{
            FLAGS_cli_verbose, op::String(FLAGS_write_keypoint), op::stringToDataFormat(FLAGS_write_keypoint_format),
            op::String(FLAGS_write_json), op::String(FLAGS_write_coco_json), FLAGS_write_coco_json_variants,
            FLAGS_write_coco_json_variant, op::String(FLAGS_write_images), op::String(FLAGS_write_images_format),
            op::String(FLAGS_write_video), FLAGS_write_video_fps, FLAGS_write_video_with_audio,
            op::String(FLAGS_write_heatmaps), op::String(FLAGS_write_heatmaps_format), op::String(FLAGS_write_video_3d),
            op::String(FLAGS_write_video_adam), op::String(FLAGS_write_bvh), op::String(FLAGS_udp_host),
            op::String(FLAGS_udp_port) };
        opWrapper.configure(wrapperStructOutput);
        // No GUI. Equivalent to: opWrapper.configure(op::WrapperStructGui{});
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

cv::Mat1b createROI(cv::Point2d& center, const cv::Mat1b& src, const cv::Rect& roiBase) {
    return src((roiBase + cv::Point(center)) & cv::Rect(cv::Point(0, 0), src.size()));
}

void save_pics(std::array<cv::Mat1b, 5000> saveImages) {    


    std::filesystem::create_directory(saveDir);
    std::filesystem::create_directory(saveDirr);

    for (int i_s = 0; i_s < EXPECTED_IMAGES; i_s++) {
        const std::string imageSavePath = saveDir + "/" + std::to_string(i_s) + ".jpg";
        cv::imwrite(imageSavePath, saveImages[i_s]);
    }
}

void PrintVec(std::vector<double>& v)
{
    for (int i = 0; i < v.size(); ++i)
        std::cout << v[i] << "  ";
    std::cout << std::endl;
}

std::vector<std::vector<double> > vectranspose(std::vector<std::vector<double> >& b)
{

    std::vector<std::vector<double> > trans_vec(b[0].size(), std::vector<double>());

    for (double i = 0; i < b.size(); i++)
    {
        for (double j = 0; j < b[i].size(); j++)
        {
            trans_vec[j].push_back(b[i][j]);
        }
    }

    return trans_vec;  
}

std::vector<std::vector<double>> LengthFormatter(std::vector<std::vector<double>> inputvect, double targetlength) {

    double unitleap = (inputvect.size() - 1) / (targetlength - 1);
    std::vector<double> indexlist;

    for (int i = 0; i < targetlength; i++) 
        indexlist.push_back(i * unitleap);
    

    std::vector<std::vector<double>> newlist;
    for (int j = 0; j < indexlist.size(); j++) {
        if (ceil(indexlist[j]) >= inputvect.size()) {
            newlist.push_back(inputvect[floor(indexlist[j])]);
            break;
        }

        if ((indexlist[j] - ceil(indexlist[j])) == 0 || (indexlist[j] - floor(indexlist[j])) == 0) {
            newlist.push_back(inputvect[round(indexlist[j])]);
        }
        else {//when the index is between two vectors
            double remain = indexlist[j] - floor(indexlist[j]);
            std::vector<double> veczeros(16, 0.00);
            std::transform(inputvect[ceil(indexlist[j])].begin(), inputvect[ceil(indexlist[j])].end(), inputvect[floor(indexlist[j])].begin(), veczeros.begin(), std::minus<double>());
            std::transform(veczeros.begin(), veczeros.end(), veczeros.begin(), [remain](double& c) { return c * remain; });
            std::transform(inputvect[floor(indexlist[j])].begin(), inputvect[floor(indexlist[j])].end(), veczeros.begin(), veczeros.begin(), std::plus<double>());
            //std::vector<double> atemp = inputvect[floor(indexlist[j])] + remain * (inputvect[ceil(indexlist[j])] - inputvect[floor(indexlist[j])]);
            newlist.push_back(veczeros);
        }
    }

    return newlist;
}

//Task number: 0 == kamae, 1 == men, 2==kote, 3 == dou, 4 == back to origin
void UR5e_run(int Task_Number, ur_rtde::RTDEControlInterface& urCtrl, std::vector<double> init_j){
    switch (Task_Number) {
    case 0://perform Kamae
        urCtrl.moveJ(KamaeState, 3.14, 5.00, false);
        std::cout << "Performing Kamae" << std::endl;
        break;
    case 1:
        urCtrl.moveJ(MenState, 3.14, 35.00, false);
        std::cout << "Performing Dou Defend" << std::endl;
        std::this_thread::sleep_for(std::chrono::duration<double>(1.5));
        urCtrl.moveJ(KamaeState, 3.14, 5.00, false);
        break;
    case 2:
        urCtrl.moveJ(KoteState, 3.14, 35.00, false);
        std::cout << "Performing Dou Defend" << std::endl;
        std::this_thread::sleep_for(std::chrono::duration<double>(1.5));
        urCtrl.moveJ(KamaeState, 3.14, 5.00, false);
        break;
    case 3:
        urCtrl.moveJ(DouState, 3.14, 35.00, false);
        std::cout << "Performing Dou Defend" << std::endl;
        std::this_thread::sleep_for(std::chrono::duration<double>(1.5));
        urCtrl.moveJ(KamaeState, 3.14, 5.00, false);
        break;
    case 4:
        urCtrl.moveJ(init_j, 3.14, 5.00, false);
        std::cout << "back to the origin" << std::endl;
        break;
    }

}

// Write a CSV file
void write_csv(std::vector<std::vector<double>> dataset) {
    char str[256];
    time_t rawtime;
    struct tm timeinfo;
    time(&rawtime);
    localtime_s(&timeinfo, &rawtime);
    strftime(str, sizeof(str), " %Y%m%d %H%M%S", &timeinfo);
    string stCurrentTime = str;
    string filename = "E:/FORLSTMPROCESSING/data" + stCurrentTime + ".csv";
    std::ofstream myFile(filename);
    // Send data to the stream
    for (int i = 0; i < dataset.at(0).size(); ++i)
    {
        for (int j = 0; j < dataset.size(); ++j)
        {
            myFile << dataset.at(j).at(i);
            if (j != dataset.size() - 1) myFile << ","; // No comma at end of line
        }
        myFile << "\n";
    }

    myFile.close(); 
}

std::vector<cv::Point2d> OpFlJointsupdate(std::vector<cv::Point2d> trackPointsCLK, cv::Mat1b cv_mat_image_left, cv::Mat1b cv_mat_image_old) {
    std::vector<uchar> status;
    std::vector<float> err;
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);

    cv::Mat frame_gray, old_gray;
    old_gray = cv_mat_image_old;
    frame_gray = cv_mat_image_left;

    std::vector<cv::Point2f> vector1, vector2;
    vector1.clear();
    vector2.clear();
    for (size_t i = 0; i < trackPointsCLK.size(); i++)
        vector1.push_back(cv::Point2f((double)trackPointsCLK[i].x, (double)trackPointsCLK[i].y));
    calcOpticalFlowPyrLK(old_gray, frame_gray, vector1, vector2, status, err, Size(20, 20), 0, criteria);//original size 30x30
    trackPointsCLK.clear();
    for (size_t i = 0; i < vector2.size(); i++)
        trackPointsCLK.push_back(cv::Point2d((double)vector2[i].x, (double)vector2[i].y));

    return trackPointsCLK;
}

std::vector<std::vector<double>> read_csv(std::string filename) {
    std::vector<std::vector<double>> labelread;
    std::ifstream myFile(filename);

    if (!myFile.is_open()) throw std::runtime_error("could not open file");
    std::string line, colname;
    double val;
    labelread.resize(6);
    while (std::getline(myFile, line)) {
        std::stringstream ss(line);
        int colIdx = 0;
        while (ss >> val) {
            labelread.at(colIdx).push_back(val);
            if (ss.peek() == ',') ss.ignore();
            colIdx++;
        }
    }
    myFile.close();

    return labelread;
}

std::vector<cv::Point2d> OPupdate(cv::Mat1b cv_mat_image_left, op::Wrapper opWrapper) {
    cv::Mat cv_mat_image1;
    cv::cvtColor(cv_mat_image_left, cv_mat_image1, cv::COLOR_RGB2BGR);
    const cv::Mat cvImageToProcess = cv_mat_image1;//cv::imread(imagePath);
    const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
    auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
    std::vector<cv::Point2d> templist = printKeypoints(datumProcessed);

    return templist;
}

int tutorialApiCpp()
{
    try
    {
        op::opLog("Starting OpenPose demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // Configuring OpenPose
        op::opLog("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{ op::ThreadManagerMode::Asynchronous };
        configureWrapper(opWrapper);

        // Starting OpenPose
        op::opLog("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        //..............initialize the ur10e.............
        const std::string urIP = "172.16.17.18";  // UR ip address
        ur_rtde::RTDEControlInterface urCtrl(urIP);
        ur_rtde::RTDEReceiveInterface urRecv(urIP);
        ur_rtde::RTDEIOInterface urDIO(urIP);

        auto init_q = urRecv.getActualQ();
        UR5e_run(0, urCtrl, init_q);

        torch::jit::script::Module module;
        module = torch::jit::load("C:/Users/savic/Downloads/model.pt");//load the classification model

        std::vector<torch::jit::IValue> inputstest;
        std::vector<torch::jit::IValue> inputs;

        inputstest.push_back(torch::randn({ 1, 99, 16 }).to(torch::kCUDA));
        at::Tensor outputtest = module.forward(inputstest).toTensor();

        int y_hattest = torch::log_softmax(outputtest, 1).argmax(1).item().toInt();
        std::cout << "test classification finish" << std::endl;//init and test the model

        const unsigned int imgWidth = 320; //画像の幅
        const unsigned int imgHeight = 240; //画像の高さ
        const unsigned int frameRate = 200; //フレームレート
        const unsigned long serialNumeber_left = 13051250; //カメラのシリアル番号
        const unsigned long serialNumeber_right = 13051150; //カメラのシリアル番号
        const unsigned int expTime = 2000; //露光時間 500 in origin
        const bool isBinning = true;

        //tracking
        spsc_queue<dispData_t> queDisp;
        dispData_t dispData;
        std::atomic<bool> isSaveImage = true;
        auto dispInfoPtr = make_unique<DispInfo>(queDisp, imgWidth, imgHeight);
        std::thread dispThread(std::ref(*dispInfoPtr), std::ref(isSaveImage));

        cv::Mat1b bin = cv::Mat1b::zeros(imgHeight, imgWidth); // 二値化画像
        std::vector<cv::Point2d> trackPoints; //トラッキング点
        std::vector<cv::Point2d> trackPointstmp; //save the clkpositions
        std::vector<cv::Point2d> trackPointsCLK; //save the clkpositions
        std::vector<cv::Point2d> trackPointsOP; //トラッキング点
        std::vector<std::vector<double>> trackPointsCLKinTS;//joints position in time sequence

        std::vector<cv::Point2d> trackPointsright; //トラッキング点
        std::vector<cv::Point2d> trackPointstmpright; //save the clkpositions
        std::vector<cv::Point2d> trackPointsCLKright; //save the clkpositions
        std::vector<cv::Point2d> trackPointsOPright; //トラッキング点
        std::vector<std::vector<double>> trackPointsCLKinTSright;

        const auto maxTrackNum = 25; //最大トラッキング点数
        cv::Point2d clkPos; //クリック位置
        const auto roiLen = 10.0; //ROIの一片の長さ 20 original
        auto roiBase = cv::Rect(-roiLen * 0.5, -roiLen * 0.5, roiLen, roiLen);
        cv::Mat1b roi = cv::Mat1b::zeros(roiLen, roiLen); //ROI画像
        int newpointflag = 0;
        int selectedjointset[5] = { 5, 6, 7, 0, 1 };//0, 1, 5, 6, 7 originally for left side
        int selectedjointsetright[3] = { 2, 3, 4 };//0, 1, 5, 6, 7 originally for right side

        Ximea cam_left(imgWidth, imgHeight, frameRate, serialNumeber_left, expTime, isBinning);
        Ximea cam_right(imgWidth, imgHeight, frameRate, serialNumeber_right, expTime, isBinning);
        cam_left.SetGain(10);
        cam_right.SetGain(10);
        cv::Mat1b cv_mat_image_left = cv::Mat1b::zeros(imgHeight, imgWidth);
        cv::Mat1b cv_mat_image_right = cv::Mat1b::zeros(imgHeight, imgWidth);

        std::array<cv::Mat1b, 5000> saveImages;
        for (int i_s = 0; i_s < saveImages.size(); i_s++) {
            saveImages[i_s] = cv::Mat1b::zeros(cv_mat_image_left.size());
        }
        int saveCount = 0; //保存画像カウンタ
        char filename[100] = "";

        printf("Starting acquisition...\n");
        cv::Mat1b cv_mat_image_old;
        trackPointsCLKinTS.resize(2 * ((sizeof(selectedjointset) / sizeof(selectedjointset[0]) + (sizeof(selectedjointsetright) / sizeof(selectedjointsetright[0])))));
        //trackPointsCLKinTS.resize(2 * (sizeof(selectedjointset) / sizeof(selectedjointset[0])));

        cv::Mat1b cv_mat_image_oldright;
        trackPointsCLKinTSright.resize(2 * (sizeof(selectedjointsetright) / sizeof(selectedjointsetright[0])));

        int endrecorder = EXPECTED_IMAGES;
        int endflag = 0;
        int writeflag = 0;
        for (int fCount = 0; fCount < EXPECTED_IMAGES; fCount += 1)
        {
            cv::Mat1b cv_mat_image_left = cam_left.GetNextImageOcvMat();
            cv::Mat1b cv_mat_image_right = cam_right.GetNextImageOcvMat();

            cv::Mat1b dst;
            cv::hconcat(cv_mat_image_left, cv_mat_image_right, dst);
            if (fCount == 0) {//initialize the track point (i % 15)
                trackPointsCLK.clear();
                trackPoints.clear();
                trackPointsOP.clear();
                trackPointsCLKright.clear();
                trackPointsright.clear();
                trackPointsOPright.clear();
                cv::Mat cv_mat_image1;
                cv::cvtColor(dst, cv_mat_image1, cv::COLOR_RGB2BGR);
                const cv::Mat cvImageToProcess = cv_mat_image1;//cv::imread(imagePath);
                const op::Matrix imageToProcess = OP_CV2OPCONSTMAT(cvImageToProcess);
                auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
                std::vector<cv::Point2d> templist = printKeypoints(datumProcessed);

                for (int k = 0; k < sizeof(selectedjointset) / sizeof(selectedjointset[0]); k++) {
                    trackPointsOP.push_back(templist[selectedjointset[k]]);
                    trackPoints.push_back(trackPointsOP[k]);
                    trackPointsCLK.push_back(trackPointsOP[k]);
                }

                for (int k = 0; k < sizeof(selectedjointsetright) / sizeof(selectedjointsetright[0]); k++) {
                    trackPointsOPright.push_back(templist[25 + selectedjointsetright[k]]);
                    trackPointsright.push_back(trackPointsOPright[k]);
                    trackPointsCLKright.push_back(trackPointsOPright[k]);
                }
                newpointflag = 1;
            }
            else {
                //*************************start optical flow*********************************
                cv::imwrite("C:/opfltest0118/002.jpg", dst);
                dst = imread("C:/opfltest0118/002.jpg", IMREAD_GRAYSCALE);
                trackPointsCLK = OpFlJointsupdate(trackPointsCLK, dst, cv_mat_image_old);
                trackPointsCLKright = OpFlJointsupdate(trackPointsCLKright, dst, cv_mat_image_old);// cv_mat_image_oldright);

            }
            cv_mat_image_old = dst;
            //cv_mat_image_old = cv_mat_image_left;
            cv_mat_image_oldright = cv_mat_image_right;

            if (isSaveImage) {
                if (saveCount < saveImages.size()) {
                    cv_mat_image_left.copyTo(saveImages[saveCount]);
                    saveCount++;
                }
                else {
                    std::cout << "The end" << std::endl;
                    break; 
                }
            }
            dst.copyTo(dispData.image);
            //cv_mat_image_left.copyTo(dispData.image);
            dispData.frameCount = fCount;

            std::vector<cv::Point2d> Ttpc(trackPointsCLK);//combine two cameras tracking joints
            Ttpc.insert(Ttpc.end(), trackPointsCLKright.begin(), trackPointsCLKright.end());
            Ttpc.clear();  // clear the vector
            // 2. only vector::insert
            Ttpc.insert(Ttpc.begin(), trackPointsCLK.begin(), trackPointsCLK.end());
            Ttpc.insert(Ttpc.end(), trackPointsCLKright.begin(), trackPointsCLKright.end());

            dispData.centroids = Ttpc;//CLK
            //dispData.centroids = trackPoints;
            queDisp.push(dispData);
            for (int i = 0; i < (trackPointsCLK.size() + trackPointsCLKright.size()); i++) {
                if (i < trackPointsCLK.size()) {
                    trackPointsCLKinTS[i * 2].push_back(trackPointsCLK[i].x);
                    trackPointsCLKinTS[i * 2 + 1].push_back(trackPointsCLK[i].y);
                }
                else {
                    trackPointsCLKinTS[i * 2].push_back(trackPointsCLKright[i - trackPointsCLK.size()].x);
                    trackPointsCLKinTS[i * 2 + 1].push_back(trackPointsCLKright[i - trackPointsCLK.size()].y);
                }
                
            }

            if (fCount >= 1) {
                double ab = sqrt((trackPointsCLK[0].x - trackPointsCLK[1].x) * (trackPointsCLK[0].x - trackPointsCLK[1].x) + (trackPointsCLK[0].y - trackPointsCLK[1].y) * (trackPointsCLK[0].y - trackPointsCLK[1].y));
                double bc = sqrt((trackPointsCLK[1].x - trackPointsCLK[2].x) * (trackPointsCLK[1].x - trackPointsCLK[2].x) + (trackPointsCLK[1].y - trackPointsCLK[2].y) * (trackPointsCLK[1].y - trackPointsCLK[2].y));
                double ca = sqrt((trackPointsCLK[0].x - trackPointsCLK[2].x) * (trackPointsCLK[0].x - trackPointsCLK[2].x) + (trackPointsCLK[0].y - trackPointsCLK[2].y) * (trackPointsCLK[0].y - trackPointsCLK[2].y));
                if (endflag == 0 && trackPointsCLK[2].y > trackPointsCLK[1].y&& trackPointsCLKinTS[5][trackPointsCLKinTS[1].size() - 3] > trackPointsCLKinTS[5][trackPointsCLKinTS[1].size() - 2] &&trackPointsCLKinTS[5][trackPointsCLKinTS[1].size()-2]> trackPointsCLKinTS[5][trackPointsCLKinTS[1].size() - 1]) {
                    //(ab * ab + bc * bc - ca * ca) / (2 * ab * bc) > 0.195 && 
                    op::opLog("waiting for start position", op::Priority::High);
                    int startrecorder = fCount;
                    trackPointsCLKinTS.clear();
                    trackPointsCLKinTS.resize(2 * ((sizeof(selectedjointset) / sizeof(selectedjointset[0]) +(sizeof(selectedjointsetright) / sizeof(selectedjointsetright[0])))));
                }
                //add speed constraint and direction constraint
                else if (endflag == 0 && (ab * ab + bc * bc - ca * ca) / (2 * ab * bc) < 0.195 && trackPointsCLK[2].y < trackPointsCLK[0].y && trackPointsCLKinTS[5][trackPointsCLKinTS[1].size() - 3] < trackPointsCLKinTS[5][trackPointsCLKinTS[1].size() - 2] && trackPointsCLKinTS[5][trackPointsCLKinTS[1].size() - 2] < trackPointsCLKinTS[5][trackPointsCLKinTS[1].size() - 1]) {
                    op::opLog("end position, start dtw processing", op::Priority::High);
                    op::opLog("end length: ", op::Priority::High);
                    UR5e_run(0, urCtrl, init_q);
                    endrecorder = fCount;
                    endflag = 1;    //stop the end position detecting
                    writeflag = 0;  //ready to write
                    std::cout << "The tracking process has finished, waiting for the formatter" << std::endl;
                    std::vector<std::vector<double>> vect = vectranspose(trackPointsCLKinTS);
                    std::vector<std::vector<double>> vectnew = LengthFormatter(vect, 99);

                    std::cout << "The formatter works fine, waiting for classification" << std::endl;
                    // Copying into a tensor
                    auto options = torch::TensorOptions().dtype(at::kFloat);
                    auto tensor1 = torch::zeros({ 99, 16 }, options);
                    for (int i = 0; i < 99; i++)
                        tensor1.slice(0, i, i + 1) = torch::from_blob(vectnew[i].data(), { 16 }, options);
                    torch::Tensor stackedtensor = torch::stack({ tensor1 });
                    inputs.push_back(stackedtensor.to(torch::kCUDA));

                    std::cout << "tensor stacked, wait for classify" << std::endl;
                    at::Tensor output = module.forward(inputs).toTensor();

                    std::cout << "Output got" << std::endl;
                    int y_hat = torch::log_softmax(output, 1).argmax(1).item().toInt();
                    std::cout << "successfully get the output of the classifiication" << std::endl;
                    switch (y_hat) {
                        case 0:
                            UR5e_run(3, urCtrl, init_q);
                            std::cout << "Dou" << std::endl;
                            break;
                        case 1:
                            UR5e_run(1, urCtrl, init_q);

                            std::cout << "Men" << std::endl;
                            break;
                        case 2:
                            UR5e_run(1, urCtrl, init_q);
                            std::cout << "Men" << std::endl;
                            break;
                        case 3:
                            UR5e_run(2, urCtrl, init_q);
                            std::cout << "Kote" << std::endl;
                            break;
                    }
                }

                if (writeflag == 0 && fCount >= endrecorder + frameRate * 0.4) {
                    write_csv(trackPointsCLKinTS);
                    writeflag = 1;
                }
                else if (fCount >= endrecorder + 3 * frameRate) {
                    endflag = 0;
                    fCount = -1;
                    endrecorder = 100000;
                }
            }

        }
        dispThread.join();
        cam_left.StopAcquisition();
        cam_left.Close();
        cam_right.StopAcquisition();
        cam_right.Close();

        return 0;
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return -1;
    }
}


int main(int argc, char* argv[])
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    return tutorialApiCpp();
}
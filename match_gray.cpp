//
// Created by 兰育青 on 2020-02-23.
//

#include "line2Dup.h"
#include <memory>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <chrono>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/types_c.h"
#include <string>
#include<ctime>
#include <jsoncpp/json/json.h>


using namespace std;
using namespace cv;


//先设置这个prefix
static std::string prefix = "/home/lipschitz/Documents/Code/C_Code/linemod_center/";

clock_t start,finish;

void readFileJson(string path){
    Json::Reader reader;
    Json::Value root;

    ifstream in(path, ios::binary);

    if (!in.is_open())
    {
        cout << "Error opening file\n";
        return;
    }

    if (reader.parse(in, root))
    {
        string camera = root["camera"].asString();
        for ( int i=0; i<root["roi"].size(); i++)
        {
            string name = root["roi"][i]["name"].asString();
            float lx = root["roi"][i]["lx"].asFloat();
            float ly = root["roi"][i]["ly"].asFloat();
            float rx = root["roi"][i]["rx"].asFloat();
            float ry = root["roi"][i]["ry"].asFloat();

            std::cout << "name is " << name << " " << lx<< " " << ly<< " " << rx << " " << ry << std::endl;
        }
    }
}


cv::Mat Resize(cv::Mat src,float scale) {

    cv::Mat dst;

    float scaleW = scale;
    //定义图像的大小，宽度缩小80%
    float scaleH = scaleW;
    //定义图像的大小，高度缩小80%

    int width = static_cast<float>(src.cols*scaleW);
    //定义想要扩大或者缩小后的宽度，src.cols为原图像的宽度，乘以80%则得到想要的大小，并强制转换成float型
    int height = static_cast<float>(src.rows*scaleH);
    //定义想要扩大或者缩小后的高度，src.cols为原图像的高度，乘以80%则得到想要的大小，并强制转换成float型

    resize(src, dst, cv::Size(width, height));
    //重新定义大小的函数


    return dst;
}

// NMS, got from cv::dnn so we don't need opencv contrib
// just collapse it
namespace  cv_dnn {
    namespace
    {

        template <typename T>
        static inline bool SortScorePairDescend(const std::pair<float, T>& pair1,
                                                const std::pair<float, T>& pair2)
        {
            return pair1.first > pair2.first;
        }

    } // namespace

    inline void GetMaxScoreIndex(const std::vector<float>& scores, const float threshold, const int top_k,
                                 std::vector<std::pair<float, int> >& score_index_vec)
    {
        for (size_t i = 0; i < scores.size(); ++i)
        {
            if (scores[i] > threshold)
            {
                score_index_vec.push_back(std::make_pair(scores[i], i));
            }
        }
        std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                         SortScorePairDescend<int>);
        if (top_k > 0 && top_k < (int)score_index_vec.size())
        {
            score_index_vec.resize(top_k);
        }
    }

    template <typename BoxType>
    inline void NMSFast_(const std::vector<BoxType>& bboxes,
                         const std::vector<float>& scores, const float score_threshold,
                         const float nms_threshold, const float eta, const int top_k,
                         std::vector<int>& indices, float (*computeOverlap)(const BoxType&, const BoxType&))
    {
        CV_Assert(bboxes.size() == scores.size());
        std::vector<std::pair<float, int> > score_index_vec;
        GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec);

        // Do nms.
        float adaptive_threshold = nms_threshold;
        indices.clear();
        for (size_t i = 0; i < score_index_vec.size(); ++i) {
            const int idx = score_index_vec[i].second;
            bool keep = true;
            for (int k = 0; k < (int)indices.size() && keep; ++k) {
                const int kept_idx = indices[k];
                float overlap = computeOverlap(bboxes[idx], bboxes[kept_idx]);
                keep = overlap <= adaptive_threshold;
            }
            if (keep)
                indices.push_back(idx);
            if (keep && eta < 1 && adaptive_threshold > 0.5) {
                adaptive_threshold *= eta;
            }
        }
    }


// copied from opencv 3.4, not exist in 3.0
    template<typename _Tp> static inline
    double jaccardDistance__(const Rect_<_Tp>& a, const Rect_<_Tp>& b) {
        _Tp Aa = a.area();
        _Tp Ab = b.area();

        if ((Aa + Ab) <= std::numeric_limits<_Tp>::epsilon()) {
            // jaccard_index = 1 -> distance = 0
            return 0.0;
        }

        double Aab = (a & b).area();
        // distance = 1 - jaccard_index
        return 1.0 - Aab / (Aa + Ab - Aab);
    }

    template <typename T>
    static inline float rectOverlap(const T& a, const T& b)
    {
        return 1.f - static_cast<float>(jaccardDistance__(a, b));
    }

    void NMSBoxes(const std::vector<Rect>& bboxes, const std::vector<float>& scores,
                  const float score_threshold, const float nms_threshold,
                  std::vector<int>& indices, const float eta=1, const int top_k=0)
    {
        NMSFast_(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, rectOverlap);
    }

}

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
                (clock_::now() - beg_).count(); }
    void out(std::string message = ""){
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};


struct circle_center{
    string name;
    float data[4];
};


void angle_test(string mode = "test", string case_num="case_f",string objid="56C86", int num_feature =100,bool write_flag= false){

    int t1=2;
    int t2=4;

    string templ_path=prefix+"data/"+objid+"/test_templ_";
    string info_path=prefix+"data/"+objid+"/test_info_";
    string result_dir=prefix+"data/"+objid;
    fstream _file;
    _file.open(result_dir, ios::in);
    if(!_file)
    {
        string cmd = "mkdir "+result_dir;
        system(cmd.data());
    }
    _file.close();

    Json::Reader reader;
    Json::Value root;

    ifstream fin(prefix+case_num+"/jsonfile/camera-CCD"+objid+".json", ios::binary);

    if (!fin.is_open())
    {
        cout << "Error opening file\n";
        return;
    }

    if(! reader.parse(fin, root))
    {
        cout << "parse Json Error"<<std::endl;
        return;
    }

    int num = 0;
    circle_center * cc_ptr;
    ifstream in(prefix+"/point/" + to_string(stoi(objid)-1) +".txt", ios::binary);
    if(in)
    {
        string line;
        getline(in, line);
        num = std::stoi(line);
        std::cout << "num is " << num << std::endl;
        cc_ptr = new circle_center[num];
        for (int i=0; i<num; i++)
        {
            getline(in, line);
            char * strc = new char[strlen(line.c_str())+1];
            strcpy(strc, line.c_str());
            vector<string> res;
            char * token = strtok(strc, " ");
            cc_ptr[i].name = string(token);
            int tmp = 0;
            while(token != NULL)
            {
                token = strtok(NULL, " ");
                if (token != NULL)
                    cc_ptr[i].data[tmp++]  = stof(string(token));
            }
            delete [] strc;
        }
    }
    else
    {
        cout << "no such file " << endl;
        return;
    }

    if (mode == "train")
    {
        //Mat ori_img = imread(prefix+path+impath);
        std::cout << "num is : " << num << std::endl;
        for (int i=0; i<num; i++){
            line2Dup::Detector detector(num_feature, {t1});
            Mat img = imread(prefix+case_num+"/roi/"+objid+"/"+cc_ptr[i].name+".png");
            std::cout << " path is " << prefix+case_num+"/roi/"+objid+"/"+cc_ptr[i].name+".png" << std::endl;

            // float lx, ly, rx, ry;
            // for ( int j=0; j<root["roi"].size(); j++)
            // {
            //     if(root["roi"][j]["name"].asString() == cc_ptr[i].name)
            //     {
            //         lx = root["roi"][j]["lx"].asFloat();
            //         ly = root["roi"][j]["ly"].asFloat();
            //         rx = root["roi"][j]["rx"].asFloat();
            //         ry = root["roi"][j]["ry"].asFloat();
            //         break;
            //     }
            // }

            // Rect test_roi = Rect((int)lx, (int)ly, (int)(rx-lx), (int)(ry-ly));
            // Mat img = ori_img(test_roi);

            std::cout << "img cols and rows is : " << img.cols << " " << img.rows << std::endl;
            float scaleW = 1.0 > round(600 / img.cols)? 1.0 : round(600/img.cols);
            float scaleH = 1.0 > round(600 / img.rows)? 1.0 : round(600/img.rows);
            float scale = max(scaleW, scaleH);
            img = Resize(img, scale);

            std::cout << "scale is : " << scale << std::endl;

            cv::cvtColor(img, img, CV_BGR2GRAY);
            Mat tmp_img =img.clone();

            cv::Canny(img, img, 15, 30, 3, true);
            //cv::threshold(img, img, 70, 255, CV_THRESH_BINARY);
            //imshow("tmp", img);
            //waitKey(0);

            std::cout << "train now" << std::endl;

            float center_x=0.0, center_y=0.0, zero=0.0;
            for ( int j=0; j<root["roi"].size(); j++)
            {
                if(root["roi"][j]["name"].asString() == cc_ptr[i].name)
                {
                    center_x = cc_ptr[i].data[0] - max(zero, root["roi"][j]["lx"].asFloat());
                    center_y = cc_ptr[i].data[1] - max(zero, root["roi"][j]["ly"].asFloat());
                    break;
                }
            }

            float max_dia = max(cc_ptr[i].data[3], cc_ptr[i].data[2]);
            std::cout << "train here: "<< max_dia << " " << center_x << " " << center_y << std::endl;
            Rect roi(ceil(center_x-max_dia)*scale, ceil(center_y - max_dia)*scale, ceil(max_dia*2*scale), ceil(max_dia*2*scale));
            Mat mask = Mat::zeros(img.size(), CV_8UC1);
            mask(roi).setTo(Scalar(255));

            std::cout << "mask here" << std::endl;

            shape_based_matching::shapeInfo_producer shapes(img, mask);

            shapes.produce_infos();
            std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
            string class_id = cc_ptr[i].name;
            for( auto & info: shapes.infos){

                int templ_id = detector.addTemplate(shapes.src_of(info), class_id, shapes.mask_of(info), num_feature);
                std::cout << "templ_id: " << templ_id << std::endl;
                if(templ_id != -1){
                    auto templ = detector.getTemplates(class_id, templ_id);
                    info.center_x = center_x*scale - (float)templ[0].tl_x;
                    info.center_y = center_y*scale - (float)templ[0].tl_y;
                    infos_have_templ.push_back(info);
                }
            }

            std::cout << "teml num is " << detector.numTemplates(class_id) << std::endl;

            cv::Vec3b randColor;
                    randColor[0] = 0;
                    randColor[1] = 0;
                    randColor[2] = 255;

            cv::cvtColor(tmp_img, tmp_img, COLOR_GRAY2BGR);

            for (int i=0; i<detector.numTemplates(class_id); i++)
            {
                auto templ = detector.getTemplates(class_id, i);
                for (int i = 0; i < templ[0].features.size(); i++) {
                        auto feat = templ[0].features[i];
                        cv::circle(tmp_img, {feat.x + templ[0].tl_x, feat.y + templ[0].tl_y}, 2, randColor, -1);
                    }
                for (auto & info: infos_have_templ){
                    cv::circle(tmp_img, {round(info.center_x) + templ[0].tl_x, round(info.center_y) + templ[0].tl_y}, 1, Vec3b(0, 255, 0), -1);
                }
            }

            fstream _tfile;
            _tfile.open(prefix+"test/"+objid, ios::in);
            if(!_tfile)
            {
                string cmd = "mkdir "+ prefix+"test/"+objid;
                system(cmd.data());
            }

            cv::imwrite(prefix+"test/"+objid+"/"+cc_ptr[i].name+".jpg", tmp_img);
            _tfile.close();
            detector.writeClasses(templ_path+cc_ptr[i].name +".yaml");
            shapes.save_infos(infos_have_templ, info_path+cc_ptr[i].name +".yaml");
            std::cout << "train end" << std::endl << std::endl;
        }
    }

    else if ( mode == "test")
    {
        std::cout << prefix+case_num+"/pic/CCD"+objid+".bmp" << std::endl;
        Mat ori_img = imread(prefix+case_num+"/pic/CCD"+objid+".bmp");
        float * c_x = new float[num];
        float * c_y = new float[num];

        for (int i=0; i<num; i++){
            line2Dup::Detector detector(num_feature, {t1});
            std::vector<std::string> class_ids;
            class_ids.push_back(cc_ptr[i].name);

            std::cout << "reading templates: " << templ_path+cc_ptr[i].name +".yaml" << std::endl;
            detector.readClasses(class_ids, templ_path+cc_ptr[i].name +".yaml");
            auto infos = shape_based_matching::shapeInfo_producer::load_infos(info_path+cc_ptr[i].name +".yaml");

            std::cout << " reading infos " << info_path+cc_ptr[i].name +".yaml" << std::endl;

            //Mat test_img = imread(prefix+case_num+"/pic/CCD"+objid+".bmp");
            float lx, ly, rx, ry;
            for ( int j=0; j<root["roi"].size(); j++)
            {
                if(root["roi"][j]["name"].asString() == cc_ptr[i].name)
                {
                    lx = max(root["roi"][j]["lx"].asFloat(), (float)0);
                    ly = max(root["roi"][j]["ly"].asFloat(), (float)0);
                    rx = min(root["roi"][j]["rx"].asFloat(), (float)ori_img.cols);
                    ry = min(root["roi"][j]["ry"].asFloat(), (float)ori_img.rows);
                    break;
                }
            }

            std::cout << "rect is " << lx << " " << ly << " " << rx << " " << ry << std::endl;
            std::cout << "img cols and rows is " << ori_img.cols << " " << ori_img.rows << std::endl;
            Rect test_roi = Rect((int)lx, (int)ly, (int)(rx-lx), (int)(ry-ly));
            Mat test_img = ori_img(test_roi);
            

            float scaleW = 1.0 > round(600 / test_img.cols)? 1.0: round(600/test_img.cols);
            float scaleH = 1.0 > round(600 / test_img.rows)? 1.0: round(600/test_img.rows);
            float scale = max(scaleW, scaleH);
            test_img = Resize(test_img, scale);
            cv::cvtColor(test_img, test_img, CV_BGR2GRAY);
            Mat tmp_img = test_img.clone();

            //cv::threshold(test_img, test_img, 70, 255, CV_THRESH_BINARY);

            //cv::Canny(test_img, test_img, 15, 30, 3, true);
            //imshow("tmp", test_img);
            //waitKey(0);

            assert(!test_img.empty() && "check your img path");

            int stride = 16;
            int n = test_img.rows/stride;
            int m = test_img.cols/stride;
            Rect roi(0, 0, stride*m , stride*n);
            Mat img = test_img(roi).clone();
            tmp_img = tmp_img(roi).clone();
            assert(img.isContinuous());

            Timer timer;
            auto matches = detector.match(img, 40, class_ids);
            timer.out();

            if(tmp_img.channels() == 1) cv::cvtColor(tmp_img, tmp_img, CV_GRAY2BGR);

            std::cout << "matches.size(): " << matches.size() << std::endl;

            // vector<Rect> boxes;
            // vector<float> scores;
            // vector<int> idxs;
            // for(auto match: matches){
            //     Rect box;
            //     box.x = match.x;
            //     box.y = match.y;

            //     auto templ = detector.getTemplates(cc_ptr[i].name, match.template_id);

            //     box.width = templ[0].width;
            //     box.height = templ[0].height;
            //     boxes.push_back(box);
            //     scores.push_back(match.similarity);
            // }
            // std::cout << "after nms !" << std::endl;
            // cv_dnn::NMSBoxes(boxes, scores, 0, 0.1f, idxs);
            // for ( int t = 0; t<idxs.size(); t++)
            //     std::cout << "idx is : " << matches[t].similarity << std::endl;
            
            size_t top5 = 1;
            if(top5 > matches.size()) top5=matches.size();

            //for(auto idx:idxs){

            float tmp_x=0, tmp_y=0;
            scale = 1.0;
            tmp_img = Resize(tmp_img, 1.0/scale);
            for (size_t ii = 0; ii<top5; ii++)
            {
                cout << ii << " " << matches.size() << endl;
                auto match = matches[ii];
                std::cout << " template id is : "<< match.template_id << std::endl;
                auto templ = detector.getTemplates(cc_ptr[i].name, match.template_id);

                std::cout << "center_x: " << infos[match.template_id].center_x << 
                " center_y: " << infos[match.template_id].center_y << std::endl;

                for (int i = 0; i<templ[0].features.size(); i++){
                    auto feat = templ[0].features[i];
                    cv::circle(tmp_img,  {(feat.x + match.x)/scale, (feat.y + match.y)/scale}, 1, cv::Vec3b(0, 0, 255), -1);
                }
                cv::putText(tmp_img, to_string(int(round(match.similarity))),Point((match.x - 10)/scale, (match.y - 3)/scale), FONT_HERSHEY_PLAIN, 2, cv::Vec3b(0, 255, 255));
                tmp_x += (float)match.x;
                tmp_y += (float)match.y;
            }

            tmp_x /= top5;
            tmp_y /= top5;

            for (auto & info:infos){
                std::cout << "info lllllllll" << std::endl;
                cv::circle(tmp_img, {(round(info.center_x) + tmp_x)/scale, (round(info.center_y) + tmp_y)/scale}, 1, Vec3b(0, 255, 0), -1);
                c_x[i] = (info.center_x + tmp_x)/scale + lx;
                c_y[i] = (info.center_y + tmp_y)/scale + ly;
                std::cout << "cx cy is : " << c_x[i] << " " << c_y[i] << std::endl;
            }

            fstream _tfile;
            _tfile.open(prefix+"result/"+objid, ios::in);
            if(!_tfile)
            {
                string cmd = "mkdir "+ prefix+"result/"+objid;
                system(cmd.data());
            }

            cv::imwrite(prefix+"result/"+objid+"/"+cc_ptr[i].name+".jpg", tmp_img);
            _tfile.close();

            std::cout << "+++++++++++++++++++++++" << std::endl;
        }

        ofstream outfile;
        outfile.open(prefix+objid+"_output.txt", ios::out | ios::trunc);
        outfile << num << "\n";
        for (int i=0; i<num; i++)
        {
            outfile << cc_ptr[i].name << " " << c_x[i] << " " << c_y[i] << std::endl;
        }

        delete [] c_x;
        delete [] c_y;
    }

    delete [] cc_ptr;
    in.close();
    fin.close();
}



void MIPP_test(){
    std::cout << "MIPP tests" << std::endl;
    std::cout << "----------" << std::endl << std::endl;

    std::cout << "Instr. type:       " << mipp::InstructionType                  << std::endl;
    std::cout << "Instr. full type:  " << mipp::InstructionFullType              << std::endl;
    std::cout << "Instr. version:    " << mipp::InstructionVersion               << std::endl;
    std::cout << "Instr. size:       " << mipp::RegisterSizeBit       << " bits" << std::endl;
    std::cout << "Instr. lanes:      " << mipp::Lanes                            << std::endl;
    std::cout << "64-bit support:    " << (mipp::Support64Bit    ? "yes" : "no") << std::endl;
    std::cout << "Byte/word support: " << (mipp::SupportByteWord ? "yes" : "no") << std::endl;

#ifndef has_max_int8_t
    std::cout << "in this SIMD, int8 max is not inplemented by MIPP" << std::endl;
#endif

#ifndef has_shuff_int8_t
    std::cout << "in this SIMD, int8 shuff is not inplemented by MIPP" << std::endl;
#endif

    std::cout << "----------" << std::endl << std::endl;
}


int main(){
    srand((unsigned) time(NULL));//diff color
    MIPP_test();

    start=clock();
    cout << "开始计算时间 .... " << endl;

    // angle_test("train","case_test","1th","TRAINDATA/2/",
    //            1.0,1.0 ,50); // train
    for (int i=1; i<=16; i++)
    {    angle_test("train","traindata",to_string(i),80); // train

        angle_test("test","testdata",to_string(i),40); // train
    }

    //angle_test("test","case_test","1th","case_test/test_img/WechatIMG27.png",
    //           0.95,1.05,40, true); // test


    finish=clock();
    cout << "time: "<<(double)(finish-start)/ CLOCKS_PER_SEC   << " (s) "<< endl;
}
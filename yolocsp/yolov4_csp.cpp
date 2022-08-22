#include "jf_Virgo.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
// #include "common.hpp"
#include <json/reader.h>
#include <json/value.h>
#include <json/writer.h>
#include "model_version.h"
#include "glog/logging.h"
#include "yololayer.h"
#include "CudaImg.h"
#include <TensorRT_utils.h>
#include "calibratorcsp.h"
#include "preprocess.h"
// #include "mish.h"

using namespace nvinfer1;
// REGISTER_TENSORRT_PLUGIN(MishPluginCreator);
REGISTER_TENSORRT_PLUGIN(YoloPluginCreator);

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

std::string trimzhn(std::string s);

namespace JF_VIRGO
{

    bool cmp(DetectedObject &a, DetectedObject &b)
    {
        return a.prob > b.prob;
    }

    struct TensorInfo
    {
        std::string blobName;
        uint32_t stride{0};
        uint32_t stride_h{0};
        uint32_t stride_w{0};
        uint32_t gridSize{0};
        uint32_t grid_h{0};
        uint32_t grid_w{0};
        uint32_t numClasses{0};
        uint32_t numBBoxes{0};
        uint64_t volume{0};
        std::vector<uint32_t> masks;
        std::vector<float> anchors;
        int bindingIndex{-1};
        float *hostBuffer{nullptr};
    };

    class Yolov4CspPrivate
    {
    public:
        std::string model_path;
        std::string json_path;
        std::string cfg_path;
        std::string calibration_path;
        std::string jfe_file;
        int gpu_id;
        int INPUT_W;
        int INPUT_H;
        // int model_type;
        int run_mode;
        int keyid;
        int channels;
        int secret_id;
        int CLASS_NUM;
        int BATCH_SIZE = 1;
        int gen_jfe = 1;
        float nms_thresh = 0.5;
        float conf_thresh = 0.5;
        int DETECTION_SIZE;
        int OUTPUT_SIZE;
        std::vector<cv::Size> vecSize; //原图尺寸
        int imgWidth;
        int imgHeight;
        Logger gLogger;
        const char *INPUT_BLOB_NAME = "data";
        const char *OUTPUT_BLOB_NAME = "prob";
        int inputIndex;
        int outputIndex;
        IRuntime *runtime;
        ICudaEngine *engine;
        IExecutionContext *context;
        cudaStream_t stream;
        void *buffers[2];
        float *dataP;
        float *prob;
        int maxSize;
        void *img_gpu_data8u;
        // void *img_gpu_data32ftmp;
        // void *m_gpuInputBuffer;

        void *m_gpuInput8U;
        const int MAX_OUTPUT_BBOX = 1000;
        int CHECK_COUNT = 3; //每层anchor数量
        int outsize = 0;     //输出层数
        std::vector<std::map<std::string, std::string>> m_configBlocks;
        std::vector<float> anchorsVec;
        std::vector<std::vector<float>> anchorsCsp;
        std::string anchorsString;
        std::vector<TensorInfo> m_OutputTensors;
        uint32_t outputTensorCount = 0;

    public:
        ~Yolov4CspPrivate()
        {
            // Release stream and buffers
            cudaStreamDestroy(stream);
            CHECK(cudaFree(buffers[inputIndex]));
            CHECK(cudaFree(buffers[outputIndex]));
            if (img_gpu_data8u)
            {
                cudaFree(img_gpu_data8u);
            }
            if (m_gpuInput8U)
            {
                cudaFree(m_gpuInput8U);
            }
            // if (img_gpu_data32ftmp)
            // {
            //     cudaFree(img_gpu_data32ftmp);
            // }
            // if (m_gpuInputBuffer)
            // {
            //     cudaFree(m_gpuInputBuffer);
            // }
            // Destroy the engine
            if (context)
            {
                context->destroy();
            }
            if (engine)
            {
                engine->destroy();
            }
            if (runtime)
            {
                runtime->destroy();
            }

            delete[] dataP;
            delete[] prob;
        }
        //转模型
        Yolov4CspPrivate(std::string weightspath, std::string jsonpath, std::string cfgpath, int gpuid) : model_path(weightspath),
                                                                                                          json_path(jsonpath),
                                                                                                          cfg_path(cfgpath),
                                                                                                          gpu_id(gpuid)
        {
            Json::Value root;
            Json::Reader reader;
            std::ifstream is(json_path);
            if (!is.is_open())
            {
                LOG(INFO) << "****open json file failed.***";
            }
            reader.parse(is, root);
            // INPUT_W = root["width"].asInt();
            // INPUT_H = root["height"].asInt();
            // channels = root["channels"].asInt();
            // CLASS_NUM = root["classes"].asInt();
            BATCH_SIZE = root["batch_size"].asInt();
            run_mode = root["run_mode"].asInt();
            gen_jfe = root["gen_jfe"].asInt();
            keyid = root["keyid"].asInt();
            secret_id = root["secret_id"].asInt();
            calibration_path = root["calibration_path"].asString();
            std::cout << " gen_jfe:" << gen_jfe << " calibration_path:" << calibration_path << std::endl;
            readCfg2json();
            assert(CLASS_NUM > 0);
            cudaSetDevice(gpu_id);
            std::cout << "BATCH_SIZE:" << BATCH_SIZE << " INPUT_W:" << INPUT_W << " INPUT_H:" << INPUT_H << " channels:" << channels << " CLASS_NUM:" << CLASS_NUM << " calibration_path:" << calibration_path << " run_mode:" << run_mode << std::endl;
            createYolov4CspEngine();
        }
        // 推理
        Yolov4CspPrivate(std::string weightspath, std::string jsonpath, int gpuid, float thresh) : jfe_file(weightspath), json_path(jsonpath), gpu_id(gpuid), conf_thresh(thresh)
        {
            Json::Value root;
            Json::Reader reader;
            std::ifstream is(json_path);
            if (!is.is_open())
            {
                LOG(INFO) << "****open json file failed.***";
                return;
            }
            reader.parse(is, root);
            INPUT_W = root["width"].asInt();
            INPUT_H = root["height"].asInt();
            run_mode = root["run_mode"].asInt();
            gen_jfe = root["gen_jfe"].asInt();
            keyid = root["keyid"].asInt();
            channels = root["channels"].asInt();
            secret_id = root["secret_id"].asInt();
            CLASS_NUM = root["classes"].asInt();
            outsize = root["output"].asInt();
            calibration_path = root["calibration_path"].asString();
            anchorsString = root["anchors"].asString();
            BATCH_SIZE = root["batch_size"].asInt();
            readAnchors();
            cudaSetDevice(gpu_id);
            DETECTION_SIZE = sizeof(DetectionCsp) / sizeof(float);
            OUTPUT_SIZE = DETECTION_SIZE * MAX_OUTPUT_BBOX + 1;
            std::cout << "DETECTION_SIZE:" << DETECTION_SIZE << ",OUTPUT_SIZE:" << OUTPUT_SIZE << std::endl;
            std::cout << " INPUT_W:" << INPUT_W << " INPUT_H:" << INPUT_H << " channels:" << channels << " CLASS_NUM:" << CLASS_NUM << " BATCH_SIZE:" << BATCH_SIZE << " calibration_path:" << calibration_path << " run_mode" << run_mode << std::endl;
            deserializeYolov4CspEngine();
            dataP = new float[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
            prob = new float[BATCH_SIZE * OUTPUT_SIZE];
            maxSize = BATCH_SIZE * 700 * 700; // init size
            CUDA_CHECK(cudaMalloc((void **)&img_gpu_data8u, maxSize * channels * sizeof(uchar)));
            // CUDA_CHECK(cudaMalloc((void **)&img_gpu_data32ftmp, INPUT_H * INPUT_W * sizeof(float3)));
            // CUDA_CHECK(cudaMalloc((void **)&m_gpuInputBuffer, BATCH_SIZE * INPUT_H * INPUT_W * sizeof(float3)));
            CUDA_CHECK(cudaMalloc((void **)&m_gpuInput8U, BATCH_SIZE * INPUT_H * INPUT_W * sizeof(uchar3)));
        }

        void readAnchors()
        {
            while (!anchorsString.empty())
            {
                int npos = anchorsString.find_first_of('_');
                if (npos != -1)
                {
                    float anchor = std::stof(trimzhn(anchorsString.substr(0, npos)));
                    anchorsVec.push_back(anchor);
                    anchorsString.erase(0, npos + 1);
                }
                else
                {
                    float anchor = std::stof(trimzhn(anchorsString));
                    anchorsVec.push_back(anchor);
                    break;
                }
            }
            int j = 0;
            for (int i = 0; i < outsize; i++)
            {
                std::vector<float> anchorsTmp;
                while (j < CHECK_COUNT * 2 * (i + 1))
                {
                    anchorsTmp.push_back(anchorsVec[j++]);
                }
                anchorsCsp.push_back(anchorsTmp);
            }
            // for (int i = 0; i < anchorsCsp.size(); i++)
            // {
            //     for (int k = 0; k < anchorsCsp[0].size(); k++)
            //     {
            //         std::cout << "anchorsCsp " << i << ",:" << anchorsCsp[i][k] << std::endl;
            //     }
            // }
        }

        void createYolov4CspEngine()
        {
            IHostMemory *modelStream{nullptr};
            IBuilder *builder = createInferBuilder(gLogger);
            IBuilderConfig *config = builder->createBuilderConfig();
            ICudaEngine *engine = createEngine(BATCH_SIZE, builder, config, DataType::kFLOAT);
            assert(engine != nullptr);
            // (*modelStream) = engine->serialize();
            modelStream = engine->serialize();
            assert(modelStream != nullptr);
            // std::ofstream p("/data3/zy/code/ScaledYOLOv4/runs/yolov4_csp_416/tmp-fp32.engine", std::ios::binary);
            // if (!p)
            // {
            //     std::cerr << "could not open plan output file" << std::endl;
            // }
            // p.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());
            // todo gen engine???
            assert(gen_jfe == 1);
            if (gen_jfe)
            {
                int modelinfo_len = 1024;
                char *chver = (char *)malloc(modelinfo_len);
                char *file_path = (char *)malloc(modelinfo_len);
                memset(chver, 0, modelinfo_len);
                memset(file_path, 0, modelinfo_len);
                if (run_mode == 0)
                {
                    gen_file_name(model_path, chver, file_path, BATCH_SIZE, "int8", secret_id);
                }
                else if (run_mode == 1)
                {
                    gen_file_name(model_path, chver, file_path, BATCH_SIZE, "fp16", secret_id);
                }
                else if (run_mode == 2)
                {
                    gen_file_name(model_path, chver, file_path, BATCH_SIZE, "fp32", secret_id);
                }
                else
                {
                    LOG(INFO) << "run_mode wrong, gen_file_name fail !!!";
                }
                string path_with_ext = file_path;
                JF_VIRGO::Cjfssl jfssl = JF_VIRGO::Cjfssl(secret_id);
                JF_VIRGO::MEM_FILE *encry_file = (JF_VIRGO::MEM_FILE *)malloc(sizeof(JF_VIRGO::MEM_FILE));
                // LOG(INFO)<<modelStream->size();
                encry_file->bufflen_encrypt = modelStream->size() + modelinfo_len;
                // LOG(INFO)<< encry_file->bufflen_encrypt ;
                encry_file->buffs = (char *)malloc(encry_file->bufflen_encrypt);
                memcpy(encry_file->buffs, (modelStream->data()), modelStream->size());
                std::string encryptedFile = path_with_ext;
                memcpy(encry_file->buffs + modelStream->size(), chver, modelinfo_len);
                free(chver);
                free(file_path);
                jfssl.mem_Encrypt_File(encry_file, encryptedFile.c_str());
                jfe_file = encryptedFile;
                cout << "Serialized encrypted file cached at location : " << encryptedFile << endl;
            }
            engine->destroy();
            builder->destroy();
            modelStream->destroy();
        }

        void deserializeYolov4CspEngine()
        {
            LOG(INFO) << "Loading TRT Engine...";
            JF_VIRGO::Cjfssl jfssl = JF_VIRGO::Cjfssl(secret_id);
            JF_VIRGO::MEM_FILE *decry_file = (JF_VIRGO::MEM_FILE *)malloc(sizeof(JF_VIRGO::MEM_FILE));
            int ret = jfssl.mem_open_Decrypt_File(jfe_file.c_str(), decry_file);
            if (ret < 0)
            {
                LOG(INFO) << "Decrypt " << jfe_file << " file failed!!!";
            }
            LOG(INFO) << "decry_file " << decry_file->bufflen_decrypt << " " << ret;
            string ver;
            int modelinfo_len = 1024;
            char *chver = (char *)malloc(modelinfo_len);
            int modelSize = decry_file->bufflen_decrypt - modelinfo_len;
            void *modelMem = malloc(modelSize);
            memcpy(modelMem, decry_file->buffs, modelSize);
            runtime = createInferRuntime(gLogger);
            assert(runtime != nullptr);
            engine = runtime->deserializeCudaEngine(modelMem, modelSize);
            assert(engine != nullptr);
            context = engine->createExecutionContext();
            assert(context != nullptr);
            free(decry_file);
            assert(engine->getNbBindings() == 2);
            cout << "engine deserializeEngine success" << endl;
            // todo malloc释放
            inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
            outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
            assert(inputIndex == 0);
            assert(outputIndex == 1);
            // Create GPU buffers on device
            CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
            CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
            // Create stream
            CHECK(cudaStreamCreate(&stream));
        }

        void readCfg2json()
        {
            m_configBlocks = parseConfigFile(cfg_path);
            parseConfigBlocks(m_configBlocks, m_OutputTensors);
            writeJson(json_path);
            int j = 0;
            for (int i = 0; i < outsize; i++)
            {
                std::vector<float> anchorsTmp;
                while (j < CHECK_COUNT * 2 * (i + 1))
                {
                    anchorsTmp.push_back(anchorsVec[j++]);
                }
                anchorsCsp.push_back(anchorsTmp);
            }
            // anchorsCsp
            // for (int i = 0; i < anchorsCsp.size(); i++)
            // {
            //     for (int k = 0; k < anchorsCsp[0].size(); k++)
            //     {
            //         std::cout << "anchorsCsp " << i << ",:" << anchorsCsp[i][k] << std::endl;
            //     }
            // }
        }

        std::vector<std::map<std::string, std::string>> parseConfigFile(const std::string cfgFilePath)
        {
            // assert(fileExistszhn(cfgFilePath));
            std::ifstream file(cfgFilePath);
            assert(file.good());
            std::string line;
            std::vector<std::map<std::string, std::string>> blocks;
            std::map<std::string, std::string> block;

            while (getline(file, line))
            {
                if (line.size() == 0)
                    continue;
                if (line.front() == '#')
                    continue;
                line = trimzhn(line);
                // std::cout << "line:" << line << std::endl;
                if (line.front() == '[')
                {
                    if (block.size() > 0)
                    {
                        blocks.push_back(block);
                        block.clear();
                    }
                    std::string key = "type";
                    std::string value = trimzhn(line.substr(1, line.size() - 2));
                    block.insert(std::pair<std::string, std::string>(key, value));
                }
                else
                {
                    int cpos = line.find('=');
                    std::string key = trimzhn(line.substr(0, cpos));
                    std::string value = trimzhn(line.substr(cpos + 1));
                    block.insert(std::pair<std::string, std::string>(key, value));
                }
            }
            blocks.push_back(block);
            return blocks;
        }

        void parseConfigBlocks(std::vector<std::map<std::string, std::string>> &m_configBlocks, std::vector<TensorInfo> &m_OutputTensors)
        {
            for (auto block : m_configBlocks)
            {
                if (block.at("type") == "net")
                {
                    assert((block.find("height") != block.end()) && "Missing 'height' param in network cfg");
                    assert((block.find("width") != block.end()) && "Missing 'width' param in network cfg");
                    assert((block.find("channels") != block.end()) && "Missing 'channels' param in network cfg");

                    INPUT_H = std::stoul(block.at("height"));
                    INPUT_W = std::stoul(block.at("width"));
                    channels = std::stoul(block.at("channels"));
                    assert(INPUT_H == INPUT_W);
                    // m_InputSize = m_InputC * m_InputH * m_InputW;
                }
                else if ((block.at("type") == "region") || (block.at("type") == "yolo"))
                {
                    assert((block.find("num") != block.end()) && std::string("Missing 'num' param in " + block.at("type") + " layer").c_str());
                    assert((block.find("classes") != block.end()) && std::string("Missing 'classes' param in " + block.at("type") + " layer")
                                                                         .c_str());
                    assert((block.find("anchors") != block.end()) && std::string("Missing 'anchors' param in " + block.at("type") + " layer")
                                                                         .c_str());

                    TensorInfo outputTensor;
                    std::string anchorString = block.at("anchors");
                    while (!anchorString.empty())
                    {
                        int npos = anchorString.find_first_of(',');
                        if (npos != -1)
                        {
                            float anchor = std::stof(trimzhn(anchorString.substr(0, npos)));
                            outputTensor.anchors.push_back(anchor);
                            anchorString.erase(0, npos + 1);
                        }
                        else
                        {
                            float anchor = std::stof(trimzhn(anchorString));
                            outputTensor.anchors.push_back(anchor);
                            break;
                        }
                    }
                    assert((block.find("mask") != block.end()) && std::string("Missing 'mask' param in " + block.at("type") + " layer")
                                                                      .c_str());

                    std::string maskString = block.at("mask");
                    while (!maskString.empty())
                    {
                        int npos = maskString.find_first_of(',');
                        if (npos != -1)
                        {
                            uint32_t mask = std::stoul(trimzhn(maskString.substr(0, npos)));
                            outputTensor.masks.push_back(mask);
                            maskString.erase(0, npos + 1);
                        }
                        else
                        {
                            uint32_t mask = std::stoul(trimzhn(maskString));
                            outputTensor.masks.push_back(mask);
                            break;
                        }
                    }
                    outputTensor.numBBoxes = outputTensor.masks.size() > 0
                                                 ? outputTensor.masks.size()
                                                 : std::stoul(trimzhn(block.at("num")));
                    outputTensor.numClasses = std::stoul(block.at("classes"));

                    // for (int i = 0; i < outputTensor.numClasses; ++i)
                    // {
                    //     m_ClassNames.push_back(std::to_string(i));
                    // }

                    m_OutputTensors.push_back(outputTensor);
                }
            }
        }

        void writeJson(std::string m_Jsonpath)
        {
            Json::Value root;
            Json::Reader reader;
            Json::StyledWriter writer;
            std::ifstream infs(m_Jsonpath);
            std::string str_cfg_json((std::istreambuf_iterator<char>(infs)), std::istreambuf_iterator<char>());
            if (reader.parse(str_cfg_json, root))
            {
                std::cout << "json reader success" << std::endl;
            }
            else
            {
                std::cout << "json reader error" << std::endl;
            }
            root["height"] = INPUT_H; // todo
            root["width"] = INPUT_W;
            root["channels"] = channels;
            for (uint32_t i = 0; i < m_configBlocks.size(); ++i)
            {
                if (m_configBlocks.at(i).at("type") == "yolo")
                {
                    outsize++;
                    TensorInfo &curYoloTensor = m_OutputTensors.at(outputTensorCount);
                    CLASS_NUM = curYoloTensor.numClasses;
                    std::string anchor1 = "";
                    for (int m = 0; m < curYoloTensor.anchors.size(); m++)
                    {
                        anchor1 += std::to_string(curYoloTensor.anchors[m]) + "_";
                        if (outsize == 1)
                        {
                            anchorsVec.push_back(curYoloTensor.anchors[m]);
                        }
                        // std::cout << "curYoloTensor.anchors[m]:" << m << "  " << std::to_string(curYoloTensor.anchors[m]) << std::endl;
                    }
                    root["anchors"] = anchor1;
                    ++outputTensorCount;
                }
            }
            root["classes"] = CLASS_NUM;
            root["output"] = outsize;
            std::string jsonnew = writer.write(root);
            std::ofstream outfile;
            outfile.open(m_Jsonpath, std::ios_base::out | std::ios_base::trunc);
            outfile << jsonnew << std::endl;
            outfile.close();
        }

        std::map<std::string, Weights> loadWeights(const std::string file)
        {
            std::cout << "Loading weights: " << file << std::endl;
            std::map<std::string, Weights> weightMap;
            // Open weights file
            std::ifstream input(file);
            assert(input.is_open() && "Unable to load weight file.");
            // Read number of weight blobs
            int32_t count;
            input >> count;
            assert(count > 0 && "Invalid weight map file.");
            while (count--)
            {
                Weights wt{DataType::kFLOAT, nullptr, 0};
                uint32_t size;
                // Read name and type of blob
                std::string name;
                input >> name >> std::dec >> size;
                wt.type = DataType::kFLOAT;
                // Load blob
                uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
                for (uint32_t x = 0, y = size; x < y; ++x)
                {
                    input >> std::hex >> val[x];
                }
                wt.values = val;
                wt.count = size;
                weightMap[name] = wt;
            }
            return weightMap;
        }

        IScaleLayer *addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, std::string lname, float eps)
        {
            float *gamma = (float *)weightMap[lname + ".weight"].values;
            float *beta = (float *)weightMap[lname + ".bias"].values;
            float *mean = (float *)weightMap[lname + ".running_mean"].values;
            float *var = (float *)weightMap[lname + ".running_var"].values;
            int len = weightMap[lname + ".running_var"].count;
            // std::cout << "len " << len << std::endl;

            float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
            for (int i = 0; i < len; i++)
            {
                scval[i] = gamma[i] / sqrt(var[i] + eps);
            }
            Weights scale{DataType::kFLOAT, scval, len};

            float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
            for (int i = 0; i < len; i++)
            {
                shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
            }
            Weights shift{DataType::kFLOAT, shval, len};

            float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
            for (int i = 0; i < len; i++)
            {
                pval[i] = 1.0;
            }
            Weights power{DataType::kFLOAT, pval, len};

            weightMap[lname + ".scale"] = scale;
            weightMap[lname + ".shift"] = shift;
            weightMap[lname + ".power"] = power;
            IScaleLayer *scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
            assert(scale_1);
            return scale_1;
        }

        ILayer *convBnMish(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int outch, int ksize, int s, int p, int linx)
        {
            // std::cout << linx << std::endl;
            Weights emptywts{DataType::kFLOAT, nullptr, 0};
            IConvolutionLayer *conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
            assert(conv1);
            conv1->setStrideNd(DimsHW{s, s});
            conv1->setPaddingNd(DimsHW{p, p});

            IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-4);

            auto creator = getPluginRegistry()->getPluginCreator("Mish_TRT", "1");
            const PluginFieldCollection *pluginData = creator->getFieldNames();
            IPluginV2 *pluginObj = creator->createPlugin(("mish" + std::to_string(linx)).c_str(), pluginData);
            ITensor *inputTensors[] = {bn1->getOutput(0)};
            auto mish = network->addPluginV2(&inputTensors[0], 1, *pluginObj);
            return mish;
        }
        ILayer *upSample(INetworkDefinition *network, std::map<std::string, Weights> &weightMap, ITensor &input, int channels)
        {
            float *deval = reinterpret_cast<float *>(malloc(sizeof(float) * channels * 2 * 2));
            for (int i = 0; i < channels * 2 * 2; i++)
            {
                deval[i] = 1.0;
            }
            Weights deconvwts{DataType::kFLOAT, deval, channels * 2 * 2};
            Weights emptywts{DataType::kFLOAT, nullptr, 0};
            IDeconvolutionLayer *deconv = network->addDeconvolutionNd(input, channels, DimsHW{2, 2}, deconvwts, emptywts);
            deconv->setStrideNd(DimsHW{2, 2});
            deconv->setNbGroups(channels);

            return deconv;
        }
        // Creat the engine using only the API and not any parser.
        ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder, IBuilderConfig *config, DataType dt)
        {
            INetworkDefinition *network = builder->createNetworkV2(0U);

            // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
            ITensor *data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{INPUT_H, INPUT_W, 3});
            assert(data);

            std::map<std::string, Weights> weightMap = loadWeights(model_path);

            // Custom preprocess (NHWC->NCHW, BGR->RGB, [0, 255]->[0, 1](Normalize))
            Preprocess preprocess{maxBatchSize, 3, INPUT_H, INPUT_W};
            IPluginCreator *preprocess_creator = getPluginRegistry()->getPluginCreator("preprocess", "1");
            IPluginV2 *preprocess_plugin = preprocess_creator->createPlugin("preprocess_plugin", (PluginFieldCollection *)&preprocess);
            IPluginV2Layer *preprocess_layer = network->addPluginV2(&data, 1, *preprocess_plugin);
            preprocess_layer->setName("preprocess_layer");
            ITensor *prep = preprocess_layer->getOutput(0);

            // define each layer.
            auto l0 = convBnMish(network, weightMap, *prep, 32, 3, 1, 1, 0);
            // Downsample
            auto l1 = convBnMish(network, weightMap, *l0->getOutput(0), 64, 3, 2, 1, 1);
            auto l2 = convBnMish(network, weightMap, *l1->getOutput(0), 32, 1, 1, 0, 2);
            auto l3 = convBnMish(network, weightMap, *l2->getOutput(0), 64, 3, 1, 1, 3);
            auto ew4 = network->addElementWise(*l3->getOutput(0), *l1->getOutput(0), ElementWiseOperation::kSUM);
            // Downsample
            auto l5 = convBnMish(network, weightMap, *ew4->getOutput(0), 128, 3, 2, 1, 5);
            auto l6 = convBnMish(network, weightMap, *l5->getOutput(0), 64, 1, 1, 0, 6);
            auto l7 = l5;
            auto l8 = convBnMish(network, weightMap, *l7->getOutput(0), 64, 1, 1, 0, 8);
            auto l9 = convBnMish(network, weightMap, *l8->getOutput(0), 64, 1, 1, 0, 9);
            auto l10 = convBnMish(network, weightMap, *l9->getOutput(0), 64, 3, 1, 1, 10);
            auto ew11 = network->addElementWise(*l10->getOutput(0), *l8->getOutput(0), ElementWiseOperation::kSUM);
            auto l12 = convBnMish(network, weightMap, *ew11->getOutput(0), 64, 1, 1, 0, 12);
            auto l13 = convBnMish(network, weightMap, *l12->getOutput(0), 64, 3, 1, 1, 13);
            auto ew14 = network->addElementWise(*l13->getOutput(0), *ew11->getOutput(0), ElementWiseOperation::kSUM);
            auto l15 = convBnMish(network, weightMap, *ew14->getOutput(0), 64, 1, 1, 0, 15);
            ITensor *inputTensors16[] = {l15->getOutput(0), l6->getOutput(0)};
            auto cat16 = network->addConcatenation(inputTensors16, 2);
            auto l17 = convBnMish(network, weightMap, *cat16->getOutput(0), 128, 1, 1, 0, 17);
            // Downsample
            auto l18 = convBnMish(network, weightMap, *l17->getOutput(0), 256, 3, 2, 1, 18);
            auto l19 = convBnMish(network, weightMap, *l18->getOutput(0), 128, 1, 1, 0, 19);
            auto l20 = l18;
            auto l21 = convBnMish(network, weightMap, *l20->getOutput(0), 128, 1, 1, 0, 21);
            auto l22 = convBnMish(network, weightMap, *l21->getOutput(0), 128, 1, 1, 0, 22);
            auto l23 = convBnMish(network, weightMap, *l22->getOutput(0), 128, 3, 1, 1, 23);
            auto ew24 = network->addElementWise(*l23->getOutput(0), *l21->getOutput(0), ElementWiseOperation::kSUM);
            auto l25 = convBnMish(network, weightMap, *ew24->getOutput(0), 128, 1, 1, 0, 25);
            auto l26 = convBnMish(network, weightMap, *l25->getOutput(0), 128, 3, 1, 1, 26);
            auto ew27 = network->addElementWise(*l26->getOutput(0), *ew24->getOutput(0), ElementWiseOperation::kSUM);
            auto l28 = convBnMish(network, weightMap, *ew27->getOutput(0), 128, 1, 1, 0, 28);
            auto l29 = convBnMish(network, weightMap, *l28->getOutput(0), 128, 3, 1, 1, 29);
            auto ew30 = network->addElementWise(*l29->getOutput(0), *ew27->getOutput(0), ElementWiseOperation::kSUM);
            auto l31 = convBnMish(network, weightMap, *ew30->getOutput(0), 128, 1, 1, 0, 31);
            auto l32 = convBnMish(network, weightMap, *l31->getOutput(0), 128, 3, 1, 1, 32);
            auto ew33 = network->addElementWise(*l32->getOutput(0), *ew30->getOutput(0), ElementWiseOperation::kSUM);
            auto l34 = convBnMish(network, weightMap, *ew33->getOutput(0), 128, 1, 1, 0, 34);
            auto l35 = convBnMish(network, weightMap, *l34->getOutput(0), 128, 3, 1, 1, 35);
            auto ew36 = network->addElementWise(*l35->getOutput(0), *ew33->getOutput(0), ElementWiseOperation::kSUM);
            auto l37 = convBnMish(network, weightMap, *ew36->getOutput(0), 128, 1, 1, 0, 37);
            auto l38 = convBnMish(network, weightMap, *l37->getOutput(0), 128, 3, 1, 1, 38);
            auto ew39 = network->addElementWise(*l38->getOutput(0), *ew36->getOutput(0), ElementWiseOperation::kSUM);
            auto l40 = convBnMish(network, weightMap, *ew39->getOutput(0), 128, 1, 1, 0, 40);
            auto l41 = convBnMish(network, weightMap, *l40->getOutput(0), 128, 3, 1, 1, 41);
            auto ew42 = network->addElementWise(*l41->getOutput(0), *ew39->getOutput(0), ElementWiseOperation::kSUM);
            auto l43 = convBnMish(network, weightMap, *ew42->getOutput(0), 128, 1, 1, 0, 43);
            auto l44 = convBnMish(network, weightMap, *l43->getOutput(0), 128, 3, 1, 1, 44);
            auto ew45 = network->addElementWise(*l44->getOutput(0), *ew42->getOutput(0), ElementWiseOperation::kSUM);
            auto l46 = convBnMish(network, weightMap, *ew45->getOutput(0), 128, 1, 1, 0, 46);
            ITensor *inputTensor47[] = {l46->getOutput(0), l19->getOutput(0)};
            auto cat47 = network->addConcatenation(inputTensor47, 2);
            auto l48 = convBnMish(network, weightMap, *cat47->getOutput(0), 256, 1, 1, 0, 48);
            // Downsample
            auto l49 = convBnMish(network, weightMap, *l48->getOutput(0), 512, 3, 2, 1, 49);
            auto l50 = convBnMish(network, weightMap, *l49->getOutput(0), 256, 1, 1, 0, 50);
            auto l51 = l49;
            auto l52 = convBnMish(network, weightMap, *l51->getOutput(0), 256, 1, 1, 0, 52);
            auto l53 = convBnMish(network, weightMap, *l52->getOutput(0), 256, 1, 1, 0, 53);
            auto l54 = convBnMish(network, weightMap, *l53->getOutput(0), 256, 3, 1, 1, 54);
            auto ew55 = network->addElementWise(*l54->getOutput(0), *l52->getOutput(0), ElementWiseOperation::kSUM);
            auto l56 = convBnMish(network, weightMap, *ew55->getOutput(0), 256, 1, 1, 0, 56);
            auto l57 = convBnMish(network, weightMap, *l56->getOutput(0), 256, 3, 1, 1, 57);
            auto ew58 = network->addElementWise(*l57->getOutput(0), *ew55->getOutput(0), ElementWiseOperation::kSUM);

            auto l59 = convBnMish(network, weightMap, *ew58->getOutput(0), 256, 1, 1, 0, 59);
            auto l60 = convBnMish(network, weightMap, *l59->getOutput(0), 256, 3, 1, 1, 60);
            auto ew61 = network->addElementWise(*l60->getOutput(0), *ew58->getOutput(0), ElementWiseOperation::kSUM);
            auto l62 = convBnMish(network, weightMap, *ew61->getOutput(0), 256, 1, 1, 0, 62);
            auto l63 = convBnMish(network, weightMap, *l62->getOutput(0), 256, 3, 1, 1, 63);
            auto ew64 = network->addElementWise(*l63->getOutput(0), *ew61->getOutput(0), ElementWiseOperation::kSUM);
            auto l65 = convBnMish(network, weightMap, *ew64->getOutput(0), 256, 1, 1, 0, 65);
            auto l66 = convBnMish(network, weightMap, *l65->getOutput(0), 256, 3, 1, 1, 66);
            auto ew67 = network->addElementWise(*l66->getOutput(0), *ew64->getOutput(0), ElementWiseOperation::kSUM);
            auto l68 = convBnMish(network, weightMap, *ew67->getOutput(0), 256, 1, 1, 0, 68);
            auto l69 = convBnMish(network, weightMap, *l68->getOutput(0), 256, 3, 1, 1, 69);
            auto ew70 = network->addElementWise(*l69->getOutput(0), *ew67->getOutput(0), ElementWiseOperation::kSUM);
            auto l71 = convBnMish(network, weightMap, *ew70->getOutput(0), 256, 1, 1, 0, 71);
            auto l72 = convBnMish(network, weightMap, *l71->getOutput(0), 256, 3, 1, 1, 72);
            auto ew73 = network->addElementWise(*l72->getOutput(0), *ew70->getOutput(0), ElementWiseOperation::kSUM);
            auto l74 = convBnMish(network, weightMap, *ew73->getOutput(0), 256, 1, 1, 0, 74);
            auto l75 = convBnMish(network, weightMap, *l74->getOutput(0), 256, 3, 1, 1, 75);
            auto ew76 = network->addElementWise(*l75->getOutput(0), *ew73->getOutput(0), ElementWiseOperation::kSUM);
            auto l77 = convBnMish(network, weightMap, *ew76->getOutput(0), 256, 1, 1, 0, 77);
            ITensor *inputTensors78[] = {l77->getOutput(0), l50->getOutput(0)};
            auto cat78 = network->addConcatenation(inputTensors78, 2);
            auto l79 = convBnMish(network, weightMap, *cat78->getOutput(0), 512, 1, 1, 0, 79);
            // Downsample
            auto l80 = convBnMish(network, weightMap, *l79->getOutput(0), 1024, 3, 2, 1, 80);
            auto l81 = convBnMish(network, weightMap, *l80->getOutput(0), 512, 1, 1, 0, 81);
            auto l82 = l80;
            auto l83 = convBnMish(network, weightMap, *l82->getOutput(0), 512, 1, 1, 0, 83);
            auto l84 = convBnMish(network, weightMap, *l83->getOutput(0), 512, 1, 1, 0, 84);
            auto l85 = convBnMish(network, weightMap, *l84->getOutput(0), 512, 3, 1, 1, 85);
            auto ew86 = network->addElementWise(*l85->getOutput(0), *l83->getOutput(0), ElementWiseOperation::kSUM);
            auto l87 = convBnMish(network, weightMap, *ew86->getOutput(0), 512, 1, 1, 0, 87);
            auto l88 = convBnMish(network, weightMap, *l87->getOutput(0), 512, 3, 1, 1, 88);
            auto ew89 = network->addElementWise(*l88->getOutput(0), *ew86->getOutput(0), ElementWiseOperation::kSUM);
            auto l90 = convBnMish(network, weightMap, *ew89->getOutput(0), 512, 1, 1, 0, 90);
            auto l91 = convBnMish(network, weightMap, *l90->getOutput(0), 512, 3, 1, 1, 91);
            auto ew92 = network->addElementWise(*l91->getOutput(0), *ew89->getOutput(0), ElementWiseOperation::kSUM);
            auto l93 = convBnMish(network, weightMap, *ew92->getOutput(0), 512, 1, 1, 0, 93);
            auto l94 = convBnMish(network, weightMap, *l93->getOutput(0), 512, 3, 1, 1, 94);
            auto ew95 = network->addElementWise(*l94->getOutput(0), *ew92->getOutput(0), ElementWiseOperation::kSUM);
            auto l96 = convBnMish(network, weightMap, *ew95->getOutput(0), 512, 1, 1, 0, 96);
            ITensor *inputTensors97[] = {l96->getOutput(0), l81->getOutput(0)};
            auto cat97 = network->addConcatenation(inputTensors97, 2);
            auto l98 = convBnMish(network, weightMap, *cat97->getOutput(0), 1024, 1, 1, 0, 98);
            //###########################
            auto l99 = convBnMish(network, weightMap, *l98->getOutput(0), 512, 1, 1, 0, 99);
            auto l100 = l98;
            auto l101 = convBnMish(network, weightMap, *l100->getOutput(0), 512, 1, 1, 0, 101);
            auto l102 = convBnMish(network, weightMap, *l101->getOutput(0), 512, 3, 1, 1, 102);
            auto l103 = convBnMish(network, weightMap, *l102->getOutput(0), 512, 1, 1, 0, 103);
            //### SPP ###
            auto pool104 = network->addPoolingNd(*l103->getOutput(0), PoolingType::kMAX, DimsHW{5, 5});
            pool104->setPaddingNd(DimsHW{2, 2});
            pool104->setStrideNd(DimsHW{1, 1});
            auto l105 = l103;
            auto pool106 = network->addPoolingNd(*l105->getOutput(0), PoolingType::kMAX, DimsHW{9, 9});
            pool106->setPaddingNd(DimsHW{4, 4});
            pool106->setStrideNd(DimsHW{1, 1});
            auto l107 = l103;
            auto pool108 = network->addPoolingNd(*l107->getOutput(0), PoolingType::kMAX, DimsHW{13, 13});
            pool108->setPaddingNd(DimsHW{6, 6});
            pool108->setStrideNd(DimsHW{1, 1});
            ITensor *inputTensors109[] = {pool108->getOutput(0), pool106->getOutput(0), pool104->getOutput(0), l103->getOutput(0)};
            auto cat109 = network->addConcatenation(inputTensors109, 4);
            auto l110 = convBnMish(network, weightMap, *cat109->getOutput(0), 512, 1, 1, 0, 110);
            auto l111 = convBnMish(network, weightMap, *l110->getOutput(0), 512, 3, 1, 1, 111);
            ITensor *inputTensors112[] = {l111->getOutput(0), l99->getOutput(0)};
            auto cat112 = network->addConcatenation(inputTensors112, 2);
            auto l113 = convBnMish(network, weightMap, *cat112->getOutput(0), 512, 1, 1, 0, 113);
            auto l114 = convBnMish(network, weightMap, *l113->getOutput(0), 256, 1, 1, 0, 114);
            auto deconv115 = upSample(network, weightMap, *l114->getOutput(0), 256);
            auto l116 = l79;
            auto l117 = convBnMish(network, weightMap, *l116->getOutput(0), 256, 1, 1, 0, 117);
            ITensor *inputTensors118[] = {l117->getOutput(0), deconv115->getOutput(0)};
            auto cat118 = network->addConcatenation(inputTensors118, 2);
            auto l119 = convBnMish(network, weightMap, *cat118->getOutput(0), 256, 1, 1, 0, 119);
            auto l120 = convBnMish(network, weightMap, *l119->getOutput(0), 256, 1, 1, 0, 120);
            auto l121 = l119;
            auto l122 = convBnMish(network, weightMap, *l121->getOutput(0), 256, 1, 1, 0, 122);
            auto l123 = convBnMish(network, weightMap, *l122->getOutput(0), 256, 3, 1, 1, 123);
            auto l124 = convBnMish(network, weightMap, *l123->getOutput(0), 256, 1, 1, 0, 124);
            auto l125 = convBnMish(network, weightMap, *l124->getOutput(0), 256, 3, 1, 1, 125);
            ITensor *inputTensors126[] = {l125->getOutput(0), l120->getOutput(0)};
            auto cat126 = network->addConcatenation(inputTensors126, 2);
            auto l127 = convBnMish(network, weightMap, *cat126->getOutput(0), 256, 1, 1, 0, 127);
            auto l128 = convBnMish(network, weightMap, *l127->getOutput(0), 128, 1, 1, 0, 128);
            auto deconv129 = upSample(network, weightMap, *l128->getOutput(0), 128);

            auto l130 = l48;
            auto l131 = convBnMish(network, weightMap, *l130->getOutput(0), 128, 1, 1, 0, 131);
            ITensor *inputTensors132[] = {l131->getOutput(0), deconv129->getOutput(0)};
            auto cat132 = network->addConcatenation(inputTensors132, 2);
            auto l133 = convBnMish(network, weightMap, *cat132->getOutput(0), 128, 1, 1, 0, 133);
            auto l134 = convBnMish(network, weightMap, *l133->getOutput(0), 128, 1, 1, 0, 134);
            auto l135 = l133;
            auto l136 = convBnMish(network, weightMap, *l135->getOutput(0), 128, 1, 1, 0, 136);
            auto l137 = convBnMish(network, weightMap, *l136->getOutput(0), 128, 3, 1, 1, 137);
            auto l138 = convBnMish(network, weightMap, *l137->getOutput(0), 128, 1, 1, 0, 138);
            auto l139 = convBnMish(network, weightMap, *l138->getOutput(0), 128, 3, 1, 1, 139);
            ITensor *inputTensors140[] = {l139->getOutput(0), l134->getOutput(0)};
            auto cat140 = network->addConcatenation(inputTensors140, 2);
            auto l141 = convBnMish(network, weightMap, *cat140->getOutput(0), 128, 1, 1, 0, 141);
            auto l142 = convBnMish(network, weightMap, *l141->getOutput(0), 256, 3, 1, 1, 142);

            IConvolutionLayer *conv143 = network->addConvolutionNd(*l142->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.143.Conv2d.weight"], weightMap["module_list.143.Conv2d.bias"]);
            assert(conv143);
            // 144 is yolo layer

            auto l145 = l141;
            auto l146 = convBnMish(network, weightMap, *l145->getOutput(0), 256, 3, 2, 1, 146);
            ITensor *inputTensors147[] = {l146->getOutput(0), l127->getOutput(0)};
            auto cat147 = network->addConcatenation(inputTensors147, 2);
            auto l148 = convBnMish(network, weightMap, *cat147->getOutput(0), 256, 1, 1, 0, 148);
            auto l149 = convBnMish(network, weightMap, *l148->getOutput(0), 256, 1, 1, 0, 149);
            auto l150 = l148;
            auto l151 = convBnMish(network, weightMap, *l150->getOutput(0), 256, 1, 1, 0, 151);
            auto l152 = convBnMish(network, weightMap, *l151->getOutput(0), 256, 3, 1, 1, 152);
            auto l153 = convBnMish(network, weightMap, *l152->getOutput(0), 256, 1, 1, 0, 153);
            auto l154 = convBnMish(network, weightMap, *l153->getOutput(0), 256, 3, 1, 1, 154);
            ITensor *inputTensors155[] = {l154->getOutput(0), l149->getOutput(0)};
            auto cat155 = network->addConcatenation(inputTensors155, 2);
            auto l156 = convBnMish(network, weightMap, *cat155->getOutput(0), 256, 1, 1, 0, 156);
            auto l157 = convBnMish(network, weightMap, *l156->getOutput(0), 512, 3, 1, 1, 157);

            IConvolutionLayer *conv158 = network->addConvolutionNd(*l157->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.158.Conv2d.weight"], weightMap["module_list.158.Conv2d.bias"]);
            assert(conv158);
            // 159 is yolo layer

            auto l160 = l156;
            auto l161 = convBnMish(network, weightMap, *l160->getOutput(0), 512, 3, 2, 1, 161);
            ITensor *inputTensors162[] = {l161->getOutput(0), l113->getOutput(0)};
            auto cat162 = network->addConcatenation(inputTensors162, 2);
            auto l163 = convBnMish(network, weightMap, *cat162->getOutput(0), 512, 1, 1, 0, 163);
            auto l164 = convBnMish(network, weightMap, *l163->getOutput(0), 512, 1, 1, 0, 164);
            auto l165 = l163;
            auto l166 = convBnMish(network, weightMap, *l165->getOutput(0), 512, 1, 1, 0, 166);
            auto l167 = convBnMish(network, weightMap, *l166->getOutput(0), 512, 3, 1, 1, 167);
            auto l168 = convBnMish(network, weightMap, *l167->getOutput(0), 512, 1, 1, 0, 168);
            auto l169 = convBnMish(network, weightMap, *l168->getOutput(0), 512, 3, 1, 1, 169);
            ITensor *inputTensors170[] = {l169->getOutput(0), l164->getOutput(0)};
            auto cat170 = network->addConcatenation(inputTensors170, 2);
            auto l171 = convBnMish(network, weightMap, *cat170->getOutput(0), 512, 1, 1, 0, 171);
            auto l172 = convBnMish(network, weightMap, *l171->getOutput(0), 1024, 3, 1, 1, 172);

            IConvolutionLayer *conv173 = network->addConvolutionNd(*l172->getOutput(0), 3 * (CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.173.Conv2d.weight"], weightMap["module_list.173.Conv2d.bias"]);
            assert(conv173);
            // 174 is yolo layer

            auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
            // origin
            // const PluginFieldCollection *pluginData = creator->getFieldNames();
            PluginField plugin_fields[2];
            int netinfo[4] = {CLASS_NUM, INPUT_W, INPUT_H, MAX_OUTPUT_BBOX};
            plugin_fields[0].data = netinfo;
            plugin_fields[0].length = 4;
            plugin_fields[0].name = "netinfo";
            plugin_fields[0].type = PluginFieldType::kFLOAT32;
            std::vector<YoloKernel> kernels;
            int scale = 8;
            for (size_t i = 0; i < anchorsCsp.size(); i++)
            {
                YoloKernel kernel;
                kernel.width = INPUT_W / scale;
                kernel.height = INPUT_H / scale;
                memcpy(kernel.anchors, &anchorsCsp[i][0], anchorsCsp[i].size() * sizeof(float));
                kernels.push_back(kernel);
                scale *= 2;
            }
            plugin_fields[1].data = &kernels[0];
            plugin_fields[1].length = kernels.size();
            plugin_fields[1].name = "kernels";
            plugin_fields[1].type = PluginFieldType::kFLOAT32;
            PluginFieldCollection pluginData;
            pluginData.nbFields = 2;
            pluginData.fields = plugin_fields;

            IPluginV2 *pluginObj = creator->createPlugin("yololayer", &pluginData);

            ITensor *inputTensors_yolo[] = {conv143->getOutput(0), conv158->getOutput(0), conv173->getOutput(0)};
            auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

            yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
            std::cout << "set name out" << std::endl;
            network->markOutput(*yolo->getOutput(0)); // todo
            // Build engine
            builder->setMaxBatchSize(maxBatchSize);
            config->setMaxWorkspaceSize(16 * (1 << 20)); // 16MB
            if (run_mode == 1)
            {
                config->setFlag(BuilderFlag::kFP16);
            }
            else if (run_mode == 0)
            {
                // todo int8
                assert(builder->platformHasFastInt8());
                config->setFlag(BuilderFlag::kINT8);
                Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, calibration_path.c_str(), "int8calib.table", INPUT_BLOB_NAME);
                config->setInt8Calibrator(calibrator);
                // todo calibrator
            }
            ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
            std::cout << "build out" << std::endl;
            // Don't need the network any more
            network->destroy();
            // Release host memory
            for (auto &mem : weightMap)
            {
                free((void *)(mem.second.values));
            }
            return engine;
        }

        void detect(std::vector<cv::Mat> &vBatchImages, std::vector<std::vector<DetectedObject>> &arrDetection, std::vector<cv::Size> &vec_size)
        {
            assert(vec_size.size() == vBatchImages.size());
            vecSize = vec_size;
            if (INPUT_W <= 0 || INPUT_H <= 0)
            {
                std::cout << "!!!!!!!!!!!!!!!!!!   json read error && init parameter error !!!!!!!!!!!" << std::endl;
            }
            preproGPU(vBatchImages);
            doinference();
            postprocess(arrDetection);
        }
        // void preprocessGPUImg(cv::Mat &oriImg)
        // {
        //     if (!oriImg.data || oriImg.cols <= 0 || oriImg.rows <= 0)
        //     {
        //         std::cout << "Unable to open image : " << std::endl;
        //         assert(0);
        //     }

        //     if (oriImg.channels() != 3)
        //     {
        //         std::cout << "Non RGB images are not supported : " << std::endl;
        //         assert(0);
        //     }
        //     if (oriImg.rows * oriImg.step * sizeof(uchar) > maxSize)
        //     {
        //         cudaFree(img_gpu_data8u);
        //         CUDA_CHECK(cudaMalloc((void **)&img_gpu_data8u, oriImg.rows * oriImg.step * sizeof(uchar)));
        //         maxSize = oriImg.rows * oriImg.step * sizeof(uchar);
        //     }
        //     cudaMemcpy(img_gpu_data8u, oriImg.data, oriImg.rows * oriImg.step * sizeof(uchar), cudaMemcpyHostToDevice);
        //     CudaImg<uchar3> gpu_src_image;
        //     gpu_src_image.csp(img_gpu_data32ftmp, img_gpu_data8u, INPUT_H, INPUT_W, oriImg.step);
        // }
        // void preprocessGPU(std::vector<cv::Mat> &vBatchImages)
        // {
        //     if (BATCH_SIZE != vBatchImages.size())
        //     {
        //         std::cout << "Batch size Error" << std::endl;
        //     }
        //     std::cout << "this is CUDA prepeocess " << std::endl;
        //     if (vBatchImages.size() <= 0)
        //     {
        //         std::cout << "Unable to open image : " << std::endl;
        //     }
        //     for (int i = 0; i < vBatchImages.size(); i++)
        //     {
        //         preprocessGPUImg(vBatchImages[i]);
        //         cudaMemcpyAsync(m_gpuInputBuffer + i * INPUT_W * INPUT_H * sizeof(float3), img_gpu_data32ftmp, INPUT_W * INPUT_H * sizeof(float3), cudaMemcpyDeviceToDevice, stream);
        //     }
        // }
        void preproGPU(std::vector<cv::Mat> &vBatchImages)
        {
            if (BATCH_SIZE != vBatchImages.size())
            {
                std::cout << "Batch size Error" << std::endl;
            }
            std::cout << "this is CUDA prepeocess " << std::endl;
            if (vBatchImages.size() <= 0)
            {
                std::cout << "Unable to open image : " << std::endl;
            }
            long addoff = 0;
            for (int n = 0; n < vBatchImages.size(); n++)
            {
                // std::cout << "step: " << vBatchImages[i].step << std::endl;
                cudaMemcpyAsync(m_gpuInput8U + n * INPUT_W * INPUT_H * sizeof(uchar3), vBatchImages[n].data, INPUT_W * INPUT_H * sizeof(uchar3), cudaMemcpyHostToDevice, stream);
                // for (int i = 0; i < vBatchImages[n].rows; i++)
                // {
                //     cudaMemcpyAsync(m_gpuInput8U + addoff + i * INPUT_W * sizeof(uchar3), vBatchImages[n].data + i * vBatchImages[n].step, INPUT_W * sizeof(uchar3), cudaMemcpyHostToDevice, stream);
                // }
                // addoff += INPUT_W * INPUT_H * sizeof(uchar3);
            }
        }

        void preprocess(std::vector<cv::Mat> &vBatchImages)
        {
            assert(vBatchImages.size() == BATCH_SIZE);
            for (int b = 0; b < BATCH_SIZE; ++b)
            {
                cv::Mat pr_img = vBatchImages[b];
                int i = 0;
                for (int row = 0; row < INPUT_H; ++row)
                {
                    uchar *uc_pixel = pr_img.data + row * pr_img.step;
                    for (int col = 0; col < INPUT_W; ++col)
                    {
                        dataP[b * 3 * INPUT_H * INPUT_W + i] = (float)uc_pixel[2] / 255.0;
                        dataP[b * 3 * INPUT_H * INPUT_W + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
                        dataP[b * 3 * INPUT_H * INPUT_W + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
                        uc_pixel += 3;
                        ++i;
                    }
                }
            }
        }

        void doinference()
        {
            // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
            // CHECK(cudaMemcpyAsync(buffers[0], dataP, BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));

            // CHECK(cudaMemcpyAsync(buffers[0], m_gpuInputBuffer, BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyDeviceToDevice, stream));

            CHECK(cudaMemcpyAsync(buffers[0], m_gpuInput8U, BATCH_SIZE * 3 * INPUT_H * INPUT_W, cudaMemcpyDeviceToDevice, stream));

            context->enqueue(BATCH_SIZE, buffers, stream, nullptr);
            CHECK(cudaMemcpyAsync(prob, buffers[1], BATCH_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
            cudaStreamSynchronize(stream);
        }

        void postprocess(std::vector<std::vector<DetectedObject>> &arrDetection)
        {
            for (int b = 0; b < BATCH_SIZE; ++b)
            {
                imgWidth = vecSize[b].width;
                imgHeight = vecSize[b].height;
                std::vector<DetectedObject> resVec;
                nms(resVec, &prob[b * OUTPUT_SIZE]);
                arrDetection.push_back(resVec);
            }
        }

        void nms(std::vector<DetectedObject> &resVec, float *probP)
        {
            std::map<int, std::vector<DetectedObject>> m;

            for (int i = 0; i < probP[0] && i < MAX_OUTPUT_BBOX; i++)
            {
                if (probP[1 + DETECTION_SIZE * i + 4] <= conf_thresh)
                    continue;
                DetectionCsp det;
                DetectedObject detObj;

                memcpy(&det, &probP[1 + DETECTION_SIZE * i], DETECTION_SIZE * sizeof(float));
                // std::cout << "det:" << det.det_confidence << ",class_id: " << det.class_id << std::endl;
                // std::cout << "bbox: " << det.bbox[0] << "_" << det.bbox[1] << "_" << det.bbox[2] << "_" << det.bbox[3] << "_" << std::endl;
                to_DetectedObject(det, detObj);
                if (m.count(detObj.object_class) == 0)
                    m.emplace(detObj.object_class, std::vector<DetectedObject>());
                m[detObj.object_class].push_back(detObj);
            }

            for (auto it = m.begin(); it != m.end(); it++)
            {
                // std::cout << it->second[0].class_id << " --- " << std::endl;
                auto &dets = it->second;
                std::sort(dets.begin(), dets.end(), cmp);
                for (size_t j = 0; j < dets.size(); ++j)
                {
                    auto &item = dets[j];
                    resVec.push_back(item);
                    for (size_t n = j + 1; n < dets.size(); ++n)
                    {
                        if (iou(item.bounding_box, dets[n].bounding_box) > nms_thresh)
                        {
                            dets.erase(dets.begin() + n);
                            --n;
                        }
                    }
                }
            }
        }

        void to_DetectedObject(DetectionCsp &d, DetectedObject &dd)
        {
            dd.object_class = (int)d.class_id;
            dd.prob = d.class_confidence;
            dd.bounding_box = get_rect_letterbox(d.bbox);
        }

        //映射回原图的结果
        cv::Rect get_rect(float bbox[4])
        {
            int l, r, t, b;
            float r_w = INPUT_W / (imgWidth * 1.0);
            float r_h = INPUT_H / (imgHeight * 1.0);

            l = bbox[0] - bbox[2] / 2.f;
            r = bbox[0] + bbox[2] / 2.f;
            t = bbox[1] - bbox[3] / 2.f;
            b = bbox[1] + bbox[3] / 2.f;
            l = l / r_w;
            r = r / r_w;
            t = t / r_h;
            b = b / r_h;

            return cv::Rect(l, t, r - l, b - t);
        }

        //映射回原图的结果
        cv::Rect get_rect_letterbox(float bbox[4])
        {
            int l, r, t, b;
            float r_w = INPUT_W / (imgWidth * 1.0);
            float r_h = INPUT_H / (imgHeight * 1.0);
            if (r_h > r_w)
            {
                l = bbox[0] - bbox[2] / 2.f;
                r = bbox[0] + bbox[2] / 2.f;
                t = bbox[1] - bbox[3] / 2.f - (INPUT_H - r_w * imgHeight) / 2;
                b = bbox[1] + bbox[3] / 2.f - (INPUT_H - r_w * imgHeight) / 2;
                l = l / r_w;
                r = r / r_w;
                t = t / r_w;
                b = b / r_w;
            }
            else
            {
                l = bbox[0] - bbox[2] / 2.f - (INPUT_W - r_h * imgWidth) / 2;
                r = bbox[0] + bbox[2] / 2.f - (INPUT_W - r_h * imgWidth) / 2;
                t = bbox[1] - bbox[3] / 2.f;
                b = bbox[1] + bbox[3] / 2.f;
                l = l / r_h;
                r = r / r_h;
                t = t / r_h;
                b = b / r_h;
            }
            return cv::Rect(l, t, r - l, b - t);
        }

        float iou(cv::Rect &lbox, cv::Rect &rbox)
        {
            float interBox[] = {

                std::max(lbox.x, rbox.x),                             // left
                std::min(lbox.x + lbox.width, rbox.x + rbox.width),   // right
                std::max(lbox.y, rbox.y),                             // top
                std::min(lbox.y + lbox.height, rbox.y + rbox.height), // bottom
            };

            if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
                return 0.0f;

            float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
            return interBoxS / (lbox.width * lbox.height + rbox.width * rbox.height - interBoxS);
        }
    };

    Yolov4Csp::Yolov4Csp(std::string weightspath, std::string jsonpath, std::string cfgpath, int gpuid)
    {
        m_pHandleYolov4Csp = std::make_shared<Yolov4CspPrivate>(weightspath, jsonpath, cfgpath, gpuid);
    }

    Yolov4Csp::Yolov4Csp(std::string weightspath, std::string jsonpath, int gpuid, float thresh)
    {
        m_pHandleYolov4Csp = std::make_shared<Yolov4CspPrivate>(weightspath, jsonpath, gpuid, thresh);
    }

    void Yolov4Csp::detect(std::vector<cv::Mat> &vBatchImages, std::vector<std::vector<DetectedObject>> &arrDetection, std::vector<cv::Size> &vec_size)
    {
        m_pHandleYolov4Csp->detect(vBatchImages, arrDetection, vec_size);
    }

    Yolov4Csp::~Yolov4Csp()
    {
        //
    }
};
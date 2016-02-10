//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "ImageDataDeserializer.h"
#include "ImageConfigHelper.h"

#ifndef UNREFERENCED_PARAMETER
#define UNREFERENCED_PARAMETER(P) (P)
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

class ImageDataDeserializer::LabelGenerator
{
public:
    virtual void CreateLabelFor(size_t classId, SparseSequenceData& data) = 0;
    virtual ~LabelGenerator() { }
};

// A helper class to generate a typed label in a sparse format.
// A label is just a category/class the image belongs to.
// It is represented as a array indexed by the category with zero values for all categories the image does not belong to, 
// and a single one for a category it belongs to: [ 0, .. 0.. 1 .. 0 ]
// The class is parameterized because the representation of 1 is type specific.
template <class TElement>
class TypedLabelGenerator : public ImageDataDeserializer::LabelGenerator
{
public:
    TypedLabelGenerator() : m_value(1)
    {
    }

    virtual void CreateLabelFor(size_t classId, SparseSequenceData& data) override
    {
        data.m_indices.resize(1);
        data.m_indices[0] = std::vector<size_t>{ classId };
        data.m_data = &m_value;
    }

private:
    TElement m_value;
};

ImageDataDeserializer::ImageDataDeserializer(const ConfigParameters& config)
{
    ImageConfigHelper configHelper(config);
    m_streams = configHelper.GetStreams();
    //assert(m_streams.size() == 2);
    labelIds = configHelper.GetLabelStreamIds();
    featureIds = configHelper.GetFeatureStreamIds();
    //const auto& label = m_streams[];
    //const auto& feature = m_streams[];

    // Expect data in HWC.
    for(auto idx : featureIds)
    {
        const auto& feature = m_streams[idx];
        ImageDimensions dimensions(*feature->m_sampleLayout, configHelper.GetDataFormat());
        feature->m_sampleLayout = std::make_shared<TensorShape>(dimensions.AsTensorShape(HWC));
        feature->m_storageType = StorageType::dense;
        m_featureElementType = feature->m_elementType;
    }
    int labelDimension = -1;
    for(auto idx : labelIds)
    {
        const auto& label = m_streams[idx];
        label->m_storageType = StorageType::sparse_csc;
        if(labelDimension == -1)
            labelDimension = static_cast<int>(label->m_sampleLayout->GetDim(0));
        else if(labelDimension != label->m_sampleLayout->GetDim(0))
            RuntimeError("All label dimensions must be the same currently");
        if (label->m_elementType == ElementType::tfloat)
        {
            m_labelGenerator.push_back(std::make_shared<TypedLabelGenerator<float>>());
        }
        else if (label->m_elementType == ElementType::tdouble)
        {
            m_labelGenerator.push_back(std::make_shared<TypedLabelGenerator<double>>());
        }
        else
        {
            RuntimeError("Unsupported label element type '%d'.", label->m_elementType);
        }
    }

    

    CreateSequenceDescriptions(configHelper.GetMapPath(), labelDimension);
}

void ImageDataDeserializer::CreateSequenceDescriptions(std::string mapPath, size_t labelDimension)
{
    UNREFERENCED_PARAMETER(labelDimension);

    std::ifstream mapFile(mapPath);
    if (!mapFile)
    {
        RuntimeError("Could not open %s for reading.", mapPath.c_str());
    }

    std::string line;
    if(labelIds.size() > featureIds.size())
    {
        RuntimeError("Currently cannot support more labels ('%d') than features('%d')", labelIds.size(), featureIds.size());
    }
    
    for (size_t lineIndex = 0; std::getline(mapFile, line); ++lineIndex)
    {
        std::stringstream ss(line);
        std::string imagePath;
        std::string classId;
        
        std::vector<ImageSequenceDescription> descriptions;
        ImageSequenceDescription description;
        description.m_numberOfSamples = 1;
        description.m_isValid = true;
        for(int i = 0; i < featureIds.size(); ++i)
        {
            if(!std::getline(ss, imagePath, '\t'))
            {
            
            }
            description.m_id = lineIndex;
            description.m_chunkId = lineIndex;
            description.m_path = imagePath;
            descriptions.push_back(description);
        }
        for(int i = 0; i < labelIds.size(); ++i)
        {
            if(!std::getline(ss, classId, '\t'))
            {
            
            }
            descriptions[i].m_classId = std::stoi(classId);
            if (descriptions[i].m_classId >= labelDimension)
            {
                RuntimeError(
                    "Image '%s' has invalid class id '%d'. Expected label dimension is '%d'.",
                    mapPath.c_str(),
                    static_cast<int>(descriptions[i].m_classId),
                    static_cast<int>(labelDimension));
            }
        }

        /*if (!std::getline(ss, imagePath, '\t') || )
        {
            RuntimeError("Invalid map file format, must contain 2 tab-delimited columns: %s, line: %d.",
                         mapPath.c_str(),
                         static_cast<int>(lineIndex));
        }*/
        m_imageSequences.push_back(descriptions);
    }
}

std::vector<StreamDescriptionPtr> ImageDataDeserializer::GetStreamDescriptions() const
{
    return m_streams;
}

std::vector<std::vector<SequenceDataPtr>> ImageDataDeserializer::GetSequencesById(const std::vector<size_t>& ids)
{
    if (ids.empty())
    {
        RuntimeError("Number of requested sequences cannot be zero.");
    }

    m_currentImages.resize(ids.size(), std::vector<cv::Mat>(featureIds.size()));
    m_labels.resize(ids.size(), std::vector<SparseSequenceDataPtr>(labelIds.size()));

    std::vector<std::vector<SequenceDataPtr>> result;
    result.resize(ids.size());

#pragma omp parallel for ordered schedule(dynamic)
    for (int i = 0; i < ids.size(); ++i)
    {
        if (ids[i] >= m_imageSequences.size())
        {
            RuntimeError("Invalid sequence id is provided '%d', expected range [0..%d].",
                         static_cast<int>(ids[i]),
                         static_cast<int>(m_imageSequences.size()) - 1);
        }

        const auto& imageSequence = m_imageSequences[ids[i]];
        int j = 0;
        for(const auto& imageSubSequence: imageSequence)
        {
            // Construct image
            m_currentImages[i][j] = std::move(cv::imread(imageSubSequence.m_path, cv::IMREAD_COLOR));
            cv::Mat& cvImage = m_currentImages[i][j];    
            
            if (!cvImage.data)
            {
                RuntimeError("Cannot open file '%s'", imageSubSequence.m_path.c_str());
            }
            // Convert element type.
            // TODO We should all native CV element types to be able to match the behavior of the old reader.
            int dataType = m_featureElementType == ElementType::tfloat ? CV_32F : CV_64F;
            if (cvImage.type() != CV_MAKETYPE(dataType, cvImage.channels()))
            {
                cvImage.convertTo(cvImage, dataType);
            }

            if (!cvImage.isContinuous())
            {
                cvImage = cvImage.clone();
            }
            assert(cvImage.isContinuous());
            ImageDimensions dimensions(cvImage.cols, cvImage.rows, cvImage.channels());
            auto image = std::make_shared<DenseSequenceData>();
            image->m_data = cvImage.data;
            image->m_sampleLayout = std::make_shared<TensorShape>(dimensions.AsTensorShape(HWC));
            image->m_numberOfSamples = 1;
            if(j < m_labels[i].size())
            {
                if (m_labels[i][j] == nullptr)
                {
                    m_labels[i][j] = std::make_shared<SparseSequenceData>();
                }
                m_labelGenerator[j]->CreateLabelFor(imageSubSequence.m_classId, *m_labels[i][j]);
                result[i] = std::move(std::vector<SequenceDataPtr>{image, m_labels[i][j]});
            }            
            ++j;
        }
    }
    return result;
}

void ImageDataDeserializer::FillSequenceDescriptions(SequenceDescriptions& timeline) const
{
    timeline.resize(m_imageSequences.size());
    std::transform(
        m_imageSequences.begin(),
        m_imageSequences.end(),
        timeline.begin(),
        [](const std::vector<ImageSequenceDescription>& desc)
        {
            return &desc[0];
        });
}

}}}
